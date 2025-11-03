import warnings
warnings.filterwarnings('ignore')

import os
import json
import streamlit as st
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from duckduckgo_search import DDGS
from mistralai import Mistral
from mistralai.models import TextChunk, ReferenceChunk
from docling.document_converter import DocumentConverter, InputFormat
from dotenv import load_dotenv
import chromadb

load_dotenv(override=False)

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    MAX_SEARCH_RESULTS: int = 5
    TOP_K_ARTICLES: int = 3
    CACHE_TTL: int = 300
    MODEL_OPTIONS: List[str] = None
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "news_articles"
    
    def __post_init__(self):
        if self.MODEL_OPTIONS is None:
            self.MODEL_OPTIONS = [
                "mistral-small-latest",
                "mistral-medium-latest",
                "mistral-large-latest",
            ]

CONFIG = Config()

# ============================================================================
# DATA MODEL
# ============================================================================
@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    url: str = ""
    date: str = ""

# ============================================================================
# DOCUMENT CONVERTER
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_document_converter() -> DocumentConverter:
    """Get or create DocumentConverter instance."""
    return DocumentConverter(allowed_formats=[InputFormat.HTML])

# ============================================================================
# MISTRAL CLIENT
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_mistral_client() -> Optional[Mistral]:
    """Get or create Mistral client instance."""
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    return Mistral(api_key=api_key) if api_key else None

def get_system_prompt():
    return "You are a helpful assistant for news analysis. You have three tools: 'news_search' for web searches, 'local_news_search' for searching the local database, and 'deep_analysis' for comprehensive analysis when users ask for detailed, in-depth, or thorough analysis. Read the output turned by the tool and provide final answer to the user."

# ============================================================================
# CHROMADB CLIENT
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_chroma_client():
    """Get or create ChromaDB client and collection."""
    try:
        client_db = chromadb.PersistentClient(path=CONFIG.CHROMA_DB_PATH)
        collection = client_db.get_collection(name=CONFIG.CHROMA_COLLECTION_NAME)
        return client_db, collection
    except Exception as e:
        st.warning(f"ChromaDB not available: {str(e)}")
        return None, None

# ============================================================================
# LOCAL NEWS SEARCH ENGINE
# ============================================================================
class LocalNewsSearchEngine:
    """Handles searches against local ChromaDB collection."""
    
    def __init__(self, mistral_client: Mistral, collection):
        self.mistral_client = mistral_client
        self.collection = collection
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Mistral."""
        try:
            embeddings_batch_response = self.mistral_client.embeddings.create(
                model="mistral-embed",
                inputs=[text]
            )
            return embeddings_batch_response.data[0].embedding
        except Exception as e:
            st.warning(f"Embedding error: {str(e)}")
            return []
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search local ChromaDB collection."""
        if not self.collection:
            return []
        
        try:
            query_embedding = self.get_text_embedding(query)
            if not query_embedding:
                return []
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            articles = []
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                articles.append({
                    "content": results['documents'][0][i],
                    "metadata": metadata,
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "title": metadata.get("title", "Local Article"),
                    "url": metadata.get("url", ""),
                    "date": metadata.get("date", ""),
                    "source": metadata.get("source", "Local DB"),
                })
            
            return articles
        except Exception as e:
            st.warning(f"Local search error: {str(e)}")
            return []

# ============================================================================
# WEB SEARCH ENGINE
# ============================================================================
class WebSearchEngine:
    """Handles DuckDuckGo news searches with caching."""
    
    @staticmethod
    @st.cache_data(show_spinner=False, ttl=CONFIG.CACHE_TTL)
    def search_news(query: str, num_results: int) -> List[Dict]:
        """Cached DDG news search."""
        try:
            ddgs = DDGS()
            return list(ddgs.news(query, max_results=num_results) or [])
        except Exception as e:
            st.warning(f"Search error: {str(e)}")
            return []
    
    @staticmethod
    def parse_to_articles(results: List[Dict]) -> List[NewsArticle]:
        """Convert raw search results to NewsArticle objects."""
        return [
            NewsArticle(
                title=item.get("title", ""),
                content=item.get("body", ""),
                source=item.get("source", "Web"),
                url=item.get("url", ""),
                date=item.get("date", ""),
            )
            for item in results
        ]
    
    def search(self, query: str, num_results: int = 5) -> List[NewsArticle]:
        """Search and parse in one call."""
        results = self.search_news(query, num_results)
        return self.parse_to_articles(results)

# ============================================================================
# NEWS CONTENT EXTRACTOR
# ============================================================================
@st.cache_data(show_spinner=False, ttl=CONFIG.CACHE_TTL)
def extract_news_content(url: str) -> Optional[str]:
    """Extract full content from a news URL using Docling."""
    try:
        converter = get_document_converter()
        result = converter.convert(url)
        return result.document.export_to_markdown()
    except Exception as e:
        return f"Error extracting content: {str(e)}"

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================
def get_news_search_tool_schema() -> dict:
    """Returns the news_search tool schema for Mistral."""
    return {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": "Search the web for recent news related to a query and return structured references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search the news in keyword form.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max number of results to return.",
                        "default": CONFIG.TOP_K_ARTICLES,
                    },
                },
                "required": ["query"],
            },
        },
    }

def get_local_news_search_tool_schema() -> dict:
    """Returns the local_news_search tool schema for Mistral."""
    return {
        "type": "function",
        "function": {
            "name": "local_news_search",
            "description": "Search local news database for relevant articles using semantic search. Use this when user asks about local/saved articles or wants to search the local database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search local news database.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max number of results to return.",
                        "default": CONFIG.TOP_K_ARTICLES,
                    },
                },
                "required": ["query"],
            },
        },
    }

def get_deep_analysis_tool_schema() -> dict:
    """Returns the deep_analysis tool schema for Mistral."""
    return {
        "type": "function",
        "function": {
            "name": "deep_analysis",
            "description": "Perform deep analysis by reading full content of all found news articles. Use this when user explicitly asks for detailed, in-depth, or comprehensive analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news topic to analyze in depth.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of articles to analyze.",
                        "default": CONFIG.TOP_K_ARTICLES,
                    },
                },
                "required": ["query"],
            },
        },
    }

def execute_local_news_search(query: str, top_k: int = 5) -> str:
    """Execute local news search and return JSON-formatted results."""
    mistral_client = get_mistral_client()
    _, collection = get_chroma_client()
    
    if not mistral_client or not collection:
        return json.dumps({}, ensure_ascii=False)
    
    engine = LocalNewsSearchEngine(mistral_client, collection)
    
    with st.spinner(f"🔍 Searching local database for: {query}"):
        articles = engine.search(query, n_results=top_k)
    
    data = {}
    for idx, art in enumerate(articles):
        # Extract snippets from content
        snippets = []
        content = art.get("content", "")
        if content:
            sentences = [s.strip() for s in content.split('. ') if s.strip()]
            snippets = sentences[:2]  # First 2 sentences
        
        data[str(idx)] = {
            "url": art.get("url") or None,
            "title": art.get("title") or None,
            "snippets": [snippets],
            "description": None,
            "date": art.get("date") or None,
            "source": art.get("source", "Local DB"),
            "distance": art.get("distance"),  # Similarity score
        }
    
    return json.dumps(data, ensure_ascii=False)

def execute_news_search(query: str, top_k: int = 5) -> str:
    """Execute news search and return JSON-formatted results."""
    engine = WebSearchEngine()
    articles = engine.search(query, num_results=top_k)
    
    data = {}
    for idx, art in enumerate(articles):
        snippets = []
        if art.content:
            sentences = [s.strip() for s in art.content.split('. ') if s.strip()]
            snippets = sentences[:2]
        
        data[str(idx)] = {
            "url": art.url or None,
            "title": art.title or None,
            "snippets": [snippets],
            "description": None,
            "date": art.date or None,
            "source": art.source or "web",
        }
    
    return json.dumps(data, ensure_ascii=False)

def execute_deep_analysis(query: str, top_k: int = 3, articles: List[NewsArticle] = None) -> str:
    """Execute deep analysis by fetching and reading full article content.
    
    Args:
        query: Search query (used only if articles not provided)
        top_k: Number of articles to fetch (used only if articles not provided)
        articles: Pre-fetched articles to analyze (avoids duplicate search)
    """
    # Only search if articles not provided
    if articles is None:
        engine = WebSearchEngine()
        with st.spinner(f"🔍 Searching for news about: {query}"):
            articles = engine.search(query, num_results=top_k)
    
    if not articles:
        return json.dumps({}, ensure_ascii=False)
    
    # Create progress container
    progress_placeholder = st.empty()
    data = {}
    
    for idx, art in enumerate(articles):
        # Show progress for each article
        title_preview = art.title[:60] + ('...' if len(art.title) > 60 else '')
        progress_placeholder.info(f"📖 Reading article {idx + 1}/{len(articles)}: **{title_preview}**")
        
        # Extract full content from the article
        full_content = None
        if art.url:
            full_content = extract_news_content(art.url)
        
        data[str(idx)] = {
            "url": art.url or None,
            "title": art.title or None,
            "date": art.date or None,
            "source": art.source or "web",
            "full_content": full_content or art.content,  # Fallback to snippet if extraction fails
        }
    
    # Clear progress message after completion
    progress_placeholder.success(f"✅ Completed deep analysis of {len(articles)} articles")
    
    return json.dumps(data, ensure_ascii=False)

# ============================================================================
# CHAT HANDLING
# ============================================================================
class ChatHandler:
    """Manages Mistral chat interactions."""
    
    def __init__(self, client: Mistral):
        self.client = client
    
    def simple_chat(self, model: str, messages: List[dict], temperature: float) -> dict:
        """Simple chat without tools."""
        resp = self.client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        choice = resp.choices[0]
        msg = choice.message if hasattr(choice, "message") else choice
        
        return {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", "")
        }
    
    def chat_with_tools(
        self, 
        model: str, 
        messages: List[dict], 
        temperature: float,
        top_k: int,
        progress_placeholder=None
    ) -> Tuple[dict, List[dict], List[dict]]:
        """
        Chat with tool usage capability.
        Returns: (final_message, sources_list)
        """
        tool_schemas = [
            get_news_search_tool_schema(),
            get_local_news_search_tool_schema(),
            get_deep_analysis_tool_schema()
        ]
        
        # Initial call
        initial = self.client.chat.complete(
            model=model,
            messages=messages,
            tools=tool_schemas,
            temperature=temperature,
        )
        
        msg0 = initial.choices[0].message
        sources = []
        tool_logs: List[dict] = []
        
        # Check for tool calls
        if hasattr(msg0, "tool_calls") and msg0.tool_calls:
            tool_call = msg0.tool_calls[0]
            tool_name = getattr(tool_call.function, "name", "")
            
            if tool_name in ["news_search", "deep_analysis", "local_news_search"]:
                # Execute appropriate tool
                args = json.loads(tool_call.function.arguments or "{}")
                query = args.get("query", "")
                tk = int(args.get("top_k", top_k))
                # Live progress: show selected tool and args
                # if progress_placeholder is not None:
                #     progress_placeholder.empty()
                #     with progress_placeholder.container():
                #         st.markdown(f"**Running tool:** `{tool_name}`")
                #         st.markdown("- **Input arguments**:")
                #         st.json(args)
                #         st.caption("Executing...")
                
                # Execute tool based on name
                if tool_name == "local_news_search":
                    wb_result = execute_local_news_search(query, tk)
                elif tool_name == "deep_analysis":
                    # Perform search once and reuse results
                    engine = WebSearchEngine()
                    with st.spinner(f"🔍 Searching for news about: {query}"):
                        articles = engine.search(query, num_results=tk)
                    wb_result = execute_deep_analysis(query, tk, articles=articles)
                else:  # news_search
                    wb_result = execute_news_search(query, tk)
                # Record tool call log (input/outputs)
                try:
                    parsed_output = json.loads(wb_result)
                except Exception:
                    parsed_output = {"raw": wb_result}
                tool_logs.append({
                    "name": tool_name,
                    "arguments": args,
                    "output": parsed_output,
                })
                # Live progress: update with output
                # if progress_placeholder is not None:
                #     progress_placeholder.empty()
                #     with progress_placeholder.container():
                #         st.markdown(f"**Tool finished:** `{tool_name}`")
                #         st.markdown("- **Input arguments**:")
                #         st.json(args)
                #         st.markdown("- **Output**:")
                #         st.json(parsed_output)
                
                # Update message history
                messages.append({
                    "role": getattr(msg0, "role", "assistant"),
                    "content": "",
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tool_call.id,
                    "content": wb_result,
                })
                
                # Final call with context
                final = self.client.chat.complete(
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    temperature=temperature,
                )
                final_msg = final.choices[0].message
                
                # Parse sources
                ref_map = json.loads(wb_result)
                sources = [
                    {
                        "url": ref_map[k].get("url"),
                        "title": ref_map[k].get("title"),
                        "date": ref_map[k].get("date"),
                        "source": ref_map[k].get("source"),
                    }
                    for k in sorted(ref_map.keys())
                ]
                
                # Extract referenced sources if model provides them
                content = getattr(final_msg, "content", "")
                refs_used = []
                
                if isinstance(content, list):
                    text_parts = []
                    for chunk in content:
                        if isinstance(chunk, TextChunk):
                            text_parts.append(getattr(chunk, "text", ""))
                        elif isinstance(chunk, ReferenceChunk):
                            refs_used.extend(getattr(chunk, "reference_ids", []))
                    content = "".join(text_parts)
                
                # Filter to only referenced sources if available
                if refs_used:
                    sources = [
                        sources[ref_id] 
                        for ref_id in sorted(set(refs_used)) 
                        if ref_id < len(sources)
                    ]
                
                return {
                    "role": getattr(final_msg, "role", "assistant"),
                    "content": content
                }, sources, tool_logs
        
        # No tool usage
        return {
            "role": getattr(msg0, "role", "assistant"),
            "content": getattr(msg0, "content", "")
        }, sources, tool_logs

# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_styles():
    """Inject custom CSS."""
    st.markdown("""
        <style>
        .sidebar-card {
            background: #fff;
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .sidebar-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .sidebar-title {
            font-size: 14px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 4px;
        }
        .sidebar-meta {
            font-size: 12px;
            color: #666;
        }
        .local-badge {
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            margin-left: 4px;
        }
        a.sidebar-link {
            text-decoration: none;
            color: inherit;
        }
        a.sidebar-link:hover .sidebar-title {
            color: #1f77b4;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sources(sources: List[dict]):
    """Render sources in sidebar."""
    st.markdown("### 📚 Sources")
    st.markdown("---")
    
    if sources:
        for ref in sources:
            url = ref.get("url") or "#"
            title = ref.get("title") or "Source"
            date = ref.get("date") or ""
            src = ref.get("source") or "web"
            
            # Add badge for local sources
            is_local = src == "Local DB"
            local_badge = '<span class="local-badge">LOCAL</span>' if is_local else ""
            
            title_display = title[:60] + ('...' if len(title) > 60 else '')
            
            st.markdown(f"""
                <a href="{url}" target="_blank" class="sidebar-link">
                    <div class="sidebar-card">
                        <div class="sidebar-title">{title_display}{local_badge}</div>
                        <div class="sidebar-meta">{date} — {src}</div>
                    </div>
                </a>
            """, unsafe_allow_html=True)
    else:
        st.markdown("*Sources from searches will appear here*")

def render_tool_details(tool_logs: List[dict]):
    """Render an expandable panel showing tool calls, inputs, and outputs."""
    if not tool_logs:
        return
    with st.expander("🔍 Tool calls: inputs and outputs", expanded=False):
        for idx, log in enumerate(tool_logs):
            st.markdown(f"**Tool {idx + 1}: {log.get('name','')}**")
            st.markdown("- **Input arguments**:")
            st.json(log.get("arguments", {}))
            st.markdown("- **Output**:")
            st.json(log.get("output", {}))
            st.markdown("---")

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": get_system_prompt()}
        ]
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "assistant_logs" not in st.session_state:
        # List aligned with final assistant messages, each item is list of tool-call logs
        st.session_state.assistant_logs = []
    if "show_tool_details" not in st.session_state:
        st.session_state.show_tool_details = False

def render_chat_history():
    """Render chat message history."""
    assistant_index = 0
    for msg in st.session_state.messages:
        role = msg.get("role")
        
        # Skip system, tool messages, and empty assistant messages
        if role in ("system", "tool"):
            continue
        if role == "assistant" and msg.get("tool_calls"):
            continue
        
        content = msg.get("content", "")
        if not content:
            continue
        
        with st.chat_message(role):
            st.markdown(content)
            # If this is a final assistant message, optionally render tool logs for this turn
            if role == "assistant" and st.session_state.get("show_tool_details", False):
                logs_for_turn = []
                if assistant_index < len(st.session_state.get("assistant_logs", [])):
                    logs_for_turn = st.session_state.assistant_logs[assistant_index]
                # Render per-message tool details (even if empty, show a small note)
                if logs_for_turn:
                    render_tool_details(logs_for_turn)
                else:
                    with st.expander("🔍 Tool calls: inputs and outputs", expanded=False):
                        st.caption("No tool calls used for this response.")
                assistant_index += 1

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="News + Chat (Mistral)",
        page_icon="📰",
        layout="wide"
    )
    
    render_styles()
    init_session_state()
    
    # Check ChromaDB availability
    _, collection = get_chroma_client()
    chroma_available = collection is not None
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        model = st.selectbox(
            "Model",
            options=CONFIG.MODEL_OPTIONS,
            index=2,
            help="Choose a Mistral model",
        )
        
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.7, 0.05,
            help="Higher = more creative"
        )
        
        top_k = st.slider(
            "Max Sources",
            1, 10, CONFIG.TOP_K_ARTICLES, 1,
            help="Number of news sources to fetch"
        )
        
        tools_enabled = st.checkbox(
            "Enable Tools",
            value=True,
            help="Allow model to use news search and deep analysis tools"
        )
        st.session_state.show_tool_details = st.checkbox(
            "Show tool details (inputs/outputs)",
            value=st.session_state.show_tool_details,
            help="Display tool calls, input arguments, and outputs for each response"
        )
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "system", "content": get_system_prompt()}
            ]
            st.session_state.sources = []
            st.session_state.assistant_logs = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Available Tools")
        st.markdown("""
        - **News Search**: Quick web search with snippets
        - **Local News Search**: Search local database with semantic search
        - **Deep Analysis**: Reads full articles for comprehensive analysis
        """)
        
        if chroma_available:
            st.success("✅ Local database connected")
        else:
            st.warning("⚠️ Local database not available")
        
        st.info("💡 Try asking for 'detailed analysis' or 'search local database' to trigger specialized tools!")
        st.markdown("---")
        st.markdown("""
        - Latest news about Nvidia
        """
        )
        
        st.markdown("---")
        render_sources(st.session_state.sources)
        st.markdown("---")
    
    # Main content
    st.title("📰 NewsLab Agent")
    st.caption("Chat with Mistral AI — searches news and local database, provides sourced answers with deep analysis")

    # Tool details are shown per assistant message below each response
    
    client = get_mistral_client()
    if not client:
        st.warning("⚠️ MISTRAL_API_KEY not found. Set it in your environment or .env file.")
        st.stop()
    
    # Render chat history
    render_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask about recent news or search local database..."):
        # Clear sources for new query
        st.session_state.sources = []
        # Do not clear assistant_logs; persist across turns
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    handler = ChatHandler(client)
                    
                    if tools_enabled and model.endswith("latest"):
                        # Chat with tools
                        # live_tool_placeholder = None
                        # if st.session_state.get("show_tool_details", False):
                        #     # Render a live, in-progress expander for this turn
                        #     live_tool_placeholder = st.empty()
                        reply, sources, tool_logs = handler.chat_with_tools(
                            model,
                            st.session_state.messages,
                            temperature,
                            top_k
                            # , progress_placeholder=live_tool_placeholder
                        )
                        st.session_state.sources = sources
                        st.session_state.assistant_logs.append(tool_logs or [])
                    else:
                        # Simple chat
                        reply = handler.simple_chat(
                            model,
                            st.session_state.messages,
                            temperature
                        )
                        st.session_state.assistant_logs.append([])
                        # Fallback: fetch sources anyway
                        try:
                            result = execute_news_search(prompt, top_k)
                            ref_map = json.loads(result)
                            st.session_state.sources = [
                                {
                                    "url": ref_map[k].get("url"),
                                    "title": ref_map[k].get("title"),
                                    "date": ref_map[k].get("date"),
                                    "source": ref_map[k].get("source"),
                                }
                                for k in sorted(ref_map.keys())
                            ]
                        except Exception:
                            pass
                    
                    st.session_state.messages.append(reply)
                    st.markdown(reply["content"])
                    
                    # Force rerun to update sidebar with new sources
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()