import json
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict
import hashlib
import numpy as np
import os
from dotenv import load_dotenv
from mistralai import Mistral

# Load environment variables from .env file
load_dotenv(override=False)

# Initialize Mistral client with API key validation
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key or api_key.strip() == "":
    raise ValueError(
        "MISTRAL_API_KEY environment variable is not set or is empty!\n"
        "Please set it in your .env file: MISTRAL_API_KEY=your-api-key\n"
        "Or export it: export MISTRAL_API_KEY='your-api-key'"
    )

#print(f"API Key loaded: {api_key[:10]}...{api_key[-4:]}")  # Show partial key for verification
client = Mistral(api_key=api_key)

def get_text_embedding(input):
    """Get embedding from Mistral API"""
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=[input] if isinstance(input, str) else input
    )
    return embeddings_batch_response.data[0].embedding

def create_document_id(record: Dict, index: int) -> str:
    """Create a unique ID for each document using link or content hash with index fallback"""
    if record.get('link'):
        # Use hash of link for consistent IDs
        doc_id = hashlib.md5(record['link'].encode()).hexdigest()
    else:
        # Fallback to content hash
        content = record.get('content', '')
        doc_id = hashlib.md5(content.encode()).hexdigest()
    
    # Add index to ensure uniqueness even for duplicates
    return f"{doc_id}_{index}"

def prepare_documents(json_file: str) -> tuple:
    """
    Prepare documents, metadata, and IDs from JSON file
    Returns: (documents, metadatas, ids)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    seen_ids = set()
    duplicates_removed = 0
    
    for index, record in enumerate(data):
        # Skip if no content
        if not record.get('content'):
            continue
        
        # Create unique ID
        doc_id = create_document_id(record, index)
        
        # Skip duplicates (shouldn't happen now, but just in case)
        if doc_id in seen_ids:
            duplicates_removed += 1
            continue
        seen_ids.add(doc_id)
        
        # Main document text - combine title and content for better context
        doc_text = ""
        if record.get('title'):
            doc_text += f"Title: {record['title']}\n\n"
        doc_text += record['content']
        
        documents.append(doc_text)
        
        # Metadata - all the additional fields (ChromaDB doesn't accept None values)
        metadata = {}
        
        # Only add non-empty metadata fields
        if record.get('title'):
            metadata['title'] = record['title']
        if record.get('link'):
            metadata['link'] = record['link']
        if record.get('source_id'):
            metadata['source_id'] = record['source_id']
        if record.get('pubDate'):
            metadata['pub_date'] = record['pubDate']
        if record.get('description'):
            metadata['description'] = record['description']
        
        # Handle list fields
        if record.get('creator'):
            creators = ', '.join(record['creator'])
            if creators:
                metadata['creator'] = creators
        
        if record.get('keywords'):
            keywords = ', '.join(record['keywords'])
            if keywords:
                metadata['keywords'] = keywords
        
        # Add image_url if present
        if record.get('image_url'):
            metadata['image_url'] = record['image_url']
        
        metadatas.append(metadata)
        ids.append(doc_id)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate documents")
    
    return documents, metadatas, ids

def get_embeddings_batch(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Get embeddings for multiple texts in batches
    Mistral API can handle multiple inputs at once
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Getting embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Get embeddings for the batch
        embeddings_response = client.embeddings.create(
            model="mistral-embed",
            inputs=batch
        )
        
        # Extract embeddings
        batch_embeddings = [item.embedding for item in embeddings_response.data]
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

def load_to_chromadb(
    json_file: str,
    collection_name: str = "news_articles",
    persist_directory: str = "./chroma_db",
    batch_size: int = 10
):
    """
    Load JSON data into ChromaDB using Mistral embeddings
    
    Args:
        json_file: Path to filtered JSON file
        collection_name: Name for the ChromaDB collection
        persist_directory: Directory to persist the database
        batch_size: Number of documents to embed at once
    """
    
    # Initialize ChromaDB client with persistence
    client_db = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get collection (without default embedding function)
    try:
        client_db.delete_collection(name=collection_name)
        print(f"Deleted existing collection '{collection_name}'")
    except:
        pass
    
    # Create collection with cosine similarity
    collection = client_db.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created new collection '{collection_name}'")
    
    # Prepare data
    print("\nPreparing documents...")
    documents, metadatas, ids = prepare_documents(json_file)
    print(f"Found {len(documents)} documents")
    
    # Generate embeddings using Mistral
    print("\nGenerating embeddings with Mistral...")
    embeddings = get_embeddings_batch(documents, batch_size=batch_size)
    
    # Add documents with embeddings to ChromaDB
    print(f"\nLoading {len(documents)} documents into ChromaDB...")
    
    # Add in batches
    db_batch_size = 100
    for i in range(0, len(documents), db_batch_size):
        batch_docs = documents[i:i+db_batch_size]
        batch_meta = metadatas[i:i+db_batch_size]
        batch_ids = ids[i:i+db_batch_size]
        batch_embeddings = embeddings[i:i+db_batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids,
            embeddings=batch_embeddings
        )
        print(f"  Loaded batch {i//db_batch_size + 1}/{(len(documents)-1)//db_batch_size + 1}")
    
    print(f"\n✓ Successfully loaded {len(documents)} documents to ChromaDB")
    print(f"✓ Collection: '{collection_name}'")
    print(f"✓ Persist directory: '{persist_directory}'")
    print(f"✓ Embedding model: mistral-embed")
    
    return collection

def query_collection(collection, query_text: str, n_results: int = 5):
    """
    Query the collection using Mistral embeddings
    """
    # Get embedding for query
    print(f"\nGenerating query embedding...")
    query_embedding = get_text_embedding(query_text)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"\nQuery: '{query_text}'")
    print(f"Top {n_results} results:\n")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"{i+1}. {metadata['title']}")
        print(f"   Source: {metadata['source_id']}")
        print(f"   Link: {metadata['link']}")
        print(f"   Similarity: {1 - distance:.3f}")
        print(f"   Snippet: {doc[:200]}...")
        print()
    
    return results

if __name__ == "__main__":
    # Install required libraries first:
    # pip install chromadb mistralai numpy
    
    # Set your Mistral API key as environment variable:
    # export MISTRAL_API_KEY='your-api-key'
    
    json_file = 'data/filtered_english_output.json'
    
    # Load data into ChromaDB with Mistral embeddings
    collection = load_to_chromadb(
        json_file=json_file,
        collection_name="news_articles",
        persist_directory="./chroma_db",
        batch_size=10  # Adjust based on API rate limits
    )
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)
    
    query_collection(collection, "inflation and economy", n_results=3)
    query_collection(collection, "technology companies", n_results=3)
    
    print("\n" + "="*60)