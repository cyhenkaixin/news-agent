import json
from langdetect import detect, LangDetectException

def is_english(text):
    """Check if text is in English"""
    if not text or text.strip() == "":
        return False
    try:
        # Detect language - returns language code like 'en', 'th', etc.
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If detection fails, assume not English
        return False

def filter_json_records(input_file, output_file):
    """Filter JSON records to keep only English content"""
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Track statistics
    total_records = len(data)
    null_content = 0
    non_english = 0
    
    # Filter records
    filtered_data = []
    for record in data:
        content = record.get('content')
        
        # Skip if content is null
        if content is None:
            null_content += 1
            continue
        
        # Skip if content is not English
        if not is_english(content):
            non_english += 1
            continue
        
        # Keep this record
        filtered_data.append(record)
    
    # Save the filtered data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print(f"Total records: {total_records}")
    print(f"Records with null content: {null_content}")
    print(f"Records with non-English content: {non_english}")
    print(f"Remaining English records: {len(filtered_data)}")
    print(f"Total filtered out: {total_records - len(filtered_data)}")

if __name__ == "__main__":
    # Install required library first: pip install langdetect
    
    input_file = './data/Business_News/Business_News.json'
    output_file = './data/filtered_english_output.json'
    
    filter_json_records(input_file, output_file)