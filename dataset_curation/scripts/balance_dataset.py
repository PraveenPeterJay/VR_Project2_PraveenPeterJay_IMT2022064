import os
import json
import pandas as pd
from tqdm import tqdm
import glob
import random

def extract_english_value(field):
    """Extract English value from a field with language tags"""
    if not field:
        return None
        
    # If field is not a list, return it directly
    if not isinstance(field, list):
        return field
    
    # Check if field has at least one item
    if not field:
        return None
    
    # Try to extract using pattern from second code first
    first = field[0]
    if isinstance(first, dict):
        if 'language_tag' in first and 'value' in first:
            if not first['language_tag'].startswith('en_'):
                return None
            return first.get('value')
    
    # For fields with language tags - find only English entries
    for item in field:
        if isinstance(item, dict):
            # Item with English language tag
            if "language_tag" in item and "value" in item and (item["language_tag"].startswith("en") or item["language_tag"].startswith("en_")):
                return item["value"]
                
    # If no English values found but we need some value, use the first available value
    for item in field:
        if isinstance(item, dict) and "value" in item:
            return item["value"]
    
    return None

def extract_all_keywords(keywords_field):
    """Extract all English keywords and deduplicate them"""
    if not keywords_field or not isinstance(keywords_field, list):
        return None
    
    # Collect only English keyword values
    keywords = [
        k['value'].strip().lower()
        for k in keywords_field
        if isinstance(k, dict) and 'value' in k and 
        ('language_tag' not in k or k['language_tag'].startswith('en') or k['language_tag'].startswith('en_'))
    ]
    
    # Deduplicate keywords
    seen = set()
    deduped_keywords = [k for k in keywords if not (k in seen or seen.add(k))]
    
    # Join all keywords with commas
    return ', '.join(deduped_keywords) if deduped_keywords else None

def process_json_files(json_directory, output_csv_path, images_csv_path, max_items_per_product_type=5000, drop_nulls=False):
    """Process all JSON files and create curated dataset
    
    Args:
        json_directory (str): Directory containing JSON files
        output_csv_path (str): Path to save the output CSV
        images_csv_path (str): Path to the images CSV file
        max_items_per_product_type (int): Maximum number of items to keep for each product type
        drop_nulls (bool): Whether to drop rows with null values in any field
    """
    
    # Load the images CSV
    print("Loading images CSV...")
    images_df = pd.read_csv(images_csv_path)
    
    # Create image ID to row mapping
    image_rows = {row['image_id']: row for _, row in images_df.iterrows()}
    print(f"Processed {len(images_df)} image IDs")
    
    # Create a dictionary to store metadata for each image ID
    image_metadata = {}
    
    # Get list of all JSON files
    json_files = sorted(glob.glob(os.path.join(json_directory, "listings_*.json")))
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # Parse JSON object from each line
                    product = json.loads(line.strip())
                    
                    # Get main_image_id if exists
                    main_image_id = product.get("main_image_id")
                    if not main_image_id or main_image_id not in image_rows:
                        continue
                    
                    # Extract required metadata fields - note we're removing material
                    metadata = {
                        'name': extract_english_value(product.get('item_name')),
                        'product_type': extract_english_value(product.get('product_type')),
                        'color': extract_english_value(product.get('color')),
                        'keywords': extract_all_keywords(product.get('item_keywords'))
                    }
                    
                    # Store metadata keyed by image ID
                    image_metadata[main_image_id] = metadata
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing entry: {str(e)}")
    
    print(f"Collected metadata for {len(image_metadata)} images")
    
    # Create the initial dataset
    result_data = []
    
    for image_id, row in tqdm(image_rows.items(), desc="Creating initial dataset"):
        if image_id in image_metadata:
            meta = image_metadata[image_id]
            
            # Filter out specific problematic image IDs
            if image_id in ['518Dk4FOzZL', '719hoe+OvIL', '71Qbh8wmhnL']:
                continue
                
            # Skip entries with missing required metadata
            if not meta.get('name') or not meta.get('product_type') or not meta.get('color') or not meta.get('keywords'):
                continue
                
            # Check if text is ASCII for certain fields
            is_ascii = lambda text: isinstance(text, str) and text.isascii()
            if not (is_ascii(meta.get('name', '')) and is_ascii(meta.get('product_type', '')) and is_ascii(meta.get('color', ''))):
                continue
                
            # Create entry for the dataset (without material field)
            entry = {
                'path': row['path'],
                'name': meta.get('name', '').lower() if meta.get('name') else '',
                'product_type': meta.get('product_type', '').lower() if meta.get('product_type') else '',
                'color': meta.get('color', '').lower() if meta.get('color') else '',
                'keywords': meta.get('keywords', '').lower() if meta.get('keywords') else ''
            }
            
            result_data.append(entry)
    
    # Create the initial dataframe
    initial_df = pd.DataFrame(result_data)
    
    # Replace empty strings with NaN if drop_nulls is True
    if drop_nulls:
        initial_df = initial_df.replace('', pd.NA)
        initial_df = initial_df.dropna()
    
    # Sort by path
    initial_df = initial_df.sort_values(by='path').reset_index(drop=True)
    
    print(f"Initial dataset size: {len(initial_df)}")
    
    # Balance the dataset by product_type
    product_type_counts = initial_df['product_type'].value_counts()
    print("\nTop 10 product types before balancing:")
    print(product_type_counts.head(10))
    
    # Create a balanced dataset
    balanced_data = []
    
    # Group by product_type
    grouped = initial_df.groupby('product_type')
    
    for product_type, group in tqdm(grouped, desc="Balancing dataset"):
        # If group size is less than or equal to max_items_per_product_type, keep all items
        if len(group) <= max_items_per_product_type:
            balanced_data.append(group)
        else:
            # Otherwise, randomly sample max_items_per_product_type items
            sampled = group.sample(max_items_per_product_type, random_state=42)
            balanced_data.append(sampled)
    
    # Concatenate all balanced groups
    balanced_df = pd.concat(balanced_data)
    
    # Sort by path
    balanced_df = balanced_df.sort_values(by='path').reset_index(drop=True)
    
    # Save to CSV
    balanced_df.to_csv(output_csv_path, index=False)
    
    # Print statistics
    final_product_type_counts = balanced_df['product_type'].value_counts()
    print("\nTop 10 product types after balancing:")
    print(final_product_type_counts.head(10))
    
    print(f"\nInitial dataset size: {len(initial_df)}")
    print(f"Final balanced dataset size: {len(balanced_df)}")
    print(f"Number of unique product types: {len(final_product_type_counts)}")
    print(f"Output saved to: {output_csv_path}")
    
    return {
        'total_images_processed': len(images_df),
        'images_with_metadata': len(image_metadata),
        'initial_dataset_size': len(initial_df),
        'final_dataset_size': len(balanced_df),
        'unique_product_types': len(final_product_type_counts)
    }

def main():
    # Use paths from the second code
    images_csv_path = "/home/shannon/IIITB-CourseWork/Sem_6/VR/VQA_MiniProject/abo-images-small/images/small/images.csv"
    json_directory = "/home/shannon/IIITB-CourseWork/Sem_6/VR/VQA_MiniProject/abo-listings/listings/metadata"
    output_csv_path = "/home/shannon/IIITB-CourseWork/Sem_6/VR/VQA_MiniProject/abo-images-small/images/small/balanced_dataset.csv"
    
    # Maximum number of items to keep for each product type
    max_items_per_product_type = 5000  # Adjust this value as needed
    
    # Set to True if you want to drop rows with null values
    drop_nulls = True  # Change this to True if you want to drop rows with null values
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Process the data
    stats = process_json_files(json_directory, output_csv_path, images_csv_path, 
                               max_items_per_product_type, drop_nulls)
    
    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total images processed: {stats['total_images_processed']}")
    print(f"Images with metadata found: {stats['images_with_metadata']}")
    print(f"Initial dataset size: {stats['initial_dataset_size']}")
    print(f"Final balanced dataset size: {stats['final_dataset_size']}")
    print(f"Number of unique product types: {stats['unique_product_types']}")

if __name__ == "__main__":
    main()