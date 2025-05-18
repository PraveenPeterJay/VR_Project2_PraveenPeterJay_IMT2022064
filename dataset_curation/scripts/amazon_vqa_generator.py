import os
import pandas as pd
import time
import logging
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vqa_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GeminiVQAGenerator:
    """A class to generate Visual Question Answering (VQA) data using Google's Gemini API."""
    
    def __init__(self, api_key, max_retries=3):
        """
        Initialize the VQA generator with API key and rate limiting parameters.
        
        Args:
            api_key (str): Google AI API key
            max_retries (int): Maximum number of retries for failed API calls
        """
        self.api_key = api_key
        self.requests_per_minute = 15  # Gemini 2.0 Flash free tier limit from screenshot
        self.seconds_per_request = 60.0 / self.requests_per_minute
        self.max_retries = max_retries
        self.request_count = 0
        self.last_request_time = time.time()
        self.last_processed_index = -1
        self.checkpoint_file = "vqa_generator_checkpoint.txt"
        self.load_checkpoint()
        self.configure_gemini(api_key)
        
    def configure_gemini(self, api_key):
        """Configure the Gemini API with the provided key."""
        logger.info("Configuring Gemini API...")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            logger.info("Gemini 2.0 Flash API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    def sleep_between_requests(self):
        """Sleep to respect the 15 RPM rate limit for Gemini 2.0 Flash."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.seconds_per_request:
            sleep_time = self.seconds_per_request - elapsed
            logger.debug(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
        if self.request_count % 5 == 0:
            logger.info(f"Made {self.request_count} API requests so far")
            
    def save_checkpoint(self, index):
        """Save the last processed index to resume if interrupted."""
        self.last_processed_index = index
        try:
            with open(self.checkpoint_file, 'w') as f:
                f.write(str(index))
            logger.debug(f"Saved checkpoint at index {index}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            
    def load_checkpoint(self):
        """Load the last processed index from checkpoint file if it exists."""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    self.last_processed_index = int(f.read().strip())
                logger.info(f"Resuming from checkpoint at index {self.last_processed_index}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            self.last_processed_index = -1
    
    def generate_questions_for_image(self, image_path, product_info):
        """
        Generate 3 VQA pairs for an image using Gemini API.
        
        Args:
            image_path (str): Path to the image file
            product_info (dict): Dictionary containing product metadata
            
        Returns:
            list: List of dictionaries containing questions and answers
        """
        try:
            image = Image.open(image_path)
            prompt = f"""
            I need you to generate 3 diverse Visual Question Answering (VQA) pairs for this product image.
            
            Product Information:
            - Name: {product_info['name']}
            - Type: {product_info['product_type']}
            - Color: {product_info['color']}
            - Keywords: {product_info['keywords']}
            
            Instructions:
	    1. Analyze the provided image and the metadata.,
	    2. Generate exactly 3 distinct questions about prominent visual features, objects, colors, materials, or attributes clearly visible in the image.,
	    3. Each question MUST have a single-word answer directly verifiable from the image.,
	    4. The 3 questions generated MUST be different from each other, and MUST be answerable just by looking at the image.,
	    5. Vary the difficulty levels across questions, but they should be answerable just by looking at the image.
            
            Format your response as a JSON array:
            [
                {{"question": [Your first question here], "answer": [Your first answer here]}},
                {{"question": [Your second question here], "answer": [Your second answer here]}},
                {{"question": [Your third question here], "answer": [Your third answer here]}}
            ]
            
            Only respond with the JSON array.
            """
            
            for attempt in range(self.max_retries):
                self.sleep_between_requests()
                try:
                    response = self.model.generate_content([prompt, image])
                    response_text = response.text
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0].strip()
                    else:
                        start_idx = response_text.find("[")
                        end_idx = response_text.rfind("]") + 1
                        json_str = response_text[start_idx:end_idx].strip()
                    vqa_pairs = json.loads(json_str)
                    return vqa_pairs
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                    else:
                        logger.error(f"All {self.max_retries} attempts failed for {image_path}")
                        exit()
                        return []
        except Exception as e:
            logger.error(f"Error generating questions for {image_path}: {e}")
            return []

    def process_dataset(self, input_csv, images_dir, output_csv, max_samples=None):
        """
        Process the dataset to generate VQA pairs for each image.
        
        Args:
            input_csv (str): Path to input CSV with metadata
            images_dir (str): Directory containing images
            output_csv (str): Path to output CSV for VQA pairs
            max_samples (int, optional): Maximum number of samples to process
        """
        try:
            logger.info(f"Loading dataset from {input_csv}")
            df = pd.read_csv(input_csv)
            if max_samples:
                df = df.head(max_samples)
            vqa_data = []
            if os.path.exists(output_csv):
                try:
                    existing_df = pd.read_csv(output_csv)
                    vqa_data = existing_df.to_dict('records')
                    logger.info(f"Loaded {len(vqa_data)} existing VQA pairs")
                except Exception as e:
                    logger.warning(f"Failed to load existing VQA data: {e}")
            total_rows = len(df)
            start_idx = max(0, self.last_processed_index + 1)
            logger.info(f"Processing {total_rows - start_idx} images from index {start_idx}")
            for idx in tqdm(range(start_idx, total_rows), desc="Generating VQA pairs"):
                row = df.iloc[idx]
                image_path = os.path.join(images_dir, row['path'])
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                product_info = {
                    'name': row['name'],
                    'product_type': row['product_type'],
                    'color': row['color'],
                    'keywords': row['keywords']
                }
                try:
                    vqa_pairs = self.generate_questions_for_image(image_path, product_info)
                    for pair in vqa_pairs:
                        vqa_data.append({
                            'image_path': row['path'],
                            'question': pair['question'],
                            'answer': pair['answer']
                        })
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
                    continue
                self.save_checkpoint(idx)
                if (idx - start_idx + 1) % 5 == 0 or idx == total_rows - 1:
                    vqa_df = pd.DataFrame(vqa_data)
                    vqa_df.to_csv(output_csv, index=False)
                    logger.info(f"Saved {len(vqa_df)} VQA pairs to {output_csv}")
            final_vqa_df = pd.DataFrame(vqa_data)
            final_vqa_df.to_csv(output_csv, index=False)
            logger.info(f"Processing complete! Saved {len(final_vqa_df)} VQA pairs")
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                logger.info("Removed checkpoint file")
            return final_vqa_df
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise

def main():
    """Main function to run the VQA generation process."""
    # Update with your path
    input_csv = "images/small/balanced_dataset.csv"
    images_dir = "images/small"
    output_csv = "vqa.csv"
                
    api_key = "AIzaSyDCnvUuaDrAu7Yn2Fh9fAUOPBHa_n-XOhc"       # Replace with your actual API key
    max_samples = None                        # Set for testing
    logger.info("Starting VQA dataset generation with 15 RPM limit")
    vqa_generator = GeminiVQAGenerator(api_key=api_key)
    vqa_generator.process_dataset(
        input_csv=input_csv,
        images_dir=images_dir,
        output_csv=output_csv,
        max_samples=max_samples
    )
    logger.info("VQA dataset generation complete!")

if __name__ == "__main__":
    main()
    output_csv = "vqa.csv"

