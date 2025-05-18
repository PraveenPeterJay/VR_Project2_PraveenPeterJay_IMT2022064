import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
import torch
import re
from torch.cuda.amp import autocast
from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel

def normalize_answer(s):
    """Normalize answer for more accurate comparison."""
    # Define reversed number map (digits to words)
    number_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
    }
    
    # Convert digits to words
    for digit, word in number_map.items():
        s = re.sub(r'\b' + digit + r'\b', word, s.lower())
    
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation and extra whitespace
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def main():
    amp_context = autocast 

    MODEL_PATH = "model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)
    
    # Load the model
    base_model_id = "Salesforce/blip-vqa-base"
    processor = BlipProcessor.from_pretrained(base_model_id)
    model = BlipForQuestionAnswering.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    generated_answers = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row['question'])
        answer = ''
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
        
            with torch.no_grad():
                with amp_context():
                    generated_ids = model.generate(**encoding)
                    predicted_answer = processor.decode(generated_ids[0], skip_special_tokens=True)

                answer = normalize_answer(predicted_answer)

        except Exception as e:
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

    print(generated_answers)

if __name__ == "__main__":
    main()
