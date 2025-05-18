import os
import random
import numpy as np
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import pandas as pd
from PIL import Image
import re
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define preprocessing function for exact matching (VQA-standard inspired)
def preprocess_answer(answer):
    """Preprocess answer by converting to lowercase and handling punctuation."""
    answer = answer.lower()
    answer = re.sub(r'[^\w\s\']', ' ', answer)  # Replace punctuation except apostrophe with space
    answer = ' '.join(answer.split())  # Normalize spaces
    return answer

# Load BLIP-VQA-base model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load the CSV file
csv_path = "/kaggle/input/vqa-dataset/small/vqa.csv"
df = pd.read_csv(csv_path)

# Drop rows where 'answer' is NaN to prevent AttributeError
print(f"Original dataset size: {len(df)}")
df = df.dropna(subset=['answer'])
print(f"After dropping NaN answers: {len(df)}")

# Randomly sample 25% of the dataset
sample_df = df.sample(frac=0.25, random_state=42)

# Lists to store generated and ground truth answers
generated_answers = []
ground_truth_answers = []

# Process each sample in the subset
print("Generating answers for sampled dataset...")
for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    # Construct full image path
    image_path = os.path.join("/kaggle/input/vqa-dataset/small", row['image_path'])
    question = row['question']
    ground_truth = row['answer']

    # Load and preprocess the image
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        continue

    # Preprocess inputs and generate answer
    inputs = processor(raw_image, question, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    generated_answer = processor.decode(out[0], skip_special_tokens=True)

    # Store results
    generated_answers.append(generated_answer)
    ground_truth_answers.append(ground_truth)

# Compute performance metrics
print("\nComputing performance metrics...")

# 1. Exact Matching Accuracy (with preprocessing)
exact_matches = [preprocess_answer(gen) == preprocess_answer(gt) 
                 for gen, gt in zip(generated_answers, ground_truth_answers)]
accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

# 2. BERT Score (F1) on raw answers
P, R, F1 = bert_score(generated_answers, ground_truth_answers, lang="en", verbose=True)
bert_f1 = F1.mean().item()

# 3. ROUGE-L Score (F1) on raw answers
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = [scorer.score(gt, gen)['rougeL'].fmeasure 
                  for gt, gen in zip(ground_truth_answers, generated_answers)]
rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

# 4. F1 Score based on word overlap (on raw answers)
def compute_f1(gen, gt):
    """Compute F1 score based on word overlap between generated and ground truth answers."""
    gen_tokens = set(gen.lower().split())
    gt_tokens = set(gt.lower().split())
    intersection = gen_tokens.intersection(gt_tokens)
    if not gen_tokens or not gt_tokens:
        return 0.0
    precision = len(intersection) / len(gen_tokens)
    recall = len(intersection) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

f1_scores = [compute_f1(gen, gt) for gen, gt in zip(generated_answers, ground_truth_answers)]
f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

# Print the results
print("\nBaseline Performance Metrics for BLIP-VQA-base (25% Sample):")
print(f"Exact Matching Accuracy: {accuracy:.4f}")
print(f"BERT Score F1: {bert_f1:.4f}")
print(f"ROUGE-L F1: {rouge_l:.4f}")
print(f"Word Overlap F1: {f1:.4f}")