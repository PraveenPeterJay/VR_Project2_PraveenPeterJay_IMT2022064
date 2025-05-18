# Install required libraries
!pip install transformers datasets torch pillow bert_score tqdm nltk pandas matplotlib evaluate rouge_score py-rouge sacrebleu

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import ViltProcessor, ViltForQuestionAnswering, BartTokenizer, BartForConditionalGeneration
from PIL import Image
from tqdm.notebook import tqdm
from bert_score import score as bert_score_calc
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import evaluate
nltk.download('punkt')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Step 1: Verify directory structure
print("Input directory:", os.listdir('/kaggle/input/'))
print("vqa-dataset contents:", os.listdir('/kaggle/input/vqa-dataset'))
print("small folder contents:", os.listdir('/kaggle/input/vqa-dataset/small'))

# Step 2: Load the full dataset from CSV
csv_path = '/kaggle/input/vqa-dataset/small/vqa.csv'
full_dataset = load_dataset("csv", data_files=csv_path)

# Step 3: Update image paths to absolute paths
base_image_path = '/kaggle/input/vqa-dataset/small'
full_dataset = full_dataset.map(lambda example: {'image_path': os.path.join(base_image_path, example['image_path'])})

# Step 4: Load pre-trained models and processors
print("Loading ViLT model and processor...")
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load BART model for BARTScore calculation
print("Loading BART model for BARTScore...")
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_model.eval()

# Initialize evaluation metrics
bertscore = evaluate.load('bertscore')

# Step 5: Define preprocessing function
def preprocess_example(example):
    image_path = example['image_path']
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    question = example['question']
    inputs = processor(image, question, return_tensors="pt")
    return inputs

# Step 6: Define prediction function
def get_prediction(example):
    inputs = preprocess_example(example)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_id = logits.argmax(-1).item()
        predicted_answer = model.config.id2label[predicted_id]
    return predicted_answer

# Step 7: Create sampling functions for different dataset sizes
def create_sampled_dataset(dataset, percentage):
    dataset_size = len(dataset['train'])
    sample_size = int(dataset_size * percentage)
    
    # Get random indices without replacement
    indices = random.sample(range(dataset_size), sample_size)
    
    # Create sampled dataset
    sampled_data = dataset['train'].select(indices)
    return sampled_data

# Step 8: Define evaluation metrics
def calculate_metrics(predictions, ground_truths):
    metrics = {}
    
    # Ensure predictions and ground_truths are valid
    valid_pairs = [(p, g) for p, g in zip(predictions, ground_truths) 
                    if isinstance(p, str) and isinstance(g, str)]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'bleu_score': 0.0,
            'bert_score_p': 0.0,
            'bert_score_r': 0.0,
            'bert_score_f1': 0.0,
            'bart_score': 0.0,
            'rouge_l': 0.0
        }
    
    valid_predictions, valid_ground_truths = zip(*valid_pairs)
    
    # Accuracy calculation
    correct = sum(1 for p, g in zip(valid_predictions, valid_ground_truths) 
                 if p.lower() == g.lower())
    metrics['accuracy'] = correct / len(valid_pairs) if valid_pairs else 0.0
    
    # F1 Score calculation (token level)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, truth in valid_pairs:
        pred_tokens = set(pred.lower().split())
        truth_tokens = set(truth.lower().split())
        
        true_positives += len(pred_tokens.intersection(truth_tokens))
        false_positives += len(pred_tokens - truth_tokens)
        false_negatives += len(truth_tokens - pred_tokens)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # BLEU Score
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, truth in valid_pairs:
        pred_tokens = nltk.word_tokenize(pred.lower())
        truth_tokens = [nltk.word_tokenize(truth.lower())]
        try:
            bleu = sentence_bleu(truth_tokens, pred_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            bleu_scores.append(0)
    
    metrics['bleu_score'] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # BERTScore using evaluate library
    try:
        bert_results = bertscore.compute(
            predictions=valid_predictions, 
            references=valid_ground_truths, 
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"
        )
        metrics['bert_score_p'] = sum(bert_results['precision']) / len(bert_results['precision'])
        metrics['bert_score_r'] = sum(bert_results['recall']) / len(bert_results['recall'])
        metrics['bert_score_f1'] = sum(bert_results['f1']) / len(bert_results['f1'])
    except Exception as e:
        print(f"BERTScore calculation error with evaluate library: {e}")
        # Fallback to bert_score package
        try:
            # Using smaller batches to avoid memory issues
            batch_size = 32
            all_bert_p = []
            all_bert_r = []
            all_bert_f1 = []
            
            for i in range(0, len(valid_predictions), batch_size):
                batch_preds = valid_predictions[i:i+batch_size]
                batch_refs = valid_ground_truths[i:i+batch_size]
                
                P, R, F1 = bert_score_calc(batch_preds, batch_refs, lang="en", verbose=False)
                all_bert_p.extend(P.tolist())
                all_bert_r.extend(R.tolist())
                all_bert_f1.extend(F1.tolist())
            
            metrics['bert_score_p'] = sum(all_bert_p) / len(all_bert_p) if all_bert_p else 0
            metrics['bert_score_r'] = sum(all_bert_r) / len(all_bert_r) if all_bert_r else 0
            metrics['bert_score_f1'] = sum(all_bert_f1) / len(all_bert_f1) if all_bert_f1 else 0
        except Exception as e:
            print(f"BERTScore fallback calculation error: {e}")
            metrics['bert_score_p'] = 0
            metrics['bert_score_r'] = 0
            metrics['bert_score_f1'] = 0
    
    # BARTScore calculation
    bart_scores = []
    try:
        with torch.no_grad():
            for pred, ref in tqdm(list(zip(valid_predictions, valid_ground_truths)), desc="Calculating BARTScore", leave=False):
                # Skip empty strings
                if not pred.strip() or not ref.strip():
                    continue
                
                # Encode the prediction and reference
                inputs = bart_tokenizer(ref, return_tensors="pt", max_length=1024, truncation=True)
                with torch.no_grad():
                    pred_encoded = bart_tokenizer(pred, return_tensors="pt", max_length=1024, truncation=True)
                    ref_ids = inputs["input_ids"]
                    
                    # Calculate log likelihood
                    outputs = bart_model(
                        input_ids=pred_encoded["input_ids"],
                        attention_mask=pred_encoded["attention_mask"],
                        labels=ref_ids
                    )
                    neg_log_likelihood = outputs.loss.item()
                    # Convert negative log likelihood to score (higher is better)
                    bart_score = -neg_log_likelihood
                    bart_scores.append(bart_score)
        
        metrics['bart_score'] = sum(bart_scores) / len(bart_scores) if bart_scores else 0
    except Exception as e:
        print(f"BARTScore calculation error: {e}")
        metrics['bart_score'] = 0
    
    # ROUGE-L Score
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        
        for pred, truth in valid_pairs:
            score = scorer.score(truth, pred)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        metrics['rouge_l'] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    except Exception as e:
        print(f"ROUGE-L calculation error: {e}")
        metrics['rouge_l'] = 0
    
    return metrics

# Step 9: Define evaluation function with progress bar
def evaluate_dataset(dataset_sample, name):
    print(f"\nEvaluating {name} dataset ({len(dataset_sample)} samples)...")
    predictions = []
    ground_truths = []
    
    # Use tqdm for progress bar
    for example in tqdm(dataset_sample, desc=f"Processing {name} dataset"):
        try:
            predicted_answer = get_prediction(example)
            ground_truth = example['answer']
            
            # Skip if ground_truth or predicted_answer is invalid
            if (ground_truth is None or not isinstance(ground_truth, str) or
                predicted_answer is None or not isinstance(predicted_answer, str)):
                continue
            
            predictions.append(predicted_answer)
            ground_truths.append(ground_truth)
        
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            continue
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)
    
    # Print metrics
    print(f"\n{name} Evaluation Results:")
    print(f"Number of samples evaluated: {len(predictions)}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"F1 Score: {metrics['f1_score'] * 100:.2f}%")
    print(f"BLEU Score: {metrics['bleu_score'] * 100:.2f}%")
    print(f"BERTScore Precision: {metrics['bert_score_p'] * 100:.2f}%")
    print(f"BERTScore Recall: {metrics['bert_score_r'] * 100:.2f}%")
    print(f"BERTScore F1: {metrics['bert_score_f1'] * 100:.2f}%")
    print(f"BARTScore: {metrics['bart_score']:.4f}")
    print(f"ROUGE-L: {metrics['rouge_l'] * 100:.2f}%")
    
    return metrics

# Step 10: Create different sized datasets
print(f"\nTotal dataset size: {len(full_dataset['train'])} samples")

# Create different sized datasets
full_sample = full_dataset['train']
half_sample = create_sampled_dataset(full_dataset, 0.5)
quarter_sample = create_sampled_dataset(full_dataset, 0.25)

print(f"Full dataset: {len(full_sample)} samples")
print(f"50% dataset: {len(half_sample)} samples")
print(f"25% dataset: {len(quarter_sample)} samples")

# Step 11: Evaluate and collect results
results = {}

# Evaluate quarter dataset
results['25% Dataset'] = evaluate_dataset(quarter_sample, "25% Dataset")

# Evaluate half dataset
results['50% Dataset'] = evaluate_dataset(half_sample, "50% Dataset")

# Evaluate full dataset
results['Full Dataset'] = evaluate_dataset(full_sample, "Full Dataset")

# Step 12: Create comparison charts
def plot_metrics_comparison(results):
    metrics_to_plot = ['accuracy', 'f1_score', 'bleu_score', 'bert_score_f1', 'bart_score', 'rouge_l']
    display_names = {
        'accuracy': 'Accuracy',
        'f1_score': 'F1 Score',
        'bleu_score': 'BLEU Score',
        'bert_score_f1': 'BERTScore F1',
        'bart_score': 'BARTScore',
        'rouge_l': 'ROUGE-L'
    }
    
    # Create figure with subplots - 2 rows of 3 metrics each
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot each metric
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        dataset_names = list(results.keys())
        
        # Handle BARTScore differently as it's not a percentage
        if metric == 'bart_score':
            values = [results[dataset][metric] for dataset in dataset_names]
            bars = axes[i].bar(dataset_names, values)
            axes[i].set_title(display_names[metric])
            axes[i].set_ylabel('Score')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.4f}', ha='center', va='bottom')
        else:
            values = [results[dataset][metric] * 100 for dataset in dataset_names]
            bars = axes[i].bar(dataset_names, values)
            axes[i].set_title(display_names[metric])
            axes[i].set_ylabel('Score (%)')
            axes[i].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 1,
                             f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('vilt_metrics_comparison.png')
    plt.close()
    
    # Create dataset size comparison
    dataset_sizes = {
        '25% Dataset': len(quarter_sample),
        '50% Dataset': len(half_sample),
        'Full Dataset': len(full_sample)
    }
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(dataset_sizes.keys(), dataset_sizes.values())
    plt.title('Dataset Size Comparison')
    plt.ylabel('Number of Samples')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dataset_size_comparison.png')
    plt.close()

# Generate comparison plots
plot_metrics_comparison(results)

# Save results to CSV
results_df = pd.DataFrame({
    'Dataset': list(results.keys()),
    'Size': [len(quarter_sample), len(half_sample), len(full_sample)],
    'Accuracy (%)': [results[ds]['accuracy'] * 100 for ds in results],
    'F1 Score (%)': [results[ds]['f1_score'] * 100 for ds in results],
    'BLEU Score (%)': [results[ds]['bleu_score'] * 100 for ds in results],
    'BERTScore P (%)': [results[ds]['bert_score_p'] * 100 for ds in results],
    'BERTScore R (%)': [results[ds]['bert_score_r'] * 100 for ds in results],
    'BERTScore F1 (%)': [results[ds]['bert_score_f1'] * 100 for ds in results],
    'BARTScore': [results[ds]['bart_score'] for ds in results],
    'ROUGE-L (%)': [results[ds]['rouge_l'] * 100 for ds in results]
})

results_df.to_csv('vilt_evaluation_results.csv', index=False)
print("\nResults saved to vilt_evaluation_results.csv")
print("\nEvaluation complete!")