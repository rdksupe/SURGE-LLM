# evaluate_balanced_sample.py

import argparse
import json
import os
import time
import torch
from scipy.stats import norm
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import roc_auc_score
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# A dictionary to map short names to Hugging Face model identifiers
model_fullnames = {
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma2-9b': 'google/gemma-3-4b-pt', # Corrected from gemma-3-4b-pt as per user code
}

def get_model_fullname(model_name):
    return model_fullnames.get(model_name, model_name)

def load_model(model_name, device, cache_dir, quantization=None):
    model_fullname = get_model_fullname(model_name)
    print(f'Loading model {model_fullname}...')
    model_kwargs = {"cache_dir": cache_dir}
    print("-> Loading model in bfloat16 (half-precision)...")
    model_kwargs["torch_dtype"] = torch.bfloat16
    model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_fullname, **model_kwargs)
    model.eval()
    return model

def load_tokenizer(model_name, cache_dir):
    model_fullname = get_model_fullname(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_fullname, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    log_likelihood_sum = log_likelihood.sum(dim=-1)
    mean_ref_sum = mean_ref.sum(dim=-1)
    var_ref_sum = var_ref.sum(dim=-1)
    denominator = torch.sqrt(torch.relu(var_ref_sum)) + 1e-6
    discrepancy = (log_likelihood_sum - mean_ref_sum) / denominator
    return discrepancy.item()

def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1 + 1e-6)
    return prob

class FastDetectGPTDetector:
    def __init__(self, scoring_model_name, sampling_model_name, device, cache_dir, quantization):
        self.scoring_model_name = scoring_model_name
        self.sampling_model_name = sampling_model_name
        self.scoring_tokenizer = load_tokenizer(scoring_model_name, cache_dir)
        self.scoring_model = load_model(scoring_model_name, device, cache_dir, quantization)
        if sampling_model_name == scoring_model_name:
            self.sampling_model = self.scoring_model
            self.sampling_tokenizer = self.scoring_tokenizer
        else:
            self.sampling_tokenizer = load_tokenizer(sampling_model_name, cache_dir)
            self.sampling_model = load_model(sampling_model_name, device, cache_dir, quantization)
        self.classifier_params = {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039}

    def compute_crit(self, text):
        tokenized_score = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", max_length=1024)
        labels = tokenized_score.input_ids[:, 1:].to(self.scoring_model.device)
        if labels.shape[1] == 0:
            return 0.0
        with torch.no_grad():
            inputs_score = {k: v.to(self.scoring_model.device) for k, v in tokenized_score.items()}
            logits_score = self.scoring_model(**inputs_score).logits[:, :-1]
            if self.sampling_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized_ref = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", max_length=1024)
                inputs_ref = {k: v.to(self.sampling_model.device) for k, v in tokenized_ref.items()}
                logits_ref = self.sampling_model(**inputs_ref).logits[:, :-1]
        crit = get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)
        return crit

    def compute_prob(self, text):
        crit = self.compute_crit(text)
        prob = compute_prob_norm(crit, **self.classifier_params)
        return prob

# --- MODIFIED: Dataset Loading and Balanced Sampling Logic ---
def load_and_sample_raid_data(samples_per_class=500, seed=42):
    """Loads RAID from HF, creates a balanced sample of human and gpt-4 texts."""
    print("Loading 'liamdugan/raid' dataset from Hugging Face Hub...")
    dataset = load_dataset("liamdugan/raid", split="train") 
    
    # Filter for human texts
    print("Filtering for 'human' samples...")
    human_ds = dataset.filter(lambda x: x['model'] == 'human')
    print(f"Found {len(human_ds)} total human samples.")

    # Filter for gpt-4 texts
    print("Filtering for 'gpt-4' samples...")
    gpt4_ds = dataset.filter(lambda x: x['model'] == 'gpt4')
    print(f"Found {len(gpt4_ds)} total gpt-4 samples.")

    # Shuffle and select samples from each class
    human_shuffled = human_ds.shuffle(seed=seed)
    gpt4_shuffled = gpt4_ds.shuffle(seed=seed)

    num_human = min(samples_per_class, len(human_shuffled))
    num_gpt4 = min(samples_per_class, len(gpt4_shuffled))
    print(f"Selecting {num_human} random human samples and {num_gpt4} random gpt-4 samples...")
    
    human_samples = human_shuffled.select(range(num_human))
    gpt4_samples = gpt4_shuffled.select(range(num_gpt4))

    # Combine the two datasets and shuffle them together
    print("Combining and shuffling the final balanced dataset...")
    combined_ds = concatenate_datasets([human_samples, gpt4_samples])
    final_ds = combined_ds.shuffle(seed=seed)

    return final_ds

def main(args):
    print("--- Initializing Fast-DetectGPT Detector ---")
    detector = FastDetectGPTDetector(
        scoring_model_name=args.scoring_model_name,
        sampling_model_name=args.sampling_model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        quantization=args.quantization
    )
    print("\n--- Detector Initialized ---")

    sampled_dataset = load_and_sample_raid_data(samples_per_class=args.samples_per_class)
    
    print(f"\n--- Running detection on {len(sampled_dataset)} balanced samples ---")
    predictions = []
    true_labels = []

    for item in tqdm(sampled_dataset, desc="Processing samples"):
        try:
            text_to_check = item['generation']
            prob = detector.compute_prob(text_to_check)
            predictions.append(prob)
            true_labels.append(0 if item['model'] == 'human' else 1)
        except Exception as e:
            print(f"Error processing a sample: {e}. Skipping.")
            continue
            
    print("\n--- Evaluating Results ---")
    if len(predictions) > 0 and len(set(true_labels)) > 1:
        roc_auc = roc_auc_score(true_labels, predictions)
        print(f"\nEvaluation Complete!")
        print(f"ROC AUC Score on balanced Human vs. GPT-4 set: {roc_auc:.4f}")
    else:
        print(f"Could not compute ROC AUC. Processed {len(predictions)} samples with labels: {set(true_labels)}.")

    results_df = pd.DataFrame({'true_label': true_labels, 'predicted_prob': predictions})
    results_df.to_csv(args.output_file, index=False)
    print(f"Detailed results saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end evaluation script for Fast-DetectGPT on a RAID sample.")
    
    parser.add_argument('--scoring_model_name', type=str, default="gemma2-9b", help="Scoring model short name.")
    parser.add_argument('--sampling_model_name', type=str, default=None, help="Sampling model short name. Defaults to scoring model.")
    parser.add_argument('--quantization', type=str, default=None, help="Use '4bit' for quantization or None for fp16/bf16.")
    
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument('--cache_dir', type=str, default="./model_cache", help="Directory to cache models.")
    
    # --- MODIFIED ARGUMENT ---
    parser.add_argument('--samples_per_class', type=int, default=500, help="Number of random samples to get from each class (human and gpt-4).")
    parser.add_argument('--output_file', type=str, default="evaluation_results_balanced.csv", help="Output file for detailed results.")
    
    args = parser.parse_args()
    
    if args.sampling_model_name is None:
        args.sampling_model_name = args.scoring_model_name

    main(args)