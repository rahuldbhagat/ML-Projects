"""
4_evaluate_model.py - Evaluate TinyLlama Fine-tuned Model
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# SSL bypass for corporate networks
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("="*70)
print("EVALUATING TINYLLAMA MODEL")
print("="*70)

# Paths
MODEL_DIR = Path("models/tinyllama-fine-tuned")
TEST_FILE = Path("data/processed/test.jsonl")

if not MODEL_DIR.exists():
    print(f"\nERROR: Model directory not found: {MODEL_DIR}")
    print("Please ensure you've extracted the model ZIP file to the models/ folder.")
    sys.exit(1)

if not TEST_FILE.exists():
    print(f"\nERROR: Test file not found: {TEST_FILE}")
    sys.exit(1)

# Load test data
print("\nLoading test data...")
test_data = []
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        test_data.append(json.loads(line))
print(f"Loaded {len(test_data)} test samples")

# Load model
print("\nLoading TinyLlama model...")
print("This may take a few minutes on first run...")

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="cpu",
    torch_dtype=torch.float32
)

model = PeftModel.from_pretrained(base_model, str(MODEL_DIR))
model.eval()
print("Model loaded successfully")

# Initialize metrics
print("\nInitializing evaluation metrics...")
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Metrics initialized")

# Generation function
def generate_answer(prompt, max_new_tokens=300):
    input_text = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("<|assistant|>")[-1].strip()
    
    if len(answer) > 1000:
        sentences = answer.split('. ')
        answer = '. '.join(sentences[:-1]) + '.'
    
    return answer

# Evaluate
print("\nEvaluating model...")
print(f"Processing {len(test_data)} samples (this will take 15-20 minutes on CPU)...\n")

results = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [], 'cosine': []}
samples_to_show = []

for idx, sample in enumerate(test_data):
    prediction = generate_answer(sample['prompt'])
    reference = sample['response']
    
    # ROUGE
    rouge_scores = rouge_scorer_obj.score(reference, prediction)
    results['rouge1'].append(rouge_scores['rouge1'].fmeasure)
    results['rouge2'].append(rouge_scores['rouge2'].fmeasure)
    results['rougeL'].append(rouge_scores['rougeL'].fmeasure)
    
    # BLEU
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
    results['bleu'].append(bleu)
    
    # Cosine
    ref_emb = similarity_model.encode([reference])
    pred_emb = similarity_model.encode([prediction])
    cosine = cosine_similarity(ref_emb, pred_emb)[0][0]
    results['cosine'].append(float(cosine))
    
    # Save samples
    if idx < 5:
        samples_to_show.append({
            'prompt': sample['prompt'],
            'prediction': prediction,
            'reference': reference[:100] + '...'
        })
    
    if (idx + 1) % 30 == 0:
        print(f"Progress: {idx + 1}/{len(test_data)}")

# Results
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

metrics = {
    'ROUGE-1': results['rouge1'],
    'ROUGE-2': results['rouge2'],
    'ROUGE-L': results['rougeL'],
    'BLEU': results['bleu'],
    'Cosine Similarity': results['cosine']
}

for name, values in metrics.items():
    mean_val = np.mean(values)
    print(f"\n{name}:")
    print(f"  Mean:  {mean_val:.4f}")
    print(f"  Std:   {np.std(values):.4f}")
    print(f"  Min:   {np.min(values):.4f}")
    print(f"  Max:   {np.max(values):.4f}")

print("\n" + "="*70)
print("TARGET COMPARISON")
print("="*70)

rouge_l = np.mean(results['rougeL'])
bleu = np.mean(results['bleu'])
cosine = np.mean(results['cosine'])

print(f"\nMetric          Target    Actual    Status")
print(f"="*45)
print(f"ROUGE-L         0.75      {rouge_l:.4f}    {'PASS' if rouge_l >= 0.75 else 'CLOSE' if rouge_l >= 0.70 else 'FAIL'}")
print(f"BLEU            0.65      {bleu:.4f}    {'PASS' if bleu >= 0.65 else 'CLOSE' if bleu >= 0.60 else 'FAIL'}")
print(f"Cosine Sim      0.90      {cosine:.4f}    {'PASS' if cosine >= 0.90 else 'CLOSE' if cosine >= 0.85 else 'FAIL'}")

print("\n" + "="*70)
print("SAMPLE PREDICTIONS")
print("="*70)

for i, sample in enumerate(samples_to_show, 1):
    print(f"\nSample {i}:")
    print(f"Q: {sample['prompt']}")
    print(f"\nA: {sample['prediction']}")
    print(f"\nExpected: {sample['reference']}")
    print("-" * 70)

# Save results
results_file = Path("evaluation_results.json")
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        'metrics': {k: float(np.mean(v)) for k, v in metrics.items()},
        'samples': samples_to_show
    }, f, indent=2)

print(f"\nResults saved to: {results_file}")
print("\nEvaluation complete")