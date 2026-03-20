# Fine-Tuning DistilGPT-2 for E-Commerce Customer Support FAQs

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [Problem Statement](#2-problem-statement)
3.  [Deliverables Checklist](#3-deliverables-checklist)
4.  [How to Run](#4-how-to-run)
5.  [Notebook Structure (Cell-by-Cell)](#5-notebook-structure-cell-by-cell)
6.  [Dataset](#6-dataset)
7.  [Model Architecture and LoRA](#7-model-architecture-and-lora)
8.  [Training Procedure](#8-training-procedure)
9.  [Overfitting Prevention](#9-overfitting-prevention)
10. [Evaluation Methodology](#10-evaluation-methodology)
11. [Inference](#11-inference)
12. [Configuration Reference](#12-configuration-reference)
13. [Expected Results and Sample Outputs](#13-expected-results-and-sample-outputs)
14. [Compute Requirements](#14-compute-requirements)
15. [Dependencies](#15-dependencies)
16. [Design Decisions](#16-design-decisions)
17. [Challenges and Limitations](#17-challenges-and-limitations)
18. [Potential Improvements](#18-potential-improvements)
19. [Troubleshooting](#19-troubleshooting)


## 1. Project Overview

This project fine-tunes DistilGPT-2 (82M parameters) to generate helpful, domain-specific
responses to customer queries in the e-commerce domain. A synthetic dataset of approximately
800 FAQ-style question-answer pairs is generated, cleaned, and used to train the model using
LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning on a single Google Colab T4 GPU.

The entire project is contained in a single Jupyter notebook. All code runs directly in
notebook cells -- there are no external Python files to manage. Open the notebook, connect
to a Colab GPU kernel, and run cells top to bottom.


## 2. Problem Statement

Generic large language models like GPT-2, LLaMA, or Mistral are trained on broad internet
text and are not optimized for domain-specific customer support. When asked e-commerce
questions, they produce verbose, off-topic, or generic responses.

This project demonstrates that fine-tuning a small LLM on domain-specific FAQ data can
produce concise, contextually appropriate answers for typical customer questions, even with
limited compute resources and a small dataset.

The fine-tuned model should learn to generate responses like:

    Input:  "Can I return a product after 30 days?"
    Output: "Returns are only accepted within 30 days of delivery. Sorry!"

    Input:  "How do I track my order?"
    Output: "You can track your order by logging into your account and visiting the
             'My Orders' section."


## 3. Deliverables Checklist

This table maps each requirement from the problem statement to where it is addressed:

| Requirement | Status | Where |
|-------------|--------|-------|
| Cleaned and formatted dataset (JSONL) | Done | Step 3 + Step 4 cells |
| Synthetic dataset generation script | Done | Step 3 cell (~800 pairs) |
| Fine-tuned LLM checkpoints | Done | Step 5 cell (saves to outputs/best_model/) |
| Inference script (CLI or Gradio) | Done | Step 7 cell (Gradio with share URL) |
| Manual evaluation (base vs fine-tuned, 10 queries) | Done | Step 6 cell (side-by-side comparison) |
| Automated evaluation (BLEU / ROUGE / embedding sim) | Done | Step 6 cell (all three metrics) |
| Early stopping / validation loss monitoring | Done | Step 5 cell (patience=2, load_best_model) |
| Short report (README) | Done | This file |
| Model <=125M parameters | Done | DistilGPT-2 = 82M |
| LoRA + accelerate | Done | Step 5 cell |
| 4-bit or 8-bit quantization | Done | Step 5 cell (4-bit NF4, configurable) |
| Colab T4 GPU compatible | Done | Tested on T4 |
| HF Trainer or trl library | Done | SFTTrainer from trl |
| 3-5 epochs, batch size 8 | Done | 5 epochs, batch 8 |
| E-commerce domain, 500-1000 Q&A pairs | Done | ~800 pairs, e-commerce |


## 4. How to Run

### Prerequisites
- VS Code with the Google Colab extension installed, OR
- A Google Colab account (web browser), OR
- Any machine with a CUDA GPU

### Option A: VS Code + Google Colab (Recommended)

1. Install the Google Colab extension in VS Code (search "Google Colab" in Extensions).
2. Open `finetuning-customer-support-faq.ipynb` from your local machine in VS Code.
3. Click "Select Kernel" -> "Select Another Kernel" -> "Colab" -> "New Colab Server".
4. Sign in with your Google account. Select T4 GPU when prompted.
5. Run all cells from top to bottom.

No files need to be uploaded to Google Drive. The notebook creates all data files directly
on the Colab runtime filesystem.

### Option B: Google Colab Web UI

1. Go to https://colab.research.google.com
2. File -> Upload notebook -> select `finetuning-customer-support-faq.ipynb`
3. Runtime -> Change runtime type -> T4 GPU -> Save
4. Run all cells from top to bottom.

### Option C: Local GPU Machine

1. Install Python 3.10+ and CUDA toolkit.
2. Install Jupyter: `pip install jupyter`
3. Install dependencies: `pip install peft trl bitsandbytes accelerate datasets sentence-transformers rouge-score nltk gradio matplotlib torch transformers`
4. Open the notebook: `jupyter notebook finetuning-customer-support-faq.ipynb`
5. Run all cells.

### What Happens When You Run

- Step 1: Installs packages (~2-3 min first time)
- Step 2: Sets all configuration variables in memory
- Step 3: Generates ~800 FAQ pairs, saves to /content/faq_finetuning/data/ (~5 sec)
- Step 4: Cleans, formats, splits data (~2 sec)
- Step 5: Fine-tunes with LoRA (~5-15 min on T4)
- Step 6: Evaluates both models, plots loss curve (~3-5 min)
- Step 7: Launches Gradio with a public URL for interactive testing


## 5. Notebook Structure (Cell-by-Cell)

The notebook has 19 cells (9 markdown, 10 code). Here is what each does:

| Cell | Type | Step | What It Does |
|------|------|------|-------------|
| 0 | Markdown | -- | Title, overview, pipeline description |
| 1 | Markdown | 1 | Section header for setup |
| 2 | Code | 1 | Installs pip packages (peft, trl, bitsandbytes, etc.) |
| 3 | Code | 1 | Verifies GPU availability, prints device name and VRAM |
| 4 | Markdown | 2 | Section header for configuration |
| 5 | Code | 2 | Defines all config variables: paths, hyperparams, LoRA settings, prompt templates |
| 6 | Markdown | 3 | Section header for dataset generation |
| 7 | Code | 3 | Defines templates for 10 sub-topics, generates ~800 Q&A pairs, saves raw JSONL |
| 8 | Markdown | 4 | Section header for data preparation |
| 9 | Code | 4 | Cleans text, deduplicates, formats, splits 80/10/10, saves train/val/test JSONL |
| 10 | Markdown | 5 | Section header for fine-tuning |
| 11 | Code | 5 | Loads model with 4-bit quant, applies LoRA, trains with SFTTrainer + early stopping |
| 12 | Markdown | 6 | Section header for evaluation |
| 13 | Code | 6 | Loads both models, computes BLEU/ROUGE/cosine on 10 queries, plots loss curve |
| 14 | Markdown | 6 | Sub-header for loss curve display |
| 15 | Code | 6 | Displays loss_curve.png inline in notebook |
| 16 | Markdown | 7 | Section header for inference |
| 17 | Code | 7 | Loads fine-tuned model, launches Gradio web UI with examples |
| 18 | Markdown | -- | Completion message, instructions for saving outputs to Drive |


## 6. Dataset

### Domain
E-commerce customer support, covering 10 sub-topics:

1. **Order tracking** -- Tracking orders, missing packages, delivery confirmation
2. **Shipping** -- Shipping times, costs, carriers, international, weekend shipping
3. **Returns** -- Return policy, initiating returns, damaged items, sale items
4. **Refunds** -- Refund timelines, payment methods, partial refunds, coupons
5. **Account** -- Registration, password reset, email changes, security, addresses
6. **Discounts** -- Promo codes, student discounts, loyalty programs, first-time offers
7. **Products** -- Stock, dimensions, materials, warranties, reviews, size guides
8. **Cancellations** -- Order cancellation, partial cancellation, fees, pre-orders
9. **Gift cards** -- Redemption, balance, expiration, denominations
10. **Technical** -- Website issues, checkout errors, app crashes, notifications

### Generation Method
Template-based expansion using Python's random module with a fixed seed (42) for
reproducibility. Each sub-topic has 6-8 question-answer template pairs with variable
placeholders (e.g., {days}, {product}, {ship_cost}) filled with randomly selected values.
Multiple generation rounds produce the target count.

No external APIs, web scraping, or pre-existing datasets are used.

### Data Format

Raw format (each line in raw_faq.jsonl):
```json
{"prompt": "How do I track my order?", "response": "You can track your order by logging into your account and visiting the 'My Orders' section."}
```

Training format (each line in train.jsonl):
```json
{"text": "Question: How do I track my order?\nAnswer: You can track your order by logging into your account and visiting the 'My Orders' section.", "prompt": "...", "response": "..."}
```

### Size and Splits
- Generated: ~800 Q&A pairs
- After deduplication: slightly fewer (depends on random seed)
- Train: 80%
- Validation: 10%
- Test: 10%

### Cleaning Steps
1. Remove control characters (except newline and tab)
2. Collapse multiple spaces into single spaces
3. Strip leading/trailing whitespace
4. Remove entries with empty prompt or response
5. Deduplicate on lowercased (prompt, response) pairs
6. Shuffle with fixed seed before splitting


## 7. Model Architecture and LoRA

### Base Model: DistilGPT-2
- Source: Hugging Face `distilgpt2`
- Parameters: ~82 million (under the 125M requirement)
- Architecture: 6-layer transformer decoder, 768 hidden dim, 12 attention heads
- Vocabulary: 50,257 BPE tokens
- Max context: 1024 tokens
- Training data: OpenWebText (distilled from GPT-2)

### LoRA (Low-Rank Adaptation)
LoRA freezes all base model weights and injects small trainable low-rank matrices into
specific layers. This means only ~0.5M parameters are trained (less than 1% of the model).

- Rank (r): 16
- Alpha: 32 (scaling factor; alpha/r = 2)
- Dropout: 0.05
- Target modules: `c_attn` (QKV projections), `c_proj` (attention output)
- Bias: none
- Task type: CAUSAL_LM

### Quantization
- Mode: 4-bit NF4 (Normal Float 4-bit) via bitsandbytes
- Compute dtype: float16
- Double quantization: enabled
- Effect: model weights shrink from ~330MB (fp16) to ~80MB
- Configurable: change QUANTIZATION_MODE to "8bit" or "none"


## 8. Training Procedure

### Trainer
SFTTrainer from the `trl` library, a wrapper around Hugging Face Trainer designed for
supervised fine-tuning. Automatically tokenizes the "text" field.

### Hyperparameters
- Epochs: 5 (max; may stop earlier)
- Batch size: 8 per device
- Gradient accumulation: 2 steps (effective batch size = 16)
- Learning rate: 2e-4
- Optimizer: AdamW
- Weight decay: 0.01
- Warmup: 5% of total steps
- Max sequence length: 256 tokens
- Mixed precision: fp16
- Seed: 42

### Training Flow
1. Load DistilGPT-2 with 4-bit quantization and device_map="auto"
2. Prepare model for quantized training (gradient checkpointing)
3. Apply LoRA adapters
4. Load tokenized train and validation datasets
5. Train with evaluation at each epoch end
6. Early stopping checks validation loss (patience=2)
7. Best checkpoint (lowest val loss) loaded automatically
8. LoRA adapter weights saved to outputs/best_model/
9. Training log saved to outputs/training_log.json


## 9. Overfitting Prevention

Four mechanisms prevent memorization of the training data:

1. **Early stopping**: EarlyStoppingCallback monitors eval_loss. If validation loss
   does not improve for 2 consecutive epochs, training halts.

2. **Best model selection**: load_best_model_at_end=True ensures the saved model has
   the lowest validation loss, not just the final epoch weights.

3. **LoRA dropout**: 5% dropout on LoRA layers provides stochastic regularization.

4. **Weight decay**: L2 regularization (0.01) via AdamW prevents large weight values.

The loss curve (outputs/loss_curve.png) visualizes overfitting detection.


## 10. Evaluation Methodology

### Manual Evaluation
10 test queries are run through both the base DistilGPT-2 and the fine-tuned model.
Outputs are printed side by side in the notebook for qualitative assessment of
relevance, conciseness, and domain accuracy.

### Automated Metrics

**BLEU** (Bilingual Evaluation Understudy):
- Measures n-gram overlap between generated and reference answers.
- Uses smoothing (method1) for short texts.
- Range: 0.0 to 1.0 (higher = more lexical overlap).
- Library: nltk.translate.bleu_score

**ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation):
- Measures longest common subsequence between texts.
- Reports F1 score (precision-recall harmonic mean).
- Range: 0.0 to 1.0 (higher = more structural overlap).
- Library: rouge-score with stemming.

**Cosine Embedding Similarity**:
- Encodes both texts as dense vectors via all-MiniLM-L6-v2 (sentence-transformers).
- Computes cosine similarity between vectors.
- Captures semantic meaning even when wording differs.
- Range: -1.0 to 1.0 (higher = more semantically similar).
- This is the most meaningful metric for FAQ evaluation.

### Output Files
- outputs/evaluation_results.json: Per-query and aggregate metrics for both models.
- outputs/loss_curve.png: Training vs validation loss plot.


## 11. Inference

### Gradio Web UI (Default)
- Browser-based interface with a text input and two sliders.
- Temperature slider: 0.1 (deterministic) to 1.5 (creative).
- Max tokens slider: 20 to 200.
- 8 pre-loaded example queries for quick testing.
- Creates a public share URL (share=True) accessible from any device.

### Post-Processing
Generated responses are cleaned by:
1. Extracting only newly generated tokens (excluding the prompt).
2. Stripping whitespace.
3. Truncating at "Question:" if the model starts generating another Q&A pair.
4. Repetition penalty (1.2) reduces repetitive outputs.


## 12. Configuration Reference

All settings are defined in the Step 2 cell. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| PROJECT_ROOT | /content/faq_finetuning | Runtime filesystem path |
| TARGET_QA_PAIRS | 800 | Synthetic pairs to generate |
| RANDOM_SEED | 42 | Reproducibility seed |
| TRAIN_RATIO / VAL / TEST | 0.80 / 0.10 / 0.10 | Dataset splits |
| MODEL_NAME | distilgpt2 | Hugging Face model ID |
| QUANTIZATION_MODE | 4bit | 4bit, 8bit, or none |
| LORA_RANK | 16 | LoRA matrix rank |
| LORA_ALPHA | 32 | LoRA scaling factor |
| LORA_DROPOUT | 0.05 | LoRA dropout rate |
| LORA_TARGET_MODULES | [c_attn, c_proj] | Attention layers |
| NUM_EPOCHS | 5 | Max training epochs |
| BATCH_SIZE | 8 | Per-device batch |
| GRADIENT_ACCUMULATION_STEPS | 2 | Effective batch = 16 |
| LEARNING_RATE | 2e-4 | Peak learning rate |
| WEIGHT_DECAY | 0.01 | L2 regularization |
| MAX_SEQ_LENGTH | 256 | Max token length |
| FP16 | True | Mixed precision |
| EARLY_STOPPING_PATIENCE | 2 | Epochs without improvement |
| NUM_MANUAL_EVAL_QUERIES | 10 | Test queries for evaluation |
| MAX_NEW_TOKENS | 100 | Inference generation limit |
| TEMPERATURE | 0.7 | Sampling temperature |
| TOP_P | 0.9 | Nucleus sampling threshold |


## 13. Expected Results and Sample Outputs

### Expected Metric Ranges

| Metric | Base DistilGPT-2 | Fine-Tuned | Direction |
|--------|------------------|------------|-----------|
| Avg BLEU | 0.01 - 0.05 | 0.10 - 0.30 | Higher is better |
| Avg ROUGE-L | 0.05 - 0.15 | 0.20 - 0.50 | Higher is better |
| Avg Cosine Sim | 0.10 - 0.30 | 0.50 - 0.80 | Higher is better |

Exact values vary with random seed and sampling.

### Sample Comparisons

**Query: "Can I return a product after 30 days?"**

Base DistilGPT-2:
> "The answer is that there is no way to get a refund on the product you purchased
> from us. We have a lot of products that are available for purchase..."
> (rambling, off-topic)

Fine-Tuned:
> "Returns are accepted within 30 days of delivery. Unfortunately, items returned
> after this period cannot be processed."
> (concise, accurate, domain-appropriate)

**Query: "How do I track my order?"**

Base DistilGPT-2:
> "I'm not sure what you mean by 'track'. I think you mean..."
> (confused, generic)

Fine-Tuned:
> "You can track your order by logging into your account and visiting the
> 'My Orders' section."
> (helpful, specific)

**Query: "Do gift cards expire?"**

Base DistilGPT-2:
> "The gift card is a great way to save money on your next purchase..."
> (generic marketing text)

Fine-Tuned:
> "No, our gift cards do not expire. You can use them at any time."
> (direct, correct answer)


## 14. Compute Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16 GB VRAM) | T4 or better |
| System RAM | 8 GB | 12+ GB |
| Disk | 5 GB free | 10 GB free |
| Training time | ~5 min | ~10-15 min |
| Evaluation time | ~2 min | ~3-5 min |
| Peak VRAM | ~4 GB | ~6 GB |

Everything fits within Google Colab's free tier (T4 GPU, 12.7 GB RAM).


## 15. Dependencies

Installed via pip in the first code cell:

| Package | Purpose |
|---------|---------|
| torch | PyTorch deep learning framework |
| transformers | Model loading, tokenization, Trainer base class |
| datasets | Efficient dataset loading from JSONL |
| accelerate | Device management, mixed precision, device_map |
| peft | LoRA adapter implementation |
| trl | SFTTrainer for supervised fine-tuning |
| bitsandbytes | 4-bit and 8-bit quantization |
| nltk | BLEU score computation |
| rouge-score | ROUGE-L score computation |
| sentence-transformers | Cosine embedding similarity |
| gradio | Interactive web inference UI |
| matplotlib | Loss curve plotting |
| numpy | Numerical operations |


## 16. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | DistilGPT-2 (82M) | Under 125M limit, fast training, clear before/after improvement |
| Fine-tuning | LoRA via PEFT | Only ~0.5M trainable params, fits T4, preserves base knowledge |
| Quantization | 4-bit NF4 | Minimizes VRAM, configurable to 8-bit or none |
| Trainer | SFTTrainer from trl | Purpose-built for supervised fine-tuning |
| Dataset | Template-based synthetic | No API dependencies, reproducible, covers 10 sub-topics |
| Training format | "Question: ... Answer: ..." | Simple causal LM completion format |
| Evaluation | BLEU + ROUGE-L + Cosine | Lexical overlap + structural + semantic similarity |
| Delivery | Single notebook, inline code | No file management, no uploads, runs anywhere |
| Inference UI | Gradio with share=True | Browser-based, public URL, works in Colab |


## 17. Challenges and Limitations

### Challenges
1. **Template diversity**: Synthetic templates lack natural variation (typos, slang,
   emotional language) found in real customer queries.
2. **Small dataset**: ~640 training samples limits generalization. LoRA helps by requiring
   fewer examples, but edge cases may not be covered.
3. **GPT-2 generation quality**: DistilGPT-2 can repeat itself or ramble past the answer
   boundary. Repetition penalty and truncation at "Question:" help mitigate this.
4. **Quantization tradeoff**: 4-bit reduces VRAM but may slightly degrade output quality.
5. **Colab file access**: VS Code + Colab extension cannot access local files. Solved by
   putting all code directly in notebook cells.

### Limitations
1. **Domain scope**: Only covers the 10 trained sub-topics. Out-of-domain queries produce
   low-quality responses.
2. **No retrieval**: Cannot look up real-time data (order statuses, inventory, accounts).
3. **Single-turn**: No conversation memory across turns.
4. **Model ceiling**: 82M parameters limits language sophistication.
5. **Metric imperfection**: BLEU/ROUGE measure surface overlap, not true answer quality.


## 18. Potential Improvements

1. **Real data**: Use actual customer support logs (PII removed) for natural language.
2. **Larger dataset**: 2,000-5,000 pairs for better coverage.
3. **Larger model**: TinyLlama 1.1B if compute constraint is relaxed.
4. **RAG**: Retrieval-augmented generation for dynamic, up-to-date answers.
5. **Multi-turn**: Fine-tune on conversation threads for follow-up handling.
6. **Human eval**: Likert-scale ratings from human evaluators.
7. **Data augmentation**: Paraphrasing or back-translation for diversity.
8. **Domain expansion**: Banking, SaaS, or multi-domain support.


## 19. Troubleshooting

**"No GPU detected" warning:**
Change runtime to T4 GPU. In VS Code: reconnect and select GPU. In Colab web:
Runtime -> Change runtime type -> T4 GPU.

**"CUDA out of memory" during training:**
Reduce BATCH_SIZE to 4 in the config cell. Or ensure QUANTIZATION_MODE is "4bit".

**Training loss not decreasing:**
Try LEARNING_RATE between 1e-4 and 5e-4. Verify data was generated correctly.

**Gradio not loading:**
Look for the "Running on public URL" message. If behind a firewall, the public share
link should still work. Ensure gradio is installed.

**Runtime disconnected, lost all files:**
Normal Colab behavior. Re-run all cells from the top. Outputs are lost unless saved
to Google Drive (see the final cell for instructions).

**Import errors after runtime restart:**
Re-run the pip install cell (Step 1) to reinstall packages.

**Low evaluation scores:**
Scores depend on sampling randomness. Run evaluation multiple times. The fine-tuned
model should consistently outperform base on all three metrics.
