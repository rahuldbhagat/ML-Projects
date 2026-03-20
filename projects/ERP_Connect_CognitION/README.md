# ERP Connect CognitION

**AI-Powered Technical Support System for CTRM-SAP Integration**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: TinyLlama](https://img.shields.io/badge/Model-TinyLlama--1.1B-green.svg)](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Complete Workflow](#complete-workflow)
- [Detailed Instructions](#detailed-instructions)
- [Configuration](#configuration)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

ERP Connect CognitION is an AI-powered technical support system designed to provide accurate, context-aware assistance for CTRM-SAP integration scenarios. The system combines **fine-tuned TinyLlama-1.1B** with **Retrieval-Augmented Generation (RAG)** to deliver expert-level guidance for complex integration issues.

### Problem Solved

- Knowledge fragmentation across multiple sources
- Inconsistent response quality from different support personnel  
- Time-intensive query resolution processes
- Difficulty scaling expertise as integration complexity grows

### Solution Approach

- **Parameter-Efficient Fine-tuning**: Uses LoRA to train only 0.7% of model parameters
- **Retrieval-Augmented Generation**: Grounds responses in technical documentation
- **CPU-Compatible**: Runs on standard hardware without GPU requirements
- **Fast Training**: Completes training in 11 minutes on Google Colab T4 GPU

---

## Key Features

✓ **Exceptional Training Performance**: 98.6% loss reduction, final validation loss of 0.033  
✓ **Strong Semantic Understanding**: Cosine similarity score of 0.849  
✓ **Production-Ready**: CPU inference with 6-10 second response times  
✓ **Resource Efficient**: Only 1.77M trainable parameters (0.7% of total)  
✓ **Comprehensive Coverage**: Trained on 1,200+ CTRM-SAP integration scenarios  
✓ **Interactive Demo**: Gradio-based web interface for easy testing  

---

## System Requirements

### Local Data Generation & Deployment

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: Not required (CPU-only)

### Training (Google Colab)

- **Google Account**: Free tier sufficient
- **GPU**: T4 GPU (provided free)
- **Training Time**: ~11 minutes

---

## Project Structure

```
ERP_Connect_CognitION/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── SETUP.bat                              # Windows: Environment setup
├── test_prompts.py                        # Test queries
│
├── configs/                               # Configuration files
│   ├── data_generation_config.yaml        # Data generation settings
│   ├── kb_generation_config.yaml          # Knowledge base settings
│   └── rag_config.yaml                    # RAG system configuration
│
├── scripts/                               # Python scripts
│   ├── generate_training_data_v2.py       # Generate training data
│   ├── generate_synthetic_kb.py           # Generate knowledge base
│   ├── build_rag_v2.py                    # Build FAISS index
│   ├── kb_templates.py                    # KB generation templates
│   ├── evaluate_model.py                  # Model evaluation
│   ├── inference_cli.py                   # CLI inference
│   └── inference_gradio.py                # Gradio web interface
│
├── data/                                  # Data files
│   ├── seed_data/                         # Original seed examples
│   ├── processed/                         # Generated training data
│   │   ├── train.jsonl                    # 1,200 samples
│   │   ├── validation.jsonl               # 150 samples
│   │   └── test.jsonl                     # 150 samples
│   └── knowledge_base/                    # RAG components
│       ├── knowledge_base_v2.index        # FAISS index
│       └── documents_v2.pkl               # Document chunks
│
├── models/                                # Model artifacts
│   └── tinyllama-fine-tuned/              # Fine-tuned model (from Colab)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
│── notebook/                                 # Model artifacts
│   └── ERP_Connect_CogitION_TinyLlama.ipynb  #Jupytor Notebook to be run in Google Colab   
|            
├── results/                               # Evaluation outputs
│   ├── evaluation_results.json
│   └── sample_predictions.txt
│
├── venv/                                  # Virtual environment
│
├── 1_GENERATE_KB.bat                      # Windows: Generate knowledge base
├── 2_GENERATE_DATA.bat                    # Windows: Generate training data
├── 3_BUILD_RAG.bat                        # Windows: Build RAG system
├── 4_DEMO_GRADIO.bat                      # Windows: Launch demo
│

```

---

## Installation

### Option 1: Automated Setup (Windows)

```bash
# Double-click or run:
SETUP.bat
```

Creates virtual environment and installs all dependencies automatically.

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Complete Workflow

```
┌───────────────────────────────────────────────────────────┐
│              LOCAL COMPUTER (Your PC)                     │
│                                                           │
│  Step 1: Generate Knowledge Base                         │
│  Step 2: Generate Training Data                          │
│  Step 3: Build RAG System                                │
└───────────────────────────────────────────────────────────┘
                        ↓ Upload Files
┌───────────────────────────────────────────────────────────┐
│              GOOGLE COLAB (Cloud GPU)                     │
│                                                           │
│  Step 4: Upload 5 files                                  │
│  Step 5: Train Model (11 minutes)                        │
│  Step 6: Download Model                                  │
└───────────────────────────────────────────────────────────┘
                        ↓ Download Model
┌───────────────────────────────────────────────────────────┐
│              LOCAL COMPUTER (Your PC)                     │
│                                                           │
│  Step 7: Evaluate Model (optional)                       │
│  Step 8: Launch Demo                                     │
└───────────────────────────────────────────────────────────┘
```

---

## Detailed Instructions

### Phase 1: Local Data Generation

#### Step 1: Generate Knowledge Base

**Windows:**
```bash
1_GENERATE_KB.bat
```

**Linux/Mac:**
```bash
python scripts/generate_synthetic_kb.py --config configs/kb_generation_config.yaml
```

**Output:** Synthetic documentation for RAG system  
**Time:** 2-5 minutes

#### Step 2: Generate Training Data

**Windows:**
```bash
2_GENERATE_DATA.bat
```

**Linux/Mac:**
```bash
python scripts/generate_training_data_v2.py --config configs/data_generation_config.yaml
```

**Output:**  
- `data/processed/train.jsonl` (1,200 samples)
- `data/processed/validation.jsonl` (150 samples)
- `data/processed/test.jsonl` (150 samples)

**Time:** 10-15 minutes with API, 2-3 minutes template-based

#### Step 3: Build RAG System

**Windows:**
```bash
3_BUILD_RAG.bat
```

**Linux/Mac:**
```bash
python scripts/build_rag_v2.py --config configs/rag_config.yaml
```

**Output:**
- `data/knowledge_base/knowledge_base_v2.index` (FAISS index)
- `data/knowledge_base/documents_v2.pkl` (document chunks)

**Time:** 5-10 minutes

---

### Phase 2: Upload to Google Colab

#### Step 4: Upload Files

**Files to Upload (5 total):**
```
☐ data/processed/train.jsonl
☐ data/processed/validation.jsonl
☐ data/processed/test.jsonl
☐ data/knowledge_base/knowledge_base_v2.index
☐ data/knowledge_base/documents_v2.pkl
```

**Process:**
1. Open `ERP_Connect_CognitION_TinyLlama.ipynb` in Google Colab
2. Runtime → Change runtime type → GPU (T4)
3. Run Cell 1 (install packages)
4. Run Cell 2A (upload files) or Cell 2B (load from Google Drive)
5. Upload the 5 files listed above

---

### Phase 3: Training on Google Colab

#### Step 5: Train Model

**Execute notebook cells in sequence:**

1. **Cell 1**: Install packages (~2 min)
2. **Cell 2A/2B**: Upload/load files
3. **Cell 3**: Verify files
4. **Cell 4**: **Train model** (~11 min)
5. **Cell 5**: Generate loss curve
6. **Cell 6**: Evaluate (optional, ~30 min)
7. **Cell 7**: Download model

**Expected Training Output:**
```
Epoch 1/3 - Step 75:  Loss: 0.956, Val: 0.287
Epoch 2/3 - Step 150: Loss: 0.042, Val: 0.038
Epoch 3/3 - Step 225: Loss: 0.032, Val: 0.033
Training complete!
```

#### Step 6: Download Model

**Method 1:** Run Cell 8 in notebook → Download ZIP  
**Method 2:** Download from Google Drive (if using persistence)

**Extract model:**
```bash
# Extract to models/ folder
unzip tinyllama-fine-tuned.zip -d models/
```

---

### Phase 4: Local Deployment

#### Step 7: Evaluate Model (Optional)

```bash
python scripts/evaluate_model.py
```

**Time:** 15-20 minutes  
**Output:** Evaluation metrics and sample predictions

#### Step 8: Launch Interactive Demo

**Windows:**
```bash
4_DEMO_GRADIO.bat
```

**Linux/Mac:**
```bash
python scripts/inference_gradio.py
```

**Access:** http://127.0.0.1:7860  
**Response Time:** 6-10 seconds per query

**Example Queries:**
- "How to add interface mapping for Cost Center TRADE456?"
- "RDS PostgreSQL slow query performance"
- "Authorization missing for company code 1000"
- "Lambda function out of memory error"

---

## Configuration

### Data Generation (`configs/data_generation_config.yaml`)

```yaml
augmentation_factor: 40  # Generate 1200 samples from 30 seeds
train_ratio: 0.8         # 80% training
val_ratio: 0.1           # 10% validation
test_ratio: 0.1          # 10% test
method: "template"       # or "gpt4" for API-based
```

### Knowledge Base (`configs/kb_generation_config.yaml`)

```yaml
categories:
  - interface_mappings
  - error_resolutions
  - aws_services
  - sap_authorization
documents_per_category: 10
chunk_size: 250
```

### RAG System (`configs/rag_config.yaml`)

```yaml
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
chunk_size: 250
chunk_overlap: 50
index_type: "Flat"
```

### Training (Colab Notebook)

```python
# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training
NUM_EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
```

---

## Results

### Training Performance

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.032 |
| Final Validation Loss | 0.033 |
| Loss Reduction | 98.6% |
| Training Time | 11 min 18 sec |
| Trainable Parameters | 1.77M (0.7%) |

### Evaluation Metrics

| Metric | Mean | Target | Status |
|--------|------|--------|--------|
| ROUGE-L | 0.523 | 0.75 | Approaching |
| BLEU | 0.364 | 0.65 | Approaching |
| Cosine Similarity | 0.849 | 0.90 | Near target |

### Model Comparison

| Model | Loss | Time | Quality |
|-------|------|------|---------|
| GPT-2 (124M) | 3.66 | 8 hrs | Repetitive |
| Flan-T5 (250M) | 8.06 | 18 min | Incoherent |
| **TinyLlama (1.1B)** | **0.032** | **45 min** | **Success** |

---

## Troubleshooting

### Common Issues

**Problem:** `ModuleNotFoundError`  
**Solution:** Activate venv and reinstall: `pip install -r requirements.txt`

**Problem:** GPU not available in Colab  
**Solution:** Runtime → Change runtime type → GPU (T4)

**Problem:** Files not found after upload  
**Solution:** Re-run upload cell, verify filenames match

**Problem:** Port 7860 in use  
**Solution:** Change port in `scripts/inference_gradio.py` or kill process

**Problem:** Slow inference (>30s)  
**Solution:** Reduce MAX_NEW_TOKENS, check CPU usage, close other apps

---

## File Descriptions

### Scripts

| File | Purpose |
|------|---------|
| `generate_synthetic_kb.py` | Create knowledge base documents |
| `generate_training_data_v2.py` | Expand seeds to training set |
| `build_rag_v2.py` | Build FAISS vector index |
| `evaluate_model.py` | Test model performance |
| `inference_gradio.py` | Launch web interface |

### Batch Files (Windows)

| File | Purpose |
|------|---------|
| `SETUP.bat` | Environment setup |
| `1_GENERATE_KB.bat` | Generate knowledge base |
| `2_GENERATE_DATA.bat` | Generate training data |
| `3_BUILD_RAG.bat` | Build RAG system |
| `4_DEMO_GRADIO.bat` | Launch demo |

### Configuration Files

| File | Purpose |
|------|---------|
| `data_generation_config.yaml` | Data generation settings |
| `kb_generation_config.yaml` | Knowledge base settings |
| `rag_config.yaml` | RAG system configuration |

---

## References

### Research Papers

1. Zhang, P., et al. (2023). "TinyLlama: An Open-Source Small Language Model"
2. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models"
3. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
4. Touvron, H., et al. (2023). "Llama: Open and Efficient Foundation Language Models"

### Documentation

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Gradio Documentation](https://gradio.app/docs)

---

## License

MIT License - See LICENSE file for details

### Third-Party Licenses

- TinyLlama: Apache 2.0
- Transformers: Apache 2.0
- PEFT: Apache 2.0
- FAISS: MIT
- Gradio: Apache 2.0

---

## Acknowledgments

- **TinyLlama Team** for the base model
- **Hugging Face** for transformers and PEFT
- **Meta AI** for Llama architecture
- **Facebook Research** for FAISS
- **Google Colab** for free GPU access

---



---

**Built for the ERP integration community**
