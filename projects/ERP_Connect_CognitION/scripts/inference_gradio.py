"""
5_launch_demo.py - Launch Gradio Demo with TinyLlama Model
"""

import os
import sys
import pickle
import torch
from pathlib import Path

# SSL bypass for corporate networks
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr

print("="*70)
print("LAUNCHING GRADIO DEMO - TINYLLAMA")
print("="*70)

# Paths
MODEL_DIR = Path("models/tinyllama-fine-tuned")
INDEX_FILE = Path("models/knowledge_base_v2.index")
DOCS_FILE = Path("models/documents_v2.pkl")

# Verify files
if not MODEL_DIR.exists():
    print(f"\nERROR: Model directory not found: {MODEL_DIR}")
    sys.exit(1)

if not INDEX_FILE.exists():
    print(f"\nERROR: Index file not found: {INDEX_FILE}")
    sys.exit(1)

if not DOCS_FILE.exists():
    print(f"\nERROR: Documents file not found: {DOCS_FILE}")
    sys.exit(1)

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

# Load RAG
print("\nLoading RAG system...")
index = faiss.read_index(str(INDEX_FILE))
with open(DOCS_FILE, 'rb') as f:
    documents = pickle.load(f)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"RAG system loaded with {len(documents)} document chunks")

# RAG retrieval
def retrieve_context(query, top_k=3):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    
    contexts = []
    sources = []
    for idx in indices[0]:
        if idx < len(documents):
            doc = documents[idx]
            contexts.append(doc['text'][:200])
            sources.append(doc.get('source_file', 'Unknown'))
    
    return "\n\n".join(contexts), sources

# Answer generation
def answer_question(question, use_rag=True, temperature=0.7):
    if not question.strip():
        return "Please enter a question.", "", ""
    
    context_text = ""
    sources_text = ""
    
    if use_rag:
        context, sources = retrieve_context(question)
        context_text = context
        sources_text = "\n".join([f"- {s}" for s in set(sources)])
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        prompt = question
    
    input_text = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=300,
            temperature=temperature,
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
    
    return answer, context_text, sources_text

# Build interface
print("\nBuilding Gradio interface...")

with gr.Blocks(title="ERP Connect CogitION - TinyLlama") as demo:
    gr.Markdown("""
    # ERP Connect CogitION - TinyLlama Edition
    
    AI-powered CTRM-SAP integration support system
    
    **Model:** TinyLlama-1.1B fine-tuned with LoRA  
    **Features:** RAG-enhanced context retrieval
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask about CTRM-SAP integration...",
                lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("Get Answer", variant="primary")
                clear_btn = gr.Button("Clear")
            
            with gr.Accordion("Advanced Options", open=False):
                use_rag = gr.Checkbox(label="Use RAG", value=True)
                temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
        
        with gr.Column(scale=3):
            answer_output = gr.Textbox(label="Answer", lines=10)
            
            with gr.Accordion("Retrieved Context", open=False):
                context_output = gr.Textbox(label="Context Used", lines=5)
            
            with gr.Accordion("Sources", open=False):
                sources_output = gr.Textbox(label="Sources", lines=3)
    
    gr.Examples(
        examples=[
            ["How to add interface mapping for Cost Center ENERGY572?"],
            ["Error: Server responded with HTTP 403 Forbidden"],
            ["ECS Fargate container won't start - exit code 137"],
            ["RDS database connection timeout"],
            ["Lambda function timeout after 30 seconds"]
        ],
        inputs=question_input
    )
    
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, use_rag, temperature],
        outputs=[answer_output, context_output, sources_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[question_input, answer_output, context_output, sources_output]
    )

print("Interface ready")
print("\nLaunching demo...")
print("Demo will open in your default web browser")
print("Press Ctrl+C to stop the server\n")

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)