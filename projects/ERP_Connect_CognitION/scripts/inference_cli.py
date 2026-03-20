"""
CLI Inference for ERP Connect CogitION v2.0
Command-line interface for quick testing and automation
"""

import argparse
import json
import pickle
import torch
import faiss
import numpy as np
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

class ERPCogitIONCLI:
    """CLI interface for ERP Connect CogitION"""
    
    def __init__(self, model_dir, rag_index=None, rag_docs=None, use_rag=True):
        print("Loading ERP Connect CogitION...")
        
        # Load model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()
        
        # Load RAG if enabled
        self.use_rag = use_rag and rag_index and rag_docs
        if self.use_rag:
            self.index = faiss.read_index(rag_index)
            with open(rag_docs, 'rb') as f:
                self.documents = pickle.load(f)
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print(f"✓ Model loaded with RAG ({len(self.documents)} chunks)")
        else:
            print("✓ Model loaded (no RAG)")
    
    def retrieve_context(self, query, top_k=5):
        """Retrieve relevant context from RAG"""
        if not self.use_rag:
            return ""
        
        # Encode query
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get documents
        contexts = []
        for idx in indices[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                contexts.append(doc['text'])
        
        return "\n\n".join(contexts[:3])  # Use top 3
    
    def generate_answer(self, question, max_length=512):
        """Generate answer for question"""
        # Get context if RAG enabled
        context = self.retrieve_context(question) if self.use_rag else ""
        
        # Format prompt
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Generate
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        if "Answer:" in generated:
            answer = generated.split("Answer:")[-1].strip()
        else:
            answer = generated
        
        return answer

def main():
    parser = argparse.ArgumentParser(description='ERP Connect CogitION CLI')
    parser.add_argument('question', nargs='?', help='Question to ask')
    parser.add_argument('--model', default='models/fine_tuned_gpt2', help='Model directory')
    parser.add_argument('--rag-index', default='models/knowledge_base_v2.index', help='RAG index file')
    parser.add_argument('--rag-docs', default='models/documents_v2.pkl', help='RAG documents file')
    parser.add_argument('--no-rag', action='store_true', help='Disable RAG')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--batch', help='Batch file with questions (one per line)')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    # Load model
    cli = ERPCogitIONCLI(
        model_dir=args.model,
        rag_index=args.rag_index if not args.no_rag else None,
        rag_docs=args.rag_docs if not args.no_rag else None,
        use_rag=not args.no_rag
    )
    
    # Interactive mode
    if args.interactive:
        print("\nInteractive mode (type 'quit' to exit)")
        print("="*60)
        while True:
            try:
                question = input("\n🤔 Question: ")
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                answer = cli.generate_answer(question)
                print(f"\n🤖 Answer: {answer}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")
        return
    
    # Batch mode
    if args.batch:
        with open(args.batch, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        results = []
        for q in questions:
            answer = cli.generate_answer(q)
            results.append({'question': q, 'answer': answer})
        
        if args.format == 'json':
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"\nQ: {r['question']}")
                print(f"A: {r['answer']}")
                print("-"*60)
        return
    
    # Single question mode
    if args.question:
        answer = cli.generate_answer(args.question)
        
        if args.format == 'json':
            print(json.dumps({'question': args.question, 'answer': answer}, indent=2))
        else:
            print(f"\n🤔 Question: {args.question}")
            print(f"\n🤖 Answer: {answer}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
