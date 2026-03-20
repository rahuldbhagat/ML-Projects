"""
RAG Builder v2.0 for ERP Connect CogitION
Enhanced with semantic chunking and rich metadata
"""

import os
import ssl
import pickle
import json
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np

# SSL bypass for corporate networks - MUST be before imports
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

import certifi
certifi.where = lambda: False
ssl._create_default_https_context = ssl._create_unverified_context

# Now import network-dependent libraries
import faiss
from sentence_transformers import SentenceTransformer

class RAGBuilderV2:
    """Build RAG system with semantic chunking and metadata"""
    
    def __init__(self, config_path='configs/rag_config.yaml'):
        print("Initializing RAG Builder v2.0...")
        
        # Load config
        self.config = self.get_default_config()
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config.update(yaml.safe_load(f))
        
        # Initialize embedder
        print(f"Loading embedding model: {self.config['embeddings']['model']}")
        self.embedder = SentenceTransformer(self.config['embeddings']['model'])
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        
        # Storage
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'by_category': {},
            'build_time': None
        }
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'knowledge_base': {
                'input_dir': 'data/knowledge_base',
                'file_extensions': ['.md', '.txt', '.json'],
                'recursive': True
            },
            'chunking': {
                'max_tokens': 512,
                'min_tokens': 50,
                'overlap_tokens': 128,
                'strategy': 'semantic'
            },
            'embeddings': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2',
                'batch_size': 32,
                'normalize': True
            },
            'index': {
                'type': 'IVFFlat',
                'nlist': 10,
                'nprobe': 3
            },
            'output': {
                'index_path': 'models/knowledge_base_v2.index',
                'docs_path': 'models/documents_v2.pkl',
                'stats_path': 'models/rag_stats_v2.json'
            }
        }
    
    def load_documents(self):
        """Load all documents from knowledge base"""
        kb_dir = Path(self.config['knowledge_base']['input_dir'])
        extensions = self.config['knowledge_base']['file_extensions']
        
        print(f"\n📂 Loading documents from: {kb_dir}")
        
        documents = []
        for ext in extensions:
            files = list(kb_dir.rglob(f'*{ext}'))
            for filepath in files:
                doc = self.load_document(filepath)
                if doc:
                    documents.append(doc)
                    
                    # Update stats
                    category = doc['metadata']['category']
                    if category not in self.stats['by_category']:
                        self.stats['by_category'][category] = 0
                    self.stats['by_category'][category] += 1
        
        self.stats['total_documents'] = len(documents)
        print(f"   ✓ Loaded {len(documents)} documents")
        
        return documents
    
    def load_document(self, filepath):
        """Load single document with metadata extraction"""
        try:
            if filepath.suffix == '.json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        'content': json.dumps(data, indent=2),
                        'metadata': {
                            'source_file': str(filepath),
                            'category': self.extract_category_from_path(filepath),
                            'component': 'JSON_DATA',
                            'difficulty': 'intermediate',
                            'file_type': 'json'
                        }
                    }
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    metadata = self.extract_metadata(content, filepath)
                    return {
                        'content': content,
                        'metadata': metadata
                    }
        except Exception as e:
            print(f"   ⚠ Error loading {filepath}: {e}")
            return None
    
    def extract_metadata(self, content, filepath):
        """Extract metadata from document"""
        metadata = {
            'source_file': str(filepath),
            'category': self.extract_category_from_path(filepath),
            'component': 'GENERAL',
            'difficulty': 'intermediate',
            'file_type': filepath.suffix[1:]
        }
        
        # Try to parse YAML frontmatter
        if content.startswith('---'):
            try:
                end_idx = content.find('---', 3)
                if end_idx > 0:
                    frontmatter = content[3:end_idx].strip()
                    fm_data = yaml.safe_load(frontmatter)
                    if fm_data:
                        metadata.update(fm_data)
            except:
                pass
        
        return metadata
    
    def extract_category_from_path(self, filepath):
        """Extract category from file path"""
        path_parts = filepath.parts
        if 'architecture' in path_parts:
            return 'ARCHITECTURE'
        elif 'http_errors' in path_parts:
            return 'HTTP_ERRORS'
        elif 'integration' in path_parts:
            return 'INTEGRATION'
        elif 'sap' in path_parts:
            return 'SAP'
        elif 'performance' in path_parts:
            return 'PERFORMANCE'
        else:
            return 'GENERAL'
    
    def chunk_documents(self, documents):
        """Chunk documents with semantic boundaries"""
        print(f"\n✂️  Chunking documents...")
        print(f"   Strategy: {self.config['chunking']['strategy']}")
        print(f"   Max tokens: {self.config['chunking']['max_tokens']}")
        print(f"   Overlap: {self.config['chunking']['overlap_tokens']}")
        
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.stats['total_chunks'] = len(all_chunks)
        print(f"   ✓ Created {len(all_chunks)} chunks")
        
        return all_chunks
    
    def chunk_document(self, doc):
        """Chunk single document with semantic boundaries"""
        content = doc['content']
        metadata = doc['metadata']
        
        # Remove frontmatter if present
        if content.startswith('---'):
            end_idx = content.find('---', 3)
            if end_idx > 0:
                content = content[end_idx + 3:].strip()
        
        chunks = []
        
        if metadata['file_type'] == 'json':
            # For JSON, chunk by top-level entries
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    for key, value in data.items():
                        chunk_text = f"{key}: {json.dumps(value, indent=2)}"
                        chunks.append(self.create_chunk(chunk_text, metadata, len(chunks)))
                elif isinstance(data, list):
                    for idx, item in enumerate(data):
                        chunk_text = json.dumps(item, indent=2)
                        chunks.append(self.create_chunk(chunk_text, metadata, idx))
            except:
                chunks.append(self.create_chunk(content, metadata, 0))
        else:
            # For markdown/text, chunk by sections
            sections = self.split_by_headers(content)
            
            for idx, section in enumerate(sections):
                # Skip very small sections
                if len(section.split()) < self.config['chunking']['min_tokens']:
                    continue
                
                # If section too large, split by paragraphs
                if len(section.split()) > self.config['chunking']['max_tokens']:
                    paragraphs = section.split('\n\n')
                    current_chunk = ""
                    
                    for para in paragraphs:
                        if len((current_chunk + para).split()) <= self.config['chunking']['max_tokens']:
                            current_chunk += para + "\n\n"
                        else:
                            if current_chunk:
                                chunks.append(self.create_chunk(current_chunk.strip(), metadata, len(chunks)))
                            current_chunk = para + "\n\n"
                    
                    if current_chunk:
                        chunks.append(self.create_chunk(current_chunk.strip(), metadata, len(chunks)))
                else:
                    chunks.append(self.create_chunk(section, metadata, idx))
        
        return chunks
    
    def split_by_headers(self, content):
        """Split content by markdown headers"""
        lines = content.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if line.startswith('#'):
                # New section
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def create_chunk(self, text, metadata, chunk_idx):
        """Create chunk with metadata"""
        return {
            'text': text,
            'chunk_id': f"{Path(metadata['source_file']).stem}_{chunk_idx}",
            'chunk_index': chunk_idx,
            'category': metadata['category'],
            'component': metadata.get('component', 'GENERAL'),
            'difficulty': metadata.get('difficulty', 'intermediate'),
            'source_file': metadata['source_file'],
            'word_count': len(text.split())
        }
    
    def generate_embeddings(self, chunks):
        """Generate embeddings for all chunks"""
        print(f"\n🔢 Generating embeddings...")
        print(f"   Model: {self.config['embeddings']['model']}")
        print(f"   Batch size: {self.config['embeddings']['batch_size']}")
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        batch_size = self.config['embeddings']['batch_size']
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(
                batch,
                normalize_embeddings=self.config['embeddings']['normalize'],
                show_progress_bar=True
            )
            embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 100 == 0:
                print(f"   Progress: {min(i + batch_size, len(texts))}/{len(texts)}")
        
        embeddings = np.array(embeddings).astype('float32')
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        
        return embeddings
    
    def build_index(self, embeddings):
        """Build FAISS index"""
        print(f"\n🏗️  Building FAISS index...")
        print(f"   Type: {self.config['index']['type']}")
        
        n_embeddings = len(embeddings)
        
        if self.config['index']['type'] == 'IVFFlat' and n_embeddings >= 100:
            # Use IVF index for better performance
            nlist = min(self.config['index']['nlist'], n_embeddings // 10)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train index
            print(f"   Training index with {n_embeddings} vectors...")
            index.train(embeddings)
            
            # Add vectors
            index.add(embeddings)
            index.nprobe = self.config['index']['nprobe']
            
            print(f"   ✓ IVF index built (nlist={nlist}, nprobe={index.nprobe})")
        else:
            # Use flat index for small datasets
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(embeddings)
            print(f"   ✓ Flat index built")
        
        print(f"   Total vectors: {index.ntotal}")
        
        return index
    
    def save(self):
        """Save index, chunks, and stats"""
        print(f"\n💾 Saving RAG system...")
        
        # Create output directory
        output_dir = Path(self.config['output']['index_path']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.config['output']['index_path']
        faiss.write_index(self.index, index_path)
        print(f"   ✓ Index saved: {index_path}")
        
        # Save chunks
        docs_path = self.config['output']['docs_path']
        with open(docs_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"   ✓ Chunks saved: {docs_path}")
        
        # Save stats
        stats_path = self.config['output']['stats_path']
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"   ✓ Stats saved: {stats_path}")
    
    def build_all(self):
        """Complete RAG building pipeline"""
        start_time = datetime.now()
        
        # Load documents
        documents = self.load_documents()
        
        # Chunk documents
        self.chunks = self.chunk_documents(documents)
        
        # Generate embeddings
        self.embeddings = self.generate_embeddings(self.chunks)
        
        # Build index
        self.index = self.build_index(self.embeddings)
        
        # Save everything
        end_time = datetime.now()
        self.stats['build_time'] = str(end_time - start_time)
        self.save()
        
        print("\n" + "="*60)
        print("✅ RAG SYSTEM BUILD COMPLETE!")
        print("="*60)
        print(f"Documents: {self.stats['total_documents']}")
        print(f"Chunks: {self.stats['total_chunks']}")
        print(f"Build time: {self.stats['build_time']}")
        print("\nBy category:")
        for category, count in self.stats['by_category'].items():
            print(f"  {category}: {count} documents")

def main():
    """Main entry point"""
    print("="*60)
    print("RAG BUILDER v2.0")
    print("="*60)
    
    builder = RAGBuilderV2()
    builder.build_all()

if __name__ == '__main__':
    main()
