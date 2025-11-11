import os
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PolicyRetriever:
    def __init__(self, file_path="data/data_policy.txt"):
        self.file_path = file_path
        self.documents = []
        self.index = None
        self.model = None
        
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._load_documents()
            self._build_index()
            
            print(f"✅ Policy Retriever initialized with {len(self.documents)} fine-grained policy clauses!")
            
        except Exception as e:
            print(f"❌ Retriever initialization failed: {e}")
            raise
    
    def _load_documents(self):
        """Load and segment policy documents into fine-grained chunks"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Policy file not found: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        sections = re.split(r'Section \d+:', content)
        if len(sections) > 1:
            sections = sections[1:]
        
        for section_idx, section in enumerate(sections, 1):
            if section.strip():
                policy_points = re.findall(r'(\d+\.\d+\.[^\.]+\.)', section)
                
                if not policy_points:
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    for sentence in sentences:
                        if len(sentence.strip()) > 20:
                            self.documents.append({
                                'text': sentence.strip(),
                                'reference': f"Section {section_idx}",
                                'type': 'sentence'
                            })
                else:
                    for point in policy_points:
                        clean_point = point.strip()
                        if clean_point:
                            self.documents.append({
                                'text': clean_point,
                                'reference': f"Section {section_idx}",
                                'type': 'policy_point'
                            })
        
        if not self.documents:
            print("⚠️ Using fallback line-based chunking")
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_section = "General"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                section_match = re.match(r'Section \d+[:\.]\s*(.+)', line)
                if section_match:
                    current_section = section_match.group(1)
                    continue
                
                if len(line) > 30 and not line.isupper():
                    self.documents.append({
                        'text': line,
                        'reference': current_section,
                        'type': 'line'
                    })
    
    def _build_index(self):
        """Build FAISS index for semantic search"""
        if not self.documents:
            raise ValueError("No documents to index")
        
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        for i, doc in enumerate(self.documents):
            doc['embedding'] = embeddings[i]
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"🔍 Built FAISS index with {len(self.documents)} documents")
    
    def retrieve(self, query, k=4):
        """Retrieve top-k relevant policy clauses with similarity scores"""
        if self.index is None:
            raise RuntimeError("Index not built")
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        search_k = min(k * 3, len(self.documents))
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        seen_texts = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                if (doc['text'] not in seen_texts and score > 0.1):
                    seen_texts.add(doc['text'])
                    results.append({
                        'text': doc['text'],
                        'reference': doc['reference'],
                        'score': float(score),
                        'type': doc.get('type', 'unknown')
                    })
                
                if len(results) >= k:
                    break
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        score_list = [f"{r['score']:.3f}" for r in results]
        print(f"🔍 Retrieved {len(results)} relevant clauses for: '{query}'")
        print(f"   Similarity scores: {score_list}")
        return results

    def debug_retrieval(self, query, k=4):
        """Debug method to see what's being retrieved"""
        print(f"\n🔍 DEBUG RETRIEVAL for: '{query}'")
        results = self.retrieve(query, k)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['reference']}] Score: {result['score']:.3f}")
            print(f"     Text: {result['text'][:80]}...")
            print(f"     Type: {result['type']}")
        
        return results
