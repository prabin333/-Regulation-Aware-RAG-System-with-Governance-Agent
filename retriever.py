import os
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class PolicyRetriever:
    """
    Efficient policy document retriever using FAISS with fine-grained chunking
    """
    def __init__(self, file_path="data/data_policy.txt"):
        self.file_path = file_path
        self.documents = []
        self.index = None
        self.model = None
        
        self._initialize_retriever()
    
    def _initialize_retriever(self):
        """Initialize the retriever with documents and index"""
        try:
            # Load embedding model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load and process documents
            self._load_documents()
            
            # Build FAISS index
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
        
        # Remove header and split into sections
        sections = re.split(r'Section \d+:', content)
        if len(sections) > 1:
            sections = sections[1:]  # Remove content before first section
        
        for section_idx, section in enumerate(sections, 1):
            if section.strip():
                # Extract individual policy points (1.1, 1.2, 2.1, etc.)
                policy_points = re.findall(r'(\d+\.\d+\.[^\.]+\.)', section)
                
                if not policy_points:
                    # Fallback: split by sentences for sections without numbered points
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    for sentence in sentences:
                        if len(sentence.strip()) > 20:  # Only meaningful sentences
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
        
        # If still no documents, fallback to line-based splitting
        if not self.documents:
            print("⚠️ Using fallback line-based chunking")
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_section = "General"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect section headers
                section_match = re.match(r'Section \d+[:\.]\s*(.+)', line)
                if section_match:
                    current_section = section_match.group(1)
                    continue
                
                # Only add meaningful lines (not too short, not headers)
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
        
        # Generate embeddings
        texts = [doc['text'] for doc in self.documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Store embeddings in documents
        for i, doc in enumerate(self.documents):
            doc['embedding'] = embeddings[i]
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"🔍 Built FAISS index with {len(self.documents)} documents")
    
    def retrieve(self, query, k=4):
        """Retrieve top-k relevant policy clauses with similarity scores"""
        if self.index is None:
            raise RuntimeError("Index not built")
        
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search with more candidates initially
        search_k = min(k * 3, len(self.documents))  # Get more candidates for diversity
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        seen_texts = set()  # Avoid duplicates
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                
                # Skip duplicates and very low similarity
                if (doc['text'] not in seen_texts and score > 0.1):
                    seen_texts.add(doc['text'])
                    results.append({
                        'text': doc['text'],
                        'reference': doc['reference'],
                        'score': float(score),
                        'type': doc.get('type', 'unknown')
                    })
                
                # Stop when we have enough unique results
                if len(results) >= k:
                    break
        
        # Sort by relevance score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # FIXED: Proper f-string formatting
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