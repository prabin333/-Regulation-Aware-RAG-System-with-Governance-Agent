from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import Dict, Any, Union, List, Optional
import os

class AgentState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def dict(self) -> Dict[str, Any]:
        return self.__dict__

class PolicyRetriever:

    def __init__(
        self,
        file_path: str = "data/data_policy.txt",
        index_path: str = "data/faiss_index_policy"
    ):
        self.file_path = file_path
        self.index_path = index_path
        self.timestamp_file = os.path.join(index_path, ".timestamp")

        self.vectorstore = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None

        os.makedirs(self.index_path, exist_ok=True)
        self._initialize_retriever()

    def _initialize_retriever(self):
        if not os.path.exists(self.file_path):
            print(f"❌ Policy file not found: {self.file_path}")
            return 
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        print("⚙️ Initializing Policy Retriever...")

        if self._index_exists() and not self._policy_changed():
            self._load_index()
        else:
            self._build_index()

        print("✅ Policy Retriever initialized.")

    def _build_index(self):
        print("🔨 Building FAISS index (Policy file modified or index missing)...")

        loader = TextLoader(self.file_path, encoding="utf-8")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        if not self.embeddings:
            raise RuntimeError("Embeddings model failed to initialize.")

        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        self.vectorstore.save_local(self.index_path)
        self._save_timestamp()

        print(f"✅ Index built and saved at: {self.index_path}")
        print(f"📄 Total chunks indexed: {len(splits)}")

    def _load_index(self):
        print(f"📦 Loading FAISS index from: {self.index_path}")

        try:
            if not self.embeddings:
                raise RuntimeError("Embeddings model failed to initialize.")

            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Index loaded successfully")
        except Exception as e:
            print(f"❌ Error loading index: {e}. Attempting to rebuild.")
            self._build_index()

    def _index_exists(self):
        return os.path.exists(self.index_path) and any(
            f.endswith((".faiss", ".pkl")) 
            for f in os.listdir(self.index_path)
        )

    def _policy_changed(self):
        if not os.path.exists(self.timestamp_file):
            print("   (Timestamp missing, rebuilding index)")
            return True

        try:
            with open(self.timestamp_file, "r") as f:
                saved_mtime = float(f.read().strip())

            current_mtime = os.path.getmtime(self.file_path)
            change = current_mtime > saved_mtime + 0.1 
            if change:
                print("   (Policy file changed, rebuilding index)")
            else:
                print("   (Policy file unchanged, loading index)")
            return change

        except Exception:
            print("   (Error reading timestamp, rebuilding index)")
            return True

    def _save_timestamp(self):
        current_mtime = os.path.getmtime(self.file_path)
        with open(self.timestamp_file, "w") as f:
            f.write(str(current_mtime))

    def retrieve(self, query: str, k: int = 4) -> List[Any]:
        if self.vectorstore is None:
            raise RuntimeError("❌ FAISS index not initialized or load/build failed.")

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

class PolicyRetrieverNode:
    """
    LangGraph node wrapper for PolicyRetriever.
    Handles the RAG step and returns the updated state dict.
    """

    def __init__(self, policy_path: str):
        self.retriever = PolicyRetriever(file_path=policy_path)

    def run(self, state: Union[AgentState, Dict[str, Any]]) -> Dict[str, Any]:

        state_dict = state.dict() if hasattr(state, 'dict') else state

        action = state_dict.get("action", "")
        context = state_dict.get("context", "")
        query = f"Action: {action}\nContext: {context}".strip()
        
        print(f"\n🔍 Step 1: RETRIEVE - Searching for relevant policies...")
        print(f"   Query: '{action[:50]}...'")

        if not query or self.retriever.vectorstore is None:
            print("   Warning: Invalid query or vector store missing. Returning empty policies.")
            state_dict["retrieved_policies"] = "No relevant policies found."
            return state_dict

        try:
            results = self.retriever.retrieve(query, k=4)

            policies_text = "\n\n".join([
                f"[Policy {i+1} | Score={score:.3f}]\n{doc.page_content}"
                for i, (doc, score) in enumerate(results)
            ])

            state_dict["retrieved_policies"] = policies_text
            print(f"   Success: Retrieved {len(results)} policy document(s).")
            
        except RuntimeError as e:
            print(f"   Warning: Retrieval failed: {e}")
            state_dict["retrieved_policies"] = "No relevant policies found. Index load failed."
        except Exception as e:
            print(f"   Warning: Unexpected error during retrieval: {e}")
            state_dict["retrieved_policies"] = "Retrieval failed due to an unexpected error."


        return state_dict
