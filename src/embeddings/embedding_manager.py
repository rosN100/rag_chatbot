import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class EmbeddingManager:
    def __init__(self, persist_dir: str = 'vector_store'):
        self.embeddings = HuggingFaceEmbeddings()
        self.persist_dir = persist_dir
    
    def create_vector_store(self, documents, force_refresh: bool = False):
        """Create or load a FAISS vector store.
        
        Args:
            documents: Documents to create embeddings for
            force_refresh: If True, create new embeddings even if saved index exists
        
        Returns:
            FAISS vector store
        """
        # Check if we have a saved index
        if os.path.exists(self.persist_dir) and not force_refresh:
            print("Loading existing vector store...")
            return FAISS.load_local(self.persist_dir, self.embeddings)
        
        print("Creating new vector store...")
        vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save the index
        os.makedirs(self.persist_dir, exist_ok=True)
        vector_store.save_local(self.persist_dir)
        
        return vector_store
    
    def create_vectorstore(self, documents):
        return self.create_vector_store(documents)
