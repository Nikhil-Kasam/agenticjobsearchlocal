import os
import PyPDF2

# Use chromadb directly instead of via langchain to minimize dependencies or use langchain chromadb
# We'll use langchain-chroma for simplicity as it integrates nicely with the embeddings.
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

DB_DIR = "./chroma_db"
# Point to our local Ollama server which serves the embedder
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
VLLM_API_BASE = os.getenv("VLLM_URL", "http://host.docker.internal:11434/v1")

class VectorDBClient:
    def __init__(self):
        print(f"  [DB] Connecting to LLM server at: {VLLM_API_BASE}")
        
        # We use the official OpenAIEmbeddings wrapper but point it to our local server
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_base=VLLM_API_BASE,
            openai_api_key="EMPTY", # Ollama/vLLM doesn't require an actual API key
            check_embedding_ctx_length=False # skip token arrays, Ollama needs string
        )
        self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
    
    def _is_already_ingested(self, pdf_path: str) -> bool:
        """Check if this PDF was already ingested by looking for existing docs with this source."""
        try:
            results = self.db.get(where={"source": pdf_path})
            return len(results.get("ids", [])) > 0
        except Exception as e:
            print(f"  [DB] Warning: could not check ingestion status: {e}")
            return False

    def ingest_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF missing at {pdf_path}")
        
        if self._is_already_ingested(pdf_path):
            print(f"  → {pdf_path} already ingested. Skipping.")
            return 0
        
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        
        # Simple chunking logic for prototype
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
            
        docs = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks if chunk.strip()]
        
        if docs:
            self.db.add_documents(docs)
        return len(docs)
    
    def search_resume(self, job_description: str, k: int = 3) -> str:
        results = self.db.similarity_search(job_description, k=k)
        return "\n\n".join([doc.page_content for doc in results])
