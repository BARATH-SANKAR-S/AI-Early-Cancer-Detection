"""
PDF Indexing Service
Processes PDFs and creates vector embeddings for RAG system
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import hashlib

logger = logging.getLogger(__name__)

class PDFIndexingService:
    """Service for indexing PDF documents into vector database"""
    
    def __init__(self, pdf_dir: str, vector_db_dir: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.pdf_dir = Path(pdf_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="cancer_knowledge_base",
            metadata={"description": "Cancer-related PDF documents"}
        )
        
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text from PDF with page numbers"""
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        # Split text into chunks
                        text_chunks = self._split_text_into_chunks(text)
                        
                        for chunk in text_chunks:
                            chunks.append({
                                "text": chunk,
                                "page": page_num,
                                "pdf_name": pdf_path.name,
                                "pdf_path": str(pdf_path)
                            })
                            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            
        return chunks
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclamation = chunk.rfind('!')
                
                last_boundary = max(last_period, last_question, last_exclamation)
                if last_boundary > self.chunk_size * 0.7:  # At least 70% of chunk size
                    chunk = text[start:start + last_boundary + 1]
                    end = start + last_boundary + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
        return [chunk for chunk in chunks if chunk.strip()]
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        return self.embedding_model.encode(texts, show_progress_bar=True).tolist()
    
    def index_pdf(self, pdf_path: Path) -> bool:
        """Index a single PDF into vector database"""
        try:
            logger.info(f"Indexing PDF: {pdf_path}")
            
            # Extract text chunks
            chunks = self.extract_text_from_pdf(pdf_path)
            
            if not chunks:
                logger.warning(f"No text extracted from {pdf_path}")
                return False
            
            # Create embeddings
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.create_embeddings(texts)
            
            # Create IDs
            ids = [f"{pdf_path.stem}_page{chunk['page']}_{i}" 
                   for i, chunk in enumerate(chunks)]
            
            # Create metadata
            metadatas = [
                {
                    "pdf_name": chunk["pdf_name"],
                    "page": chunk["page"],
                    "pdf_path": chunk["pdf_path"],
                    "chunk_index": i
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Successfully indexed {len(chunks)} chunks from {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing PDF {pdf_path}: {e}")
            return False
    
    def index_all_pdfs(self) -> Dict[str, Any]:
        """Index all PDFs in the directory"""
        results = {
            "total_pdfs": 0,
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        results["total_pdfs"] = len(pdf_files)
        
        logger.info(f"Found {len(pdf_files)} PDFs to index")
        
        for pdf_path in pdf_files:
            try:
                if self.index_pdf(pdf_path):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to index {pdf_path.name}")
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error with {pdf_path.name}: {str(e)}")
        
        # Get total chunks
        results["total_chunks"] = self.collection.count()
        
        logger.info(f"Indexing complete: {results}")
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed collection"""
        try:
            count = self.collection.count()
            
            # Get sample metadata to analyze
            if count > 0:
                sample = self.collection.get(limit=1, include=["metadatas"])
                if sample["metadatas"]:
                    pdf_names = set()
                    for metadata in self.collection.get(include=["metadatas"])["metadatas"]:
                        pdf_names.add(metadata.get("pdf_name", "unknown"))
                    
                    return {
                        "total_chunks": count,
                        "total_pdfs": len(pdf_names),
                        "pdf_names": list(pdf_names)
                    }
            
            return {
                "total_chunks": count,
                "total_pdfs": 0,
                "pdf_names": []
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete and recreate collection
            self.chroma_client.delete_collection("cancer_knowledge_base")
            self.collection = self.chroma_client.create_collection(
                name="cancer_knowledge_base",
                metadata={"description": "Cancer-related PDF documents"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False


if __name__ == "__main__":
    # Test the indexing service
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from chatbot_config import PDF_DIR, VECTOR_DB_DIR
    
    logging.basicConfig(level=logging.INFO)
    
    service = PDFIndexingService(PDF_DIR, VECTOR_DB_DIR)
    
    # Index all PDFs
    results = service.index_all_pdfs()
    print(f"Indexing Results: {results}")
    
    # Get stats
    stats = service.get_collection_stats()
    print(f"Collection Stats: {stats}")