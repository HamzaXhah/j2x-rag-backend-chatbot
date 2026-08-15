import logging
from typing import Dict, List, Optional, Any
import os
from pathlib import Path

# Updated imports for PGVector
from langchain_postgres import PGVector
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store implementation using PostgreSQL with PGVector extension"""
    
    def __init__(self):
        self.collections = {}
        self.embeddings = self._get_embeddings()
        self.connection_string = self._get_connection_string()
    
    def _get_connection_string(self) -> str:
        """Get PostgreSQL connection string from settings"""
        if settings.DATABASE_URL:
            # Convert PostgresDsn to string
            return str(settings.DATABASE_URL)
        
        # Fallback: construct from individual components
        # Note: These should be provided via environment variables for security
        db_user = os.environ.get("DATABASE_USER")
        db_password = os.environ.get("DATABASE_PASSWORD")
        db_host = os.environ.get("DATABASE_HOST", "localhost")
        db_port = os.environ.get("DATABASE_PORT", "5432")
        db_name = os.environ.get("DATABASE_NAME")
        
        if not db_user or not db_password or not db_name:
            raise ValueError(
                "Database credentials not configured. Please set DATABASE_URL or "
                "DATABASE_USER, DATABASE_PASSWORD, and DATABASE_NAME environment variables."
            )
        
        return f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def _get_embeddings(self):
        """Initialize BGE Large locally through sentence-transformers."""
        configured_path = Path(settings.BGE_MODEL_PATH)
        if not configured_path.is_absolute():
            backend_dir = Path(__file__).resolve().parents[2]
            configured_path = backend_dir / configured_path
        has_local_model = (configured_path / "config.json").is_file() and any(
            (configured_path / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        model_source = str(configured_path) if has_local_model else settings.BGE_MODEL_NAME

        if not has_local_model:
            logger.warning(
                "BGE model not found at %s. It will be downloaded from %s; "
                "run `python scripts/download_bge_model.py` during setup to pre-download it.",
                configured_path,
                settings.BGE_MODEL_NAME,
            )

        try:
            logger.info(
                "Loading BGE embeddings from %s on %s",
                model_source,
                settings.BGE_DEVICE,
            )
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_source,
                model_kwargs={"device": settings.BGE_DEVICE},
                encode_kwargs={
                    "normalize_embeddings": settings.BGE_NORMALIZE_EMBEDDINGS,
                },
                query_instruction="Represent this sentence for searching relevant passages: ",
            )
            actual_dimensions = len(embeddings.embed_query("Test embedding"))
            if actual_dimensions != settings.EMBEDDING_MODEL_DIMENSIONS:
                raise ValueError(
                    f"BGE produced {actual_dimensions} dimensions, but "
                    f"EMBEDDING_MODEL_DIMENSIONS={settings.EMBEDDING_MODEL_DIMENSIONS}."
                )
            logger.info(
                "BGE embeddings initialized successfully with %s dimensions",
                actual_dimensions,
            )
            return embeddings
        except Exception as exc:
            logger.error("Failed to initialize local BGE embeddings: %s", exc)
            raise RuntimeError(f"Failed to initialize local BGE embeddings: {exc}") from exc

    def get_embeddings(self):
        """Return the configured embedding implementation."""
        return self.embeddings
    
    def get_collection(self, collection_name: str) -> PGVector:
        """Get or create a PGVector collection"""
        if collection_name not in self.collections:
            try:
                logger.info(f"Initializing PGVector collection: {collection_name}")
                
                # Create PGVector store - it will auto-create tables if they don't exist
                self.collections[collection_name] = PGVector(
                    connection=self.connection_string,
                    collection_name=collection_name,
                    embeddings=self.embeddings,
                    use_jsonb=True,
                )
                
                logger.info(f"PGVector collection '{collection_name}' initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize PGVector collection '{collection_name}': {str(e)}")
                raise
        
        return self.collections[collection_name]
    
    def add_documents(
        self, 
        documents: List[Document], 
        collection_name: str = "documents",
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the vector store"""
        collection = self.get_collection(collection_name)
        
        try:
            logger.info(f"Adding {len(documents)} documents to PGVector collection '{collection_name}'")
            return collection.add_documents(documents, ids=ids)
        except Exception as e:
            logger.error(f"Error adding documents to PGVector: {str(e)}")
            raise
    
    def search(
        self, 
        query: str,
        collection_name: str = "documents",
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[Document]:
        """Search for documents similar to the query"""
        collection = self.get_collection(collection_name)
        
        try:
            logger.info(f"Searching for '{query}' in PGVector collection '{collection_name}'")
            return collection.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
        except Exception as e:
            logger.error(f"Error searching PGVector: {str(e)}")
            raise
    
    def search_with_score(
        self, 
        query: str,
        collection_name: str = "documents",
        filter: Optional[Dict[str, Any]] = None,
        k: int = 5
    ) -> List[tuple[Document, float]]:
        """Search for documents similar to the query and return with similarity scores"""
        collection = self.get_collection(collection_name)
        
        try:
            logger.info(f"Searching for '{query}' with scores in PGVector collection '{collection_name}'")
            return collection.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
        except Exception as e:
            logger.error(f"Error searching PGVector with scores: {str(e)}")
            raise
    
    def delete(
        self,
        ids: List[str],
        collection_name: str = "documents"
    ) -> None:
        """Delete documents from the vector store"""
        collection = self.get_collection(collection_name)
        
        try:
            logger.info(f"Deleting {len(ids)} documents from PGVector collection '{collection_name}'")
            collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"Error deleting documents from PGVector: {str(e)}")
            raise


# Singleton instance
vector_store = VectorStore()
