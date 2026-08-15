"""
Chunking Service for document processing.

Provides multiple chunking strategies:
- recursive: Traditional recursive character text splitting (default)
- semantic: Splits based on semantic similarity using embeddings
- token: Splits based on token count (useful for LLM context limits)
- sentence: Splits at sentence boundaries

Usage:
    from app.services.chunking_service import chunking_service
    
    # Split text into chunks
    chunks = chunking_service.split_text(text)
    
    # Split documents
    docs = chunking_service.split_documents(documents)
"""

import logging
from typing import List, Optional, Callable, Literal
from dataclasses import dataclass, field

from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    strategy: Literal["recursive", "semantic", "token", "sentence"] = "recursive"
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])
    length_function: Callable[[str], int] = len
    # For semantic chunking
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile", "gradient"] = "percentile"
    # Minimum chunk size
    min_chunk_size: int = 100
    # Stable tokenizer encoding for token-based chunking.
    tokenizer_encoding: str = "cl100k_base"
    
    @classmethod
    def from_settings(cls) -> "ChunkingConfig":
        """Create a ChunkingConfig from application settings"""
        return cls(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            strategy=settings.CHUNKING_STRATEGY,
            separators=settings.CHUNK_SEPARATORS,
            breakpoint_threshold_type=settings.SEMANTIC_BREAKPOINT_TYPE,
            min_chunk_size=settings.MIN_CHUNK_SIZE,
        )


class ChunkingService:
    """
    Service to handle text chunking with multiple strategies.
    
    Strategies:
    - recursive: Uses RecursiveCharacterTextSplitter for general-purpose chunking
    - semantic: Uses SemanticChunker for context-aware chunking (requires embeddings)
    - token: Uses TokenTextSplitter for token-based chunking
    - sentence: Uses sentence-aware splitting
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the ChunkingService.
        
        Args:
            config: Optional ChunkingConfig. If not provided, uses settings.
        """
        self.config = config or ChunkingConfig.from_settings()
        self._text_splitter = None
        self._semantic_chunker = None
        self._init_splitter()
        
    def _init_splitter(self):
        """Initialize the appropriate text splitter based on strategy"""
        strategy = self.config.strategy
        
        if strategy == "recursive":
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=self.config.length_function,
                separators=self.config.separators,
            )
            logger.info(f"Initialized RecursiveCharacterTextSplitter with chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap}")
            
        elif strategy == "token":
            self._text_splitter = TokenTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                encoding_name=self.config.tokenizer_encoding,
            )
            logger.info(
                "Initialized TokenTextSplitter with chunk_size=%s tokens, encoding=%s",
                self.config.chunk_size,
                self.config.tokenizer_encoding,
            )
            
        elif strategy == "sentence":
            # Use sentence transformers token text splitter for sentence-aware splitting
            self._text_splitter = SentenceTransformersTokenTextSplitter(
                chunk_overlap=min(self.config.chunk_overlap, 50),  # Sentence splitter uses smaller overlap
                tokens_per_chunk=self.config.chunk_size // 4,  # Approximate tokens from chars
            )
            logger.info(f"Initialized SentenceTransformersTokenTextSplitter")
            
        elif strategy == "semantic":
            # Lazy initialization for semantic chunker (requires embeddings)
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators,
            )
            logger.info("Semantic chunking configured - will use lazy initialization with embeddings")
            
        else:
            # Default to recursive
            logger.warning(f"Unknown chunking strategy '{strategy}', falling back to recursive")
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators,
            )
    
    def _get_semantic_chunker(self):
        """Lazily initialize semantic chunker when needed"""
        if self._semantic_chunker is None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from app.services.vector_store import vector_store
                
                embeddings = vector_store.get_embeddings()
                self._semantic_chunker = SemanticChunker(
                    embeddings=embeddings,
                    breakpoint_threshold_type=self.config.breakpoint_threshold_type,
                )
                logger.info(f"Initialized SemanticChunker with breakpoint_type={self.config.breakpoint_threshold_type}")
            except ImportError:
                logger.warning("langchain_experimental not installed. Install with: pip install langchain-experimental")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize semantic chunker: {e}")
                return None
        return self._semantic_chunker
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on the configured strategy.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) < self.config.min_chunk_size:
            return [text] if text and text.strip() else []
        
        if self.config.strategy == "semantic":
            chunker = self._get_semantic_chunker()
            if chunker:
                try:
                    return chunker.split_text(text)
                except Exception as e:
                    logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
        
        return self._text_splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of Document objects (chunked)
        """
        if not documents:
            return []
        
        if self.config.strategy == "semantic":
            chunker = self._get_semantic_chunker()
            if chunker:
                try:
                    return chunker.split_documents(documents)
                except Exception as e:
                    logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
        
        return self._text_splitter.split_documents(documents)
    
    def create_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Create Document objects from texts with optional metadata.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of Document objects (chunked)
        """
        return self._text_splitter.create_documents(texts, metadatas)
    
    def get_strategy(self) -> str:
        """Get the current chunking strategy"""
        return self.config.strategy
    
    def get_config(self) -> dict:
        """Get the current configuration as a dictionary"""
        return {
            "strategy": self.config.strategy,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "separators": self.config.separators,
            "min_chunk_size": self.config.min_chunk_size,
            "breakpoint_threshold_type": self.config.breakpoint_threshold_type if self.config.strategy == "semantic" else None,
            "tokenizer_encoding": self.config.tokenizer_encoding if self.config.strategy == "token" else None,
        }
    
    @classmethod
    def create_with_strategy(
        cls,
        strategy: Literal["recursive", "semantic", "token", "sentence"],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs
    ) -> "ChunkingService":
        """
        Factory method to create a ChunkingService with a specific strategy.
        
        Args:
            strategy: The chunking strategy to use
            chunk_size: Optional custom chunk size
            chunk_overlap: Optional custom chunk overlap
            **kwargs: Additional configuration options
            
        Returns:
            Configured ChunkingService instance
        """
        config = ChunkingConfig.from_settings()
        config.strategy = strategy
        if chunk_size is not None:
            config.chunk_size = chunk_size
        if chunk_overlap is not None:
            config.chunk_overlap = chunk_overlap
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return cls(config)


# Singleton instance using settings
chunking_service = ChunkingService()


def get_text_splitter():
    """
    Get the default text splitter for backward compatibility.
    
    Returns:
        The text splitter from the singleton chunking service
    """
    return chunking_service._text_splitter
