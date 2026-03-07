"""
Chunking Strategies for RAG

Provides multiple strategies for splitting text into chunks:
- Word: Simple word-based splitting (fast, works well)
- Sentence: Split at sentence boundaries (better coherence)
- Semantic: Use AI to find semantic boundaries (best quality, slower)

Usage:
    from kk_utils.rag.chunking import ChunkingStrategy
    
    # Create chunker
    chunker = ChunkingStrategy.WORD.create_chunker(chunk_size=500, chunk_overlap=50)
    
    # Chunk text
    chunks = chunker.chunk(long_text)
"""
from abc import ABC, abstractmethod
from typing import List
import logging
import re

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target number of words per chunk
            chunk_overlap: Number of overlapping words between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        pass


class WordChunker(BaseChunker):
    """
    Word-based chunking.
    
    Simple and fast, splits text by word count.
    Works well for most use cases.
    """
    
    def chunk(self, text: str) -> List[str]:
        """Split text into chunks by word count with overlap."""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
        
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            # Break if we've reached the end
            if i + self.chunk_size >= len(words):
                break
        
        return chunks if chunks else [text]


class SentenceChunker(BaseChunker):
    """
    Sentence-based chunking.
    
    Splits text at sentence boundaries, then groups into chunks.
    Better coherence than word-based chunking.
    Preserves complete thoughts in each chunk.
    """
    
    def chunk(self, text: str) -> List[str]:
        """Split text at sentence boundaries, then group into chunks."""
        # Split into sentences (handles ., !, ? followed by space or end)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_word_count + sentence_word_count > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap by retaining last few sentences
                overlap_words = []
                overlap_count = 0
                for s in reversed(current_chunk):
                    if overlap_count + len(s.split()) <= self.chunk_overlap:
                        overlap_words.insert(0, s)
                        overlap_count += len(s.split())
                    else:
                        break
                current_chunk = overlap_words
                current_word_count = overlap_count
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using AI.
    
    Uses AI to identify topic boundaries for optimal chunking.
    Best quality but requires API access and is slower.
    
    Note: This is a placeholder - full implementation would require
    calling an LLM to identify topic boundaries.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        logger.warning("Semantic chunking not fully implemented, falling back to sentence chunking")
    
    def chunk(self, text: str) -> List[str]:
        """Fall back to sentence chunking."""
        # For now, fall back to sentence chunking
        # Full implementation would:
        # 1. Call LLM to identify topic boundaries
        # 2. Split at those boundaries
        # 3. Ensure chunks are within size limits
        chunker = SentenceChunker(self.chunk_size, self.chunk_overlap)
        return chunker.chunk(text)


class ChunkingStrategy:
    """Enum-like class for chunking strategies."""
    
    WORD = "word"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    
    @classmethod
    def create_chunker(cls, strategy: str, chunk_size: int = 500, chunk_overlap: int = 50) -> BaseChunker:
        """
        Create chunker for specified strategy.
        
        Args:
            strategy: Chunking strategy ("word", "sentence", "semantic")
            chunk_size: Target words per chunk
            chunk_overlap: Overlapping words between chunks
        
        Returns:
            Appropriate chunker instance
        """
        if strategy == cls.SENTENCE:
            return SentenceChunker(chunk_size, chunk_overlap)
        elif strategy == cls.SEMANTIC:
            return SemanticChunker(chunk_size, chunk_overlap)
        else:  # default to WORD
            return WordChunker(chunk_size, chunk_overlap)


# Convenience functions for direct usage
def chunk_by_word(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text by word count."""
    chunker = WordChunker(chunk_size, chunk_overlap)
    return chunker.chunk(text)


def chunk_by_sentence(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text at sentence boundaries."""
    chunker = SentenceChunker(chunk_size, chunk_overlap)
    return chunker.chunk(text)


def chunk_by_semantic(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split text using AI (falls back to sentence)."""
    chunker = SemanticChunker(chunk_size, chunk_overlap)
    return chunker.chunk(text)
