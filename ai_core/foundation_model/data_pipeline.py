"""
Data Pipeline for Foundation Model Training

Handles 10-15 Trillion tokens from:
- Common Crawl (web data)
- Books and academic papers
- Code repositories (GitHub, GitLab)
- High-quality human-annotated data

Features:
- Multi-stage data cleaning
- Deduplication (exact and near-duplicate)
- Quality filtering
- Optimized storage (HDF5, Parquet)
- Streaming for large-scale processing
"""

import asyncio
import hashlib
import logging
import mmap
import os
import re
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Generator, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class DataSource(str, Enum):
    """Data source types."""
    COMMON_CRAWL = "common_crawl"
    BOOKS = "books"
    PAPERS = "papers"
    CODE = "code"
    WIKIPEDIA = "wikipedia"
    SOCIAL_MEDIA = "social_media"
    CUSTOM = "custom"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    GERMAN = "de"
    FRENCH = "fr"
    SPANISH = "es"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    CODE = "code"
    MULTI = "multi"


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # Storage paths
    raw_data_path: str = "/data/raw"
    processed_data_path: str = "/data/processed"
    cache_path: str = "/data/cache"
    
    # Processing
    num_workers: int = 64
    batch_size: int = 10000
    max_sequence_length: int = 131072  # 128K tokens
    min_sequence_length: int = 50
    
    # Quality thresholds
    min_quality_score: float = 0.3
    max_perplexity: float = 1000.0
    min_unique_ratio: float = 0.5
    
    # Deduplication
    dedup_threshold: float = 0.8  # Jaccard similarity threshold
    minhash_num_perm: int = 128
    lsh_threshold: float = 0.5
    
    # Storage
    storage_format: str = "parquet"  # parquet or hdf5
    compression: str = "zstd"
    shard_size_gb: float = 1.0
    
    # Sampling
    target_tokens: int = 15_000_000_000_000  # 15T tokens
    language_distribution: Dict[str, float] = field(default_factory=lambda: {
        "en": 0.45,
        "zh": 0.15,
        "code": 0.20,
        "multi": 0.20,
    })


@dataclass
class DataStats:
    """Statistics for processed data."""
    total_documents: int = 0
    total_tokens: int = 0
    filtered_documents: int = 0
    deduplicated_documents: int = 0
    bytes_processed: int = 0
    processing_time_seconds: float = 0.0
    
    # Per-language stats
    language_stats: Dict[str, int] = field(default_factory=dict)
    source_stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "total_tokens": self.total_tokens,
            "filtered_documents": self.filtered_documents,
            "deduplicated_documents": self.deduplicated_documents,
            "bytes_processed": self.bytes_processed,
            "processing_time_seconds": self.processing_time_seconds,
            "language_stats": self.language_stats,
            "source_stats": self.source_stats,
        }


# =============================================================================
# Document Representation
# =============================================================================

@dataclass
class Document:
    """Single document representation."""
    doc_id: str
    text: str
    source: DataSource
    language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics (computed during processing)
    quality_score: float = 0.0
    perplexity: float = 0.0
    token_count: int = 0
    unique_token_ratio: float = 0.0
    
    # Deduplication
    minhash: Optional[np.ndarray] = None
    fingerprint: Optional[str] = None
    is_duplicate: bool = False
    
    def compute_fingerprint(self) -> str:
        """Compute document fingerprint for exact deduplication."""
        normalized = self.text.lower().strip()
        self.fingerprint = hashlib.sha256(normalized.encode()).hexdigest()
        return self.fingerprint


# =============================================================================
# Data Cleaning
# =============================================================================

class DataCleaner:
    """
    Multi-stage data cleaning pipeline.
    
    Stages:
    1. Basic cleaning (whitespace, encoding)
    2. Language detection and filtering
    3. Quality filtering
    4. Content filtering (toxic, PII)
    5. Normalization
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Load blocklists
        self._load_blocklists()
    
    def _compile_patterns(self):
        """Compile regex patterns for cleaning."""
        # URL pattern
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'[\w.+-]+@[\w-]+\.[\w.-]+'
        )
        
        # Phone pattern
        self.phone_pattern = re.compile(
            r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        )
        
        # Repeated characters
        self.repeated_char_pattern = re.compile(r'(.)\1{4,}')
        
        # Repeated words
        self.repeated_word_pattern = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)
        
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Control characters
        self.control_char_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
        
        # Excessive whitespace
        self.whitespace_pattern = re.compile(r'\s+')
    
    def _load_blocklists(self):
        """Load content blocklists."""
        # Toxic words (simplified - in production, use comprehensive list)
        self.toxic_words: Set[str] = set()
        
        # PII patterns
        self.pii_patterns: List[re.Pattern] = [
            self.email_pattern,
            self.phone_pattern,
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b\d{16}\b'),  # Credit card
        ]
    
    def clean(self, doc: Document) -> Optional[Document]:
        """
        Clean a single document.
        
        Returns None if document should be filtered out.
        """
        text = doc.text
        
        # Stage 1: Basic cleaning
        text = self._basic_clean(text)
        
        if not text or len(text) < self.config.min_sequence_length:
            return None
        
        # Stage 2: Remove HTML
        text = self.html_pattern.sub(' ', text)
        
        # Stage 3: Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Stage 4: Remove control characters
        text = self.control_char_pattern.sub('', text)
        
        # Stage 5: Remove repeated patterns
        text = self.repeated_char_pattern.sub(r'\1\1\1', text)
        
        # Stage 6: Quality check
        if not self._passes_quality_check(text):
            return None
        
        # Stage 7: PII removal (optional, can be configured)
        text = self._remove_pii(text)
        
        # Update document
        doc.text = text
        doc.compute_fingerprint()
        
        return doc
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        
        # Fix encoding issues
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        return text
    
    def _passes_quality_check(self, text: str) -> bool:
        """Check if text passes quality thresholds."""
        if not text:
            return False
        
        # Length check
        if len(text) < self.config.min_sequence_length:
            return False
        
        # Check for excessive repetition
        words = text.split()
        if len(words) < 10:
            return False
        
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        
        if unique_ratio < self.config.min_unique_ratio:
            return False
        
        # Check for mostly punctuation/numbers
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.5:
            return False
        
        return True
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information."""
        for pattern in self.pii_patterns:
            text = pattern.sub('[REDACTED]', text)
        return text
    
    def clean_batch(self, docs: List[Document]) -> List[Document]:
        """Clean a batch of documents in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            results = list(executor.map(self.clean, docs))
        
        return [doc for doc in results if doc is not None]


# =============================================================================
# Deduplication
# =============================================================================

class MinHasher:
    """MinHash implementation for near-duplicate detection."""
    
    def __init__(self, num_perm: int = 128, ngram_size: int = 5):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        
        # Generate random hash functions
        self.a = np.random.randint(1, 2**31, size=num_perm, dtype=np.int64)
        self.b = np.random.randint(0, 2**31, size=num_perm, dtype=np.int64)
        self.prime = 2**61 - 1
    
    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract character n-grams from text."""
        text = text.lower()
        ngrams = set()
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.add(text[i:i + self.ngram_size])
        return ngrams
    
    def _hash_ngram(self, ngram: str) -> int:
        """Hash a single n-gram."""
        return hash(ngram) & 0xFFFFFFFF
    
    def compute(self, text: str) -> np.ndarray:
        """Compute MinHash signature for text."""
        ngrams = self._get_ngrams(text)
        
        if not ngrams:
            return np.full(self.num_perm, 2**31, dtype=np.int64)
        
        # Initialize signature with max values
        signature = np.full(self.num_perm, 2**31, dtype=np.int64)
        
        for ngram in ngrams:
            h = self._hash_ngram(ngram)
            # Apply all hash functions
            hashes = (self.a * h + self.b) % self.prime
            signature = np.minimum(signature, hashes)
        
        return signature
    
    @staticmethod
    def jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        return np.mean(sig1 == sig2)


class LSHIndex:
    """Locality-Sensitive Hashing index for efficient similarity search."""
    
    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        threshold: float = 0.5,
    ):
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.threshold = threshold
        
        # Band buckets: band_idx -> hash -> set of doc_ids
        self.buckets: List[Dict[int, Set[str]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]
    
    def insert(self, doc_id: str, minhash: np.ndarray):
        """Insert a document into the LSH index."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = minhash[start:end]
            
            # Hash the band
            band_hash = hash(tuple(band))
            self.buckets[band_idx][band_hash].add(doc_id)
    
    def query(self, minhash: np.ndarray) -> Set[str]:
        """Find candidate duplicates for a MinHash signature."""
        candidates: Set[str] = set()
        
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = minhash[start:end]
            
            band_hash = hash(tuple(band))
            candidates.update(self.buckets[band_idx][band_hash])
        
        return candidates


class Deduplicator:
    """
    Document deduplication using MinHash LSH.
    
    Handles:
    - Exact duplicates (via fingerprint)
    - Near-duplicates (via MinHash LSH)
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # MinHasher
        self.minhasher = MinHasher(num_perm=config.minhash_num_perm)
        
        # LSH Index
        self.lsh_index = LSHIndex(
            num_perm=config.minhash_num_perm,
            threshold=config.lsh_threshold,
        )
        
        # Exact duplicate tracking
        self.seen_fingerprints: Set[str] = set()
        
        # Document MinHashes for similarity comparison
        self.doc_minhashes: Dict[str, np.ndarray] = {}
        
        # Statistics
        self.exact_dups = 0
        self.near_dups = 0
    
    def is_duplicate(self, doc: Document) -> bool:
        """Check if document is a duplicate."""
        # Check exact duplicate first
        if doc.fingerprint in self.seen_fingerprints:
            self.exact_dups += 1
            return True
        
        # Compute MinHash if not already done
        if doc.minhash is None:
            doc.minhash = self.minhasher.compute(doc.text)
        
        # Query LSH index for candidates
        candidates = self.lsh_index.query(doc.minhash)
        
        # Check similarity with candidates
        for candidate_id in candidates:
            if candidate_id in self.doc_minhashes:
                similarity = MinHasher.jaccard_similarity(
                    doc.minhash,
                    self.doc_minhashes[candidate_id]
                )
                if similarity >= self.config.dedup_threshold:
                    self.near_dups += 1
                    return True
        
        # Not a duplicate - add to index
        self.seen_fingerprints.add(doc.fingerprint)
        self.lsh_index.insert(doc.doc_id, doc.minhash)
        self.doc_minhashes[doc.doc_id] = doc.minhash
        
        return False
    
    def deduplicate_batch(self, docs: List[Document]) -> List[Document]:
        """Deduplicate a batch of documents."""
        unique_docs = []
        
        for doc in docs:
            if not self.is_duplicate(doc):
                unique_docs.append(doc)
            else:
                doc.is_duplicate = True
        
        return unique_docs
    
    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics."""
        return {
            "exact_duplicates": self.exact_dups,
            "near_duplicates": self.near_dups,
            "unique_documents": len(self.seen_fingerprints),
        }


# =============================================================================
# Data Source Processors
# =============================================================================

class DataSourceProcessor(ABC):
    """Abstract base class for data source processors."""
    
    @abstractmethod
    def process(self, path: str) -> Iterator[Document]:
        """Process data from source and yield documents."""
        pass
    
    @abstractmethod
    def estimate_size(self, path: str) -> int:
        """Estimate number of documents in source."""
        pass


class CommonCrawlProcessor(DataSourceProcessor):
    """
    Process Common Crawl WARC files.
    
    Common Crawl is ~250TB+ of web data per crawl.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.source = DataSource.COMMON_CRAWL
    
    def process(self, path: str) -> Iterator[Document]:
        """Process WARC files from Common Crawl."""
        warc_files = list(Path(path).glob("**/*.warc.gz"))
        
        for warc_file in warc_files:
            try:
                yield from self._process_warc(warc_file)
            except Exception as e:
                logger.error(f"Error processing {warc_file}: {e}")
    
    def _process_warc(self, warc_file: Path) -> Iterator[Document]:
        """Process a single WARC file."""
        try:
            import warcio
            from warcio.archiveiterator import ArchiveIterator
            
            with open(warc_file, 'rb') as f:
                for record in ArchiveIterator(f):
                    if record.rec_type == 'response':
                        try:
                            content = record.content_stream().read()
                            text = content.decode('utf-8', errors='ignore')
                            
                            # Extract text from HTML
                            text = self._extract_text_from_html(text)
                            
                            if text and len(text) >= self.config.min_sequence_length:
                                doc_id = record.rec_headers.get_header('WARC-Record-ID', '')
                                
                                yield Document(
                                    doc_id=doc_id,
                                    text=text,
                                    source=self.source,
                                    language=Language.MULTI,  # Detect later
                                    metadata={
                                        "url": record.rec_headers.get_header('WARC-Target-URI', ''),
                                        "date": record.rec_headers.get_header('WARC-Date', ''),
                                    }
                                )
                        except Exception:
                            continue
        except ImportError:
            logger.warning("warcio not installed, using fallback")
            # Fallback: read as text
            yield from self._process_warc_fallback(warc_file)
    
    def _process_warc_fallback(self, warc_file: Path) -> Iterator[Document]:
        """Fallback WARC processing without warcio."""
        import gzip
        
        with gzip.open(warc_file, 'rt', errors='ignore') as f:
            content = f.read()
            
            # Simple extraction (not production-ready)
            for i, chunk in enumerate(content.split('\r\n\r\n')):
                if len(chunk) >= self.config.min_sequence_length:
                    yield Document(
                        doc_id=f"{warc_file.stem}_{i}",
                        text=chunk,
                        source=self.source,
                        language=Language.MULTI,
                        metadata={"file": str(warc_file)},
                    )
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            text = soup.get_text(separator=' ', strip=True)
            return text
        except ImportError:
            # Fallback: regex-based extraction
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            return text
    
    def estimate_size(self, path: str) -> int:
        """Estimate number of documents."""
        warc_files = list(Path(path).glob("**/*.warc.gz"))
        # Rough estimate: ~10000 docs per WARC file
        return len(warc_files) * 10000


class CodeDataProcessor(DataSourceProcessor):
    """
    Process code repositories.
    
    Sources:
    - GitHub public repositories
    - GitLab public repositories
    - The Stack dataset
    """
    
    # Supported programming languages
    SUPPORTED_LANGUAGES = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.sh': 'bash',
        '.sql': 'sql',
        '.r': 'r',
        '.m': 'matlab',
        '.cs': 'csharp',
    }
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.source = DataSource.CODE
    
    def process(self, path: str) -> Iterator[Document]:
        """Process code files from repositories."""
        repo_path = Path(path)
        
        for ext, lang in self.SUPPORTED_LANGUAGES.items():
            for code_file in repo_path.glob(f"**/*{ext}"):
                try:
                    yield from self._process_code_file(code_file, lang)
                except Exception as e:
                    logger.debug(f"Error processing {code_file}: {e}")
    
    def _process_code_file(self, file_path: Path, lang: str) -> Iterator[Document]:
        """Process a single code file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip files that are too small or too large
            if len(content) < 50 or len(content) > 1_000_000:
                return
            
            # Skip auto-generated files
            if self._is_autogenerated(content):
                return
            
            # Extract documentation and code
            doc_text = self._extract_with_context(content, lang)
            
            yield Document(
                doc_id=hashlib.md5(str(file_path).encode()).hexdigest(),
                text=doc_text,
                source=self.source,
                language=Language.CODE,
                metadata={
                    "file_path": str(file_path),
                    "programming_language": lang,
                    "file_size": len(content),
                },
            )
        except Exception as e:
            logger.debug(f"Failed to process {file_path}: {e}")
    
    def _is_autogenerated(self, content: str) -> bool:
        """Check if file is auto-generated."""
        autogen_markers = [
            'auto-generated',
            'autogenerated',
            'do not edit',
            'generated by',
            'automatically generated',
            '@generated',
        ]
        
        first_lines = content[:500].lower()
        return any(marker in first_lines for marker in autogen_markers)
    
    def _extract_with_context(self, content: str, lang: str) -> str:
        """Extract code with documentation context."""
        # Format: language marker + code
        return f"```{lang}\n{content}\n```"
    
    def estimate_size(self, path: str) -> int:
        """Estimate number of code documents."""
        count = 0
        for ext in self.SUPPORTED_LANGUAGES:
            count += len(list(Path(path).glob(f"**/*{ext}")))
        return count


# =============================================================================
# Storage Backends
# =============================================================================

class StorageBackend(ABC):
    """Abstract storage backend."""
    
    @abstractmethod
    def write(self, docs: List[Document], shard_id: str):
        """Write documents to storage."""
        pass
    
    @abstractmethod
    def read(self, shard_id: str) -> Iterator[Document]:
        """Read documents from storage."""
        pass
    
    @abstractmethod
    def list_shards(self) -> List[str]:
        """List available shards."""
        pass


class ParquetStorage(StorageBackend):
    """
    Parquet storage backend.
    
    Optimized for:
    - Columnar storage efficiency
    - Fast reads with predicate pushdown
    - Good compression ratios
    """
    
    def __init__(self, base_path: str, compression: str = "zstd"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
    
    def write(self, docs: List[Document], shard_id: str):
        """Write documents to Parquet file."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Convert to Arrow table
            data = {
                'doc_id': [d.doc_id for d in docs],
                'text': [d.text for d in docs],
                'source': [d.source.value for d in docs],
                'language': [d.language.value for d in docs],
                'quality_score': [d.quality_score for d in docs],
                'token_count': [d.token_count for d in docs],
            }
            
            table = pa.table(data)
            
            # Write with compression
            output_path = self.base_path / f"{shard_id}.parquet"
            pq.write_table(
                table,
                output_path,
                compression=self.compression,
            )
            
            logger.info(f"Wrote {len(docs)} docs to {output_path}")
            
        except ImportError:
            logger.warning("pyarrow not installed, using JSON fallback")
            self._write_json_fallback(docs, shard_id)
    
    def _write_json_fallback(self, docs: List[Document], shard_id: str):
        """Fallback JSON storage."""
        import json
        
        output_path = self.base_path / f"{shard_id}.jsonl"
        with open(output_path, 'w') as f:
            for doc in docs:
                record = {
                    'doc_id': doc.doc_id,
                    'text': doc.text,
                    'source': doc.source.value,
                    'language': doc.language.value,
                }
                f.write(json.dumps(record) + '\n')
    
    def read(self, shard_id: str) -> Iterator[Document]:
        """Read documents from Parquet file."""
        try:
            import pyarrow.parquet as pq
            
            file_path = self.base_path / f"{shard_id}.parquet"
            table = pq.read_table(file_path)
            
            for i in range(table.num_rows):
                yield Document(
                    doc_id=table['doc_id'][i].as_py(),
                    text=table['text'][i].as_py(),
                    source=DataSource(table['source'][i].as_py()),
                    language=Language(table['language'][i].as_py()),
                    quality_score=table['quality_score'][i].as_py(),
                    token_count=table['token_count'][i].as_py(),
                )
        except ImportError:
            yield from self._read_json_fallback(shard_id)
    
    def _read_json_fallback(self, shard_id: str) -> Iterator[Document]:
        """Fallback JSON reading."""
        import json
        
        file_path = self.base_path / f"{shard_id}.jsonl"
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                yield Document(
                    doc_id=record['doc_id'],
                    text=record['text'],
                    source=DataSource(record['source']),
                    language=Language(record['language']),
                )
    
    def list_shards(self) -> List[str]:
        """List available shards."""
        shards = []
        for f in self.base_path.glob("*.parquet"):
            shards.append(f.stem)
        for f in self.base_path.glob("*.jsonl"):
            shards.append(f.stem)
        return shards


class HDF5Storage(StorageBackend):
    """
    HDF5 storage backend.
    
    Optimized for:
    - Large tensor storage
    - Random access
    - Hierarchical organization
    """
    
    def __init__(self, base_path: str, compression: str = "gzip"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.compression = compression
    
    def write(self, docs: List[Document], shard_id: str):
        """Write documents to HDF5 file."""
        try:
            import h5py
            
            output_path = self.base_path / f"{shard_id}.h5"
            
            with h5py.File(output_path, 'w') as f:
                # Store texts as variable-length strings
                dt = h5py.special_dtype(vlen=str)
                
                texts = [d.text for d in docs]
                doc_ids = [d.doc_id for d in docs]
                sources = [d.source.value for d in docs]
                languages = [d.language.value for d in docs]
                
                f.create_dataset('text', data=texts, dtype=dt, compression=self.compression)
                f.create_dataset('doc_id', data=doc_ids, dtype=dt)
                f.create_dataset('source', data=sources, dtype=dt)
                f.create_dataset('language', data=languages, dtype=dt)
                f.create_dataset(
                    'quality_score', 
                    data=[d.quality_score for d in docs],
                    compression=self.compression
                )
                f.create_dataset(
                    'token_count',
                    data=[d.token_count for d in docs],
                    compression=self.compression
                )
                
                f.attrs['num_docs'] = len(docs)
                f.attrs['created_at'] = datetime.now(timezone.utc).isoformat()
            
            logger.info(f"Wrote {len(docs)} docs to {output_path}")
            
        except ImportError:
            logger.warning("h5py not installed, falling back to Parquet")
            ParquetStorage(str(self.base_path), "gzip").write(docs, shard_id)
    
    def read(self, shard_id: str) -> Iterator[Document]:
        """Read documents from HDF5 file."""
        try:
            import h5py
            
            file_path = self.base_path / f"{shard_id}.h5"
            
            with h5py.File(file_path, 'r') as f:
                num_docs = f.attrs['num_docs']
                
                for i in range(num_docs):
                    yield Document(
                        doc_id=f['doc_id'][i].decode() if isinstance(f['doc_id'][i], bytes) else f['doc_id'][i],
                        text=f['text'][i].decode() if isinstance(f['text'][i], bytes) else f['text'][i],
                        source=DataSource(f['source'][i].decode() if isinstance(f['source'][i], bytes) else f['source'][i]),
                        language=Language(f['language'][i].decode() if isinstance(f['language'][i], bytes) else f['language'][i]),
                        quality_score=float(f['quality_score'][i]),
                        token_count=int(f['token_count'][i]),
                    )
        except ImportError:
            # Fallback to Parquet
            yield from ParquetStorage(str(self.base_path), "gzip").read(shard_id)
    
    def list_shards(self) -> List[str]:
        """List available shards."""
        shards = []
        for f in self.base_path.glob("*.h5"):
            shards.append(f.stem)
        return shards


# =============================================================================
# Main Data Pipeline
# =============================================================================

class DataPipeline:
    """
    Complete data processing pipeline for foundation model training.
    
    Pipeline stages:
    1. Data ingestion from multiple sources
    2. Cleaning and normalization
    3. Deduplication
    4. Quality filtering
    5. Tokenization (optional)
    6. Storage in optimized format
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Initialize components
        self.cleaner = DataCleaner(config)
        self.deduplicator = Deduplicator(config)
        
        # Storage backend
        if config.storage_format == "hdf5":
            self.storage = HDF5Storage(config.processed_data_path, config.compression)
        else:
            self.storage = ParquetStorage(config.processed_data_path, config.compression)
        
        # Source processors
        self.processors: Dict[DataSource, DataSourceProcessor] = {
            DataSource.COMMON_CRAWL: CommonCrawlProcessor(config),
            DataSource.CODE: CodeDataProcessor(config),
        }
        
        # Statistics
        self.stats = DataStats()
    
    def process_source(
        self,
        source: DataSource,
        path: str,
        max_docs: Optional[int] = None,
    ) -> DataStats:
        """
        Process a single data source.
        
        Args:
            source: Data source type
            path: Path to source data
            max_docs: Maximum documents to process (for testing)
        """
        start_time = datetime.now(timezone.utc)
        
        processor = self.processors.get(source)
        if not processor:
            raise ValueError(f"No processor for source: {source}")
        
        logger.info(f"Processing {source.value} from {path}")
        
        # Process in batches
        batch: List[Document] = []
        shard_idx = 0
        processed = 0
        
        for doc in processor.process(path):
            batch.append(doc)
            
            if len(batch) >= self.config.batch_size:
                self._process_batch(batch, source, shard_idx)
                shard_idx += 1
                processed += len(batch)
                batch = []
                
                if max_docs and processed >= max_docs:
                    break
                
                logger.info(f"Processed {processed:,} documents from {source.value}")
        
        # Process remaining
        if batch:
            self._process_batch(batch, source, shard_idx)
            processed += len(batch)
        
        # Update stats
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.stats.processing_time_seconds += elapsed
        self.stats.source_stats[source.value] = processed
        
        logger.info(f"Completed {source.value}: {processed:,} docs in {elapsed:.1f}s")
        
        return self.stats
    
    def _process_batch(
        self,
        docs: List[Document],
        source: DataSource,
        shard_idx: int,
    ):
        """Process a batch of documents."""
        original_count = len(docs)
        
        # Stage 1: Clean
        docs = self.cleaner.clean_batch(docs)
        cleaned_count = len(docs)
        
        # Stage 2: Deduplicate
        docs = self.deduplicator.deduplicate_batch(docs)
        deduped_count = len(docs)
        
        # Update stats
        self.stats.total_documents += deduped_count
        self.stats.filtered_documents += original_count - cleaned_count
        self.stats.deduplicated_documents += cleaned_count - deduped_count
        
        # Stage 3: Store
        if docs:
            shard_id = f"{source.value}_{shard_idx:06d}"
            self.storage.write(docs, shard_id)
    
    async def process_all_sources(
        self,
        source_paths: Dict[DataSource, str],
    ) -> DataStats:
        """Process all data sources."""
        for source, path in source_paths.items():
            await asyncio.to_thread(self.process_source, source, path)
        
        return self.stats
    
    def get_stats(self) -> DataStats:
        """Get current pipeline statistics."""
        dedup_stats = self.deduplicator.get_stats()
        self.stats.deduplicated_documents = (
            dedup_stats["exact_duplicates"] + dedup_stats["near_duplicates"]
        )
        return self.stats
    
    def create_training_shards(
        self,
        output_path: str,
        tokens_per_shard: int = 10_000_000,  # 10M tokens per shard
    ) -> List[str]:
        """
        Create training shards from processed data.
        
        Combines and shuffles documents into fixed-size training shards.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all shards
        all_shards = self.storage.list_shards()
        
        # Read and combine
        training_shards = []
        current_docs: List[Document] = []
        current_tokens = 0
        shard_idx = 0
        
        for shard_id in all_shards:
            for doc in self.storage.read(shard_id):
                current_docs.append(doc)
                current_tokens += doc.token_count or len(doc.text.split())
                
                if current_tokens >= tokens_per_shard:
                    # Write training shard
                    training_shard_id = f"train_{shard_idx:06d}"
                    
                    # Shuffle before writing
                    np.random.shuffle(current_docs)
                    
                    train_storage = ParquetStorage(output_path, "zstd")
                    train_storage.write(current_docs, training_shard_id)
                    
                    training_shards.append(training_shard_id)
                    shard_idx += 1
                    current_docs = []
                    current_tokens = 0
        
        # Write remaining
        if current_docs:
            training_shard_id = f"train_{shard_idx:06d}"
            np.random.shuffle(current_docs)
            train_storage = ParquetStorage(output_path, "zstd")
            train_storage.write(current_docs, training_shard_id)
            training_shards.append(training_shard_id)
        
        logger.info(f"Created {len(training_shards)} training shards")
        
        return training_shards


# =============================================================================
# Streaming Data Loader
# =============================================================================

class StreamingDataLoader:
    """
    Streaming data loader for efficient training.
    
    Features:
    - Memory-efficient streaming
    - Prefetching
    - Dynamic batching
    - Multi-worker loading
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        max_length: int = 4096,
        num_workers: int = 4,
        prefetch_factor: int = 2,
    ):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        # Initialize storage
        self.storage = ParquetStorage(str(data_path))
        self.shards = self.storage.list_shards()
        
        # Shuffle shards
        np.random.shuffle(self.shards)
        
        self._current_shard_idx = 0
        self._current_docs: List[Document] = []
        self._doc_idx = 0
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches."""
        while True:
            batch = self._get_batch()
            if batch is None:
                break
            yield batch
    
    def _get_batch(self) -> Optional[Dict[str, List[str]]]:
        """Get next batch of documents."""
        batch_texts = []
        
        while len(batch_texts) < self.batch_size:
            # Load more docs if needed
            if self._doc_idx >= len(self._current_docs):
                if not self._load_next_shard():
                    break
            
            doc = self._current_docs[self._doc_idx]
            self._doc_idx += 1
            
            # Truncate if needed
            text = doc.text[:self.max_length * 4]  # Rough char estimate
            batch_texts.append(text)
        
        if not batch_texts:
            return None
        
        return {"texts": batch_texts}
    
    def _load_next_shard(self) -> bool:
        """Load the next shard."""
        if self._current_shard_idx >= len(self.shards):
            return False
        
        shard_id = self.shards[self._current_shard_idx]
        self._current_docs = list(self.storage.read(shard_id))
        self._doc_idx = 0
        self._current_shard_idx += 1
        
        # Shuffle within shard
        np.random.shuffle(self._current_docs)
        
        return True
    
    def reset(self):
        """Reset the data loader."""
        np.random.shuffle(self.shards)
        self._current_shard_idx = 0
        self._current_docs = []
        self._doc_idx = 0
