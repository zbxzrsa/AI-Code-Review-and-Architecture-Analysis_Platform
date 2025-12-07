"""
Format Normalization for Collected Data

Standardizes content format across different sources.
"""

import html
import logging
import re
import unicodedata
from typing import Optional

from ..collectors.base import CollectedItem, ContentType

logger = logging.getLogger(__name__)


class FormatNormalizer:
    """
    Normalizes content format for consistency.
    
    Operations:
    - HTML entity decoding
    - Unicode normalization
    - Whitespace standardization
    - Markdown cleanup
    - Code block standardization
    """
    
    def normalize(self, item: CollectedItem) -> CollectedItem:
        """
        Normalize a collected item.
        
        Args:
            item: Item to normalize
            
        Returns:
            New CollectedItem with normalized content
        """
        # Normalize content
        normalized_content = self._normalize_text(item.content)
        
        # Normalize title
        normalized_title = self._normalize_title(item.title)
        
        # Normalize tags
        normalized_tags = self._normalize_tags(item.tags)
        
        # Create new item with normalized fields
        return CollectedItem(
            source=item.source,
            source_id=item.source_id,
            url=item.url,
            title=normalized_title,
            content=normalized_content,
            content_type=item.content_type,
            language=item.language,
            tags=normalized_tags,
            author=item.author,
            created_at=item.created_at,
            updated_at=item.updated_at,
            collected_at=item.collected_at,
            metadata=item.metadata,
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize content text."""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Unicode normalization (NFC form)
        text = unicodedata.normalize("NFC", text)
        
        # Standardize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Standardize code blocks
        text = self._standardize_code_blocks(text)
        
        # Remove excessive blank lines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Standardize whitespace in non-code sections
        text = self._standardize_whitespace(text)
        
        # Remove control characters (except newlines/tabs)
        text = "".join(
            char for char in text
            if not unicodedata.category(char).startswith("C")
            or char in "\n\t"
        )
        
        return text.strip()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title text."""
        if not title:
            return ""
        
        # Decode HTML entities
        title = html.unescape(title)
        
        # Unicode normalization
        title = unicodedata.normalize("NFC", title)
        
        # Collapse whitespace
        title = " ".join(title.split())
        
        # Remove leading/trailing punctuation (except useful ones)
        title = title.strip()
        
        return title
    
    def _normalize_tags(self, tags: list) -> list:
        """Normalize tags list."""
        normalized = []
        seen = set()
        
        for tag in tags:
            # Lowercase and strip
            tag = str(tag).lower().strip()
            
            # Skip empty tags
            if not tag:
                continue
            
            # Replace special characters
            tag = re.sub(r"[^\w\-]", "-", tag)
            tag = re.sub(r"-+", "-", tag).strip("-")
            
            # Skip duplicates
            if tag in seen:
                continue
            
            seen.add(tag)
            normalized.append(tag)
        
        return normalized
    
    def _standardize_code_blocks(self, text: str) -> str:
        """Standardize code block formatting."""
        # Convert various code block formats to standard markdown
        
        # HTML <code> tags
        text = re.sub(
            r"<code>(.*?)</code>",
            r"`\1`",
            text,
            flags=re.DOTALL,
        )
        
        # HTML <pre> tags
        text = re.sub(
            r"<pre>(.*?)</pre>",
            r"```\n\1\n```",
            text,
            flags=re.DOTALL,
        )
        
        # Standardize fence style (prefer ```)
        text = re.sub(r"~~~(\w*)\n", r"```\1\n", text)
        text = re.sub(r"\n~~~", r"\n```", text)
        
        # Ensure code blocks have language hints where possible
        text = self._add_language_hints(text)
        
        return text
    
    def _add_language_hints(self, text: str) -> str:
        """Add language hints to code blocks without them."""
        
        def detect_language(code: str) -> Optional[str]:
            """Simple language detection from code content."""
            code_lower = code.lower()
            
            # Python indicators
            if re.search(r"^(import |from |def |class )", code, re.MULTILINE):
                return "python"
            
            # JavaScript indicators
            if re.search(r"(const |let |var |function |=>|require\(|import .+ from)", code):
                return "javascript"
            
            # TypeScript indicators
            if re.search(r"(interface |type |: string|: number|<[A-Z]\w+>)", code):
                return "typescript"
            
            # Go indicators
            if re.search(r"^(package |func |import \()", code, re.MULTILINE):
                return "go"
            
            # Rust indicators
            if re.search(r"^(fn |use |impl |pub fn|let mut)", code, re.MULTILINE):
                return "rust"
            
            # Bash indicators
            if re.search(r"^(#!/bin/|export |echo |\$ )", code, re.MULTILINE):
                return "bash"
            
            # SQL indicators
            if re.search(r"^(SELECT |INSERT |UPDATE |CREATE |DROP )", code, re.MULTILINE | re.IGNORECASE):
                return "sql"
            
            # YAML indicators
            if re.search(r"^\w+:\s*$", code, re.MULTILINE) and ":" in code:
                return "yaml"
            
            # JSON indicators
            if code.strip().startswith("{") and code.strip().endswith("}"):
                return "json"
            
            return None
        
        def add_hint(match):
            _fence = match.group(1)  # Available for fence style customization
            lang = match.group(2)
            code = match.group(3)
            
            if not lang:
                detected = detect_language(code)
                if detected:
                    return f"```{detected}\n{code}```"
            
            return match.group(0)
        
        # Match code blocks and add hints
        text = re.sub(
            r"```(\w*)\n(.*?)```",
            add_hint,
            text,
            flags=re.DOTALL,
        )
        
        return text
    
    def _standardize_whitespace(self, text: str) -> str:
        """Standardize whitespace outside code blocks."""
        # Split by code blocks to preserve code formatting
        parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
        
        normalized_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block - preserve
                normalized_parts.append(part)
            else:  # Regular text - normalize
                # Collapse multiple spaces to single
                part = re.sub(r"[ \t]+", " ", part)
                # Fix spacing around punctuation
                part = re.sub(r"\s+([,.])", r"\1", part)
                normalized_parts.append(part)
        
        return "".join(normalized_parts)
