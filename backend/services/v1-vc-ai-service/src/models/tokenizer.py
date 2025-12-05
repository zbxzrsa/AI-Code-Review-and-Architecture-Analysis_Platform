"""
Custom Tokenizer for V1 VC-AI

Enhanced Byte-Pair Encoding (BPE) tokenizer with:
- Custom code/commit vocabulary
- Special tokens for version control concepts
- Efficient encoding for diffs and commit messages
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import re

import torch


@dataclass
class SpecialTokens:
    """
    Special tokens for version control tasks.
    
    These tokens help the model understand the structure
    and semantics of commits and diffs.
    """
    # Structural tokens
    COMMIT = "<COMMIT>"
    DIFF = "<DIFF>"
    FILE = "<FILE>"
    FILE_END = "</FILE>"
    HUNK = "<HUNK>"
    
    # Change type tokens
    CHANGE_TYPE = "<CHANGE_TYPE>"
    IMPACT_LEVEL = "<IMPACT_LEVEL>"
    
    # Diff content tokens
    ADD = "<ADD>"
    DEL = "<DEL>"
    CTX = "<CTX>"           # Context line
    
    # Semantic tokens
    BREAK = "<BREAK>"       # Semantic break
    FUNCTION = "<FUNC>"
    CLASS = "<CLASS>"
    IMPORT = "<IMPORT>"
    
    # Change type values
    BUG_FIX = "<BUG_FIX>"
    FEATURE = "<FEATURE>"
    REFACTOR = "<REFACTOR>"
    OPTIMIZATION = "<OPTIMIZATION>"
    DOCS = "<DOCS>"
    TEST = "<TEST>"
    CHORE = "<CHORE>"
    
    # Impact level values
    CRITICAL = "<CRITICAL>"
    HIGH = "<HIGH>"
    MEDIUM = "<MEDIUM>"
    LOW = "<LOW>"
    
    # Standard tokens
    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    MASK = "<MASK>"
    
    @classmethod
    def all_tokens(cls) -> List[str]:
        """Get all special tokens as a list"""
        return [
            getattr(cls, attr) for attr in dir(cls)
            if not attr.startswith('_') and isinstance(getattr(cls, attr), str)
        ]
    
    @classmethod
    def to_dict(cls) -> Dict[str, str]:
        """Get all special tokens as a dictionary"""
        return {
            attr: getattr(cls, attr) for attr in dir(cls)
            if not attr.startswith('_') and isinstance(getattr(cls, attr), str)
        }


class CodeCommitTokenizer:
    """
    Custom BPE tokenizer optimized for code and commit messages.
    
    Features:
    - Preserves whitespace and indentation
    - Handles code-specific punctuation
    - Includes special tokens for version control
    - Efficient encoding of diffs
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[SpecialTokens] = None,
        preserve_whitespace: bool = True,
        split_on_punctuation: bool = True,
        lowercase: bool = False,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or SpecialTokens()
        self.preserve_whitespace = preserve_whitespace
        self.split_on_punctuation = split_on_punctuation
        self.lowercase = lowercase
        
        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # BPE merges
        self.merges: Dict[Tuple[str, str], str] = {}
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        
        # Initialize with special tokens
        self._init_special_tokens()
        
        # Compiled regex patterns for tokenization
        self._compile_patterns()
        
    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens"""
        for i, token in enumerate(self.special_tokens.all_tokens()):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        self.pad_token_id = self.token_to_id[SpecialTokens.PAD]
        self.bos_token_id = self.token_to_id[SpecialTokens.BOS]
        self.eos_token_id = self.token_to_id[SpecialTokens.EOS]
        self.unk_token_id = self.token_to_id[SpecialTokens.UNK]
        
    def _compile_patterns(self):
        """Compile regex patterns for pre-tokenization"""
        # Pattern for code tokens
        self.code_pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.VERBOSE
        )
        
        # Pattern for diff lines
        self.diff_line_pattern = re.compile(r'^([+\-\s])', re.MULTILINE)
        
        # Pattern for file paths
        self.file_path_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_/.-]*\.[a-zA-Z]+')
        
        # Pattern for function/method names
        self.function_pattern = re.compile(r'\b(def|function|func|fn)\s+(\w+)')
        
        # Pattern for class names
        self.class_pattern = re.compile(r'\b(class|struct|interface)\s+(\w+)')
        
    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into initial tokens"""
        if self.lowercase:
            text = text.lower()
        
        # Handle special patterns first
        tokens = []
        
        # Split on whitespace while preserving it
        if self.preserve_whitespace:
            parts = re.split(r'(\s+)', text)
            for part in parts:
                if part.strip():
                    tokens.extend(self._split_code_token(part))
                elif part:
                    tokens.append(part)
        else:
            tokens = text.split()
        
        return tokens
    
    def _split_code_token(self, token: str) -> List[str]:
        """Split a code token on punctuation if enabled"""
        if not self.split_on_punctuation:
            return [token]
        
        # Split on common code punctuation while keeping operators together
        parts = re.split(r'([{}()\[\];,.:=<>!&|+\-*/\\@#$%^~`])', token)
        return [p for p in parts if p]
    
    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _bpe(self, token: str) -> List[str]:
        """Apply BPE encoding to a single token"""
        if token in self.token_to_id:
            return [token]
        
        word = list(token)
        
        while len(word) > 1:
            pairs = self._get_pairs(word)
            
            # Find the highest priority merge
            bigram = min(
                pairs,
                key=lambda pair: self.merge_ranks.get(pair, float('inf'))
            )
            
            if bigram not in self.merge_ranks:
                break
            
            # Apply the merge
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        
        return word
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            truncation: Truncate if exceeds max_length
            padding: Pad to max_length
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        # Pre-tokenize
        pre_tokens = self._pre_tokenize(text)
        
        # Apply BPE
        tokens = []
        for pre_token in pre_tokens:
            tokens.extend(self._bpe(pre_token))
        
        # Convert to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        # Handle truncation
        if max_length and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length - 1] + [self.eos_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Handle padding
        if max_length and padding:
            pad_length = max_length - len(token_ids)
            if pad_length > 0:
                token_ids.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
        
        return {
            "input_ids": torch.tensor([token_ids]),
            "attention_mask": torch.tensor([attention_mask]),
        }
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens.all_tokens():
                    continue
                tokens.append(token)
        
        return "".join(tokens)
    
    def encode_diff(
        self,
        diff: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a git diff with special structure preservation.
        
        Adds special tokens for additions, deletions, and context.
        """
        lines = diff.split('\n')
        processed_lines = []
        
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                processed_lines.append(f"{SpecialTokens.ADD} {line[1:]}")
            elif line.startswith('-') and not line.startswith('---'):
                processed_lines.append(f"{SpecialTokens.DEL} {line[1:]}")
            elif line.startswith('@@'):
                processed_lines.append(f"{SpecialTokens.HUNK} {line}")
            elif line.startswith('diff --git'):
                # Extract file path
                match = re.search(r'a/(.+?) b/', line)
                if match:
                    processed_lines.append(f"{SpecialTokens.FILE} {match.group(1)}")
            else:
                processed_lines.append(f"{SpecialTokens.CTX} {line}")
        
        processed_diff = '\n'.join(processed_lines)
        return self.encode(processed_diff, max_length=max_length)
    
    def encode_commit(
        self,
        message: str,
        diff: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a complete commit (message + diff).
        """
        # Combine message and diff with special tokens
        combined = f"{SpecialTokens.COMMIT} {message} {SpecialTokens.DIFF} {diff}"
        return self.encode(combined, max_length=max_length)
    
    def train(
        self,
        texts: List[str],
        vocab_size: Optional[int] = None,
    ):
        """
        Train BPE tokenizer on a corpus.
        
        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
        """
        vocab_size = vocab_size or self.vocab_size
        
        # Count initial character frequencies
        char_freq: Dict[str, int] = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Initialize vocabulary with characters
        current_vocab_size = len(self.token_to_id)
        for char, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
            if freq >= self.min_frequency and char not in self.token_to_id:
                self.token_to_id[char] = current_vocab_size
                self.id_to_token[current_vocab_size] = char
                current_vocab_size += 1
        
        # Learn BPE merges
        word_freqs: Dict[str, int] = {}
        for text in texts:
            for word in self._pre_tokenize(text):
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # Convert words to character lists
        splits = {word: list(word) for word in word_freqs}
        
        merge_rank = 0
        while current_vocab_size < vocab_size:
            # Count pair frequencies
            pair_freqs: Dict[Tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                split = splits[word]
                if len(split) < 2:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            self.token_to_id[new_token] = current_vocab_size
            self.id_to_token[current_vocab_size] = new_token
            self.merges[best_pair] = new_token
            self.merge_ranks[best_pair] = merge_rank
            
            # Update splits
            for word in splits:
                split = splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split
            
            current_vocab_size += 1
            merge_rank += 1
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        with open(path / "vocab.json", "w") as f:
            json.dump(self.token_to_id, f, indent=2)
        
        # Save merges
        merges_list = [(k[0], k[1], v) for k, v in self.merges.items()]
        with open(path / "merges.json", "w") as f:
            json.dump(merges_list, f, indent=2)
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "preserve_whitespace": self.preserve_whitespace,
            "split_on_punctuation": self.split_on_punctuation,
            "lowercase": self.lowercase,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "CodeCommitTokenizer":
        """Load tokenizer from disk"""
        path = Path(path)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        
        tokenizer = cls(**config)
        
        # Load vocabulary
        with open(path / "vocab.json", "r") as f:
            tokenizer.token_to_id = json.load(f)
            tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        
        # Load merges
        with open(path / "merges.json", "r") as f:
            merges_list = json.load(f)
            for i, (first, second, merged) in enumerate(merges_list):
                tokenizer.merges[(first, second)] = merged
                tokenizer.merge_ranks[(first, second)] = i
        
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.token_to_id)
