"""Cross-provider token counting for AI-coding cost estimation"""

import tiktoken

_ENCODING_NAME = "cl100k_base"
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def count_tokens(text: str) -> int:
    """Approximate token count for text, via a generic tiktoken encoding."""
    if not text:
        return 0
    return len(_get_encoding().encode(text))
