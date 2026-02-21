from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_glm_tokenizer = None
_GLM_MODEL_ID = "zai-org/GLM-5"


def _get_glm_tokenizer():
    """Return the cached GLM tokenizer, loading on first call."""
    global _glm_tokenizer
    if _glm_tokenizer is None:
        try:
            from transformers import AutoTokenizer

            _glm_tokenizer = AutoTokenizer.from_pretrained(
                _GLM_MODEL_ID,
                trust_remote_code=True,
            )
            logger.info("GLM tokenizer loaded from %s", _GLM_MODEL_ID)
        except Exception as exc:
            logger.warning("Failed to load GLM tokenizer: %s", exc)
    return _glm_tokenizer


def glm_token_count(text: str) -> int:
    """Return token count using the GLM-5 tokenizer.

    Falls back to ``len(text) // 4`` if tokenizer loading is unavailable.
    """
    tok = _get_glm_tokenizer()
    if tok is None:
        return len(text) // 4
    return len(tok.encode(text))

