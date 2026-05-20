"""Text normalization and ASR error metrics (WER/CER)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text_for_eval(
    text: str,
    *,
    mode: str = "nfkc",
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize reference and hypothesis strings for comparable WER/CER."""
    value = text.strip()
    if mode == "nfkc":
        value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())
    if lowercase:
        value = value.lower()
    if remove_punctuation:
        value = _PUNCT_RE.sub(" ", value)
        value = " ".join(value.split())
    return value


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class ErrorRate:
    """Word or character error rate with edit counts."""

    errors: int
    ref_length: int
    rate: float


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    errors = _edit_distance(ref_tokens, hyp_tokens)
    ref_len = len(ref_tokens)
    rate = errors / ref_len if ref_len else (0.0 if not errors else 1.0)
    return ErrorRate(errors=errors, ref_length=ref_len, rate=rate)


def char_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    errors = _edit_distance(ref_chars, hyp_chars)
    ref_len = len(ref_chars)
    rate = errors / ref_len if ref_len else (0.0 if not errors else 1.0)
    return ErrorRate(errors=errors, ref_length=ref_len, rate=rate)
