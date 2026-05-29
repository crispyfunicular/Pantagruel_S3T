"""Tests for ASR text metrics."""

from scripts_communs.utils_text_eval import (
    char_error_rate,
    normalize_text_for_eval,
    word_error_rate,
)


def test_normalize_text_nfkc_lowercase():
    text = normalize_text_for_eval("  Café!  ", mode="nfkc", lowercase=True)
    assert text == "café"


def test_word_error_rate_identical():
    ref = "au nord du pays"
    hyp = "au nord du pays"
    wer = word_error_rate(ref, hyp)
    assert wer.rate == 0.0
    assert wer.errors == 0


def test_word_error_rate_substitution():
    wer = word_error_rate("un deux trois", "un quatre trois")
    assert wer.errors == 1
    assert wer.ref_length == 3


def test_char_error_rate():
    cer = char_error_rate("abc", "adc")
    assert cer.errors == 1
    assert cer.ref_length == 3
