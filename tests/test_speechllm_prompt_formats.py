"""Tests formats de prompt speechLLM (sans Hugging Face)."""

from __future__ import annotations

import pytest
from speechLLM.speechllm_common import (
    build_prompt_text_parts,
    resolve_prompt_format,
)


def test_build_prompt_text_parts_phi2() -> None:
    parts = build_prompt_text_parts("phi2", "Translate the French speech to English.")
    assert parts.prefix == "USER: "
    assert parts.suffix == "Translate the French speech to English. ASSISTANT: "
    assert parts.assistant_marker == "ASSISTANT:"


def test_build_prompt_text_parts_qwen_chatml() -> None:
    parts = build_prompt_text_parts("qwen_chatml", "Translate.")
    assert parts.prefix == "<|im_start|>user\n"
    assert parts.suffix == "Translate.\n<|im_start|>assistant\n"
    assert parts.assistant_marker == "assistant"


def test_build_prompt_text_parts_mistral_inst() -> None:
    parts = build_prompt_text_parts("mistral_inst", "Translate.")
    assert parts.prefix == "[INST] "
    assert parts.suffix == "Translate. [/INST] "
    assert parts.assistant_marker == "[/INST]"


def test_resolve_prompt_format_explicit() -> None:
    config = {
        "prompt": {"format": "mistral_inst"},
        "model": {"llm_name": "microsoft/phi-2"},
    }
    assert resolve_prompt_format(config) == "mistral_inst"


def test_resolve_prompt_format_infers_from_llm_name() -> None:
    assert (
        resolve_prompt_format({"model": {"llm_name": "Qwen/Qwen2.5-3B-Instruct"}})
        == "qwen_chatml"
    )
    assert (
        resolve_prompt_format(
            {"model": {"llm_name": "mistralai/Mistral-7B-Instruct-v0.3"}}
        )
        == "mistral_inst"
    )
    assert resolve_prompt_format({"model": {"llm_name": "microsoft/phi-2"}}) == "phi2"


def test_resolve_prompt_format_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt.format"):
        resolve_prompt_format({"prompt": {"format": "unknown"}, "model": {}})
