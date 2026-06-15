"""Tests SpecAugment ST (st_common)."""

from __future__ import annotations

import torch
from scripts_communs.st_common import (
    SpecAugmentConfig,
    apply_feature_freq_mask,
    apply_waveform_time_mask,
    parse_spec_augment_config,
)


def test_parse_spec_augment_config_defaults() -> None:
    cfg = parse_spec_augment_config({})
    assert cfg.enabled is False
    assert cfg.mask_time_prob == 0.05


def test_apply_waveform_time_mask_zeros_segment() -> None:
    spec = SpecAugmentConfig(enabled=True, mask_time_prob=1.0, mask_time_length=10)
    waves = torch.ones(1, 16000)
    mask = torch.ones(1, 16000, dtype=torch.long)
    torch.manual_seed(0)
    out = apply_waveform_time_mask(
        waves,
        mask,
        spec_augment=spec,
        sample_rate=16000,
    )
    assert out.sum().item() < waves.sum().item()
    assert torch.any(out == 0.0)


def test_apply_waveform_time_mask_disabled_is_noop() -> None:
    spec = SpecAugmentConfig(enabled=False)
    waves = torch.randn(2, 1000)
    mask = torch.ones(2, 1000, dtype=torch.long)
    out = apply_waveform_time_mask(
        waves,
        mask,
        spec_augment=spec,
        sample_rate=16000,
    )
    assert torch.equal(out, waves)


def test_apply_feature_freq_mask_zeros_band() -> None:
    spec = SpecAugmentConfig(
        enabled=True,
        mask_freq_prob=1.0,
        mask_freq_length=4,
    )
    features = torch.ones(1, 8, 16)
    torch.manual_seed(0)
    out = apply_feature_freq_mask(features, spec_augment=spec)
    assert out.sum().item() < features.sum().item()
    assert torch.any(out == 0.0)
