#!/usr/bin/env python3
"""
Smoke test — charge les encodeurs Pantagruel 1k / 14k / 114k (HF) et un forward minimal.

Vérifie que les checkpoints Large sont accessibles sur la machine (tour GPU ou local)
avant de lancer des entraînements longs. Ne nécessite pas les manifests m-TEDx.

Usage :
    source .venv/bin/activate
    python scripts/smoke_pantagruel_encoders.py
    python scripts/smoke_pantagruel_encoders.py --encoders 14k,114k
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import torch
from transformers import AutoModel

# Aligné Table 8 Pantagruel (papier L-14k / L-114k) et réplication B-1k S3T.
ENCODER_CATALOG: dict[str, tuple[str, int]] = {
    "1k": ("PantagrueLLM/speech-base-1K", 768),
    "14k": ("PantagrueLLM/speech-large-14K", 1024),
    "114k": ("PantagrueLLM/speech-large-114K", 1024),
}


def smoke_encoder(key: str, hf_id: str, expected_hidden: int) -> int:
    """
    Télécharger (si besoin) l'encodeur et exécuter un forward sur bruit synthétique.

    Returns:
        0 si OK, 1 si échec.
    """
    print(f"=== [{key}] {hf_id} (hidden_size attendu {expected_hidden}) ===")
    try:
        model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        model.eval()
        hidden = int(model.config.hidden_size)
        if hidden != expected_hidden:
            print(
                f"WARN: hidden_size={hidden} != {expected_hidden} (vérifier la config décodeur ST)",
                file=sys.stderr,
            )
        # ~0,5 s @ 16 kHz — même convention que st_common (input_values + mask).
        wav = torch.randn(1, 8000)
        mask = torch.ones(1, wav.size(1), dtype=torch.long)
        with torch.no_grad():
            out = model(input_values=wav, attention_mask=mask)
        last = out.last_hidden_state
        print(f"OK: last_hidden_state {tuple(last.shape)}")
        return 0
    except Exception as exc:
        print(f"FAIL [{key}]: {exc}", file=sys.stderr)
        return 1


def parse_encoder_keys(raw: str) -> list[str]:
    """Parser la liste d'identifiants courts (1k, 14k, 114k)."""
    keys = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [k for k in keys if k not in ENCODER_CATALOG]
    if unknown:
        raise ValueError(
            f"encodeurs inconnus: {unknown} ; choix: {sorted(ENCODER_CATALOG)}"
        )
    return keys


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoders",
        default="1k,14k,114k",
        help="Liste séparée par des virgules : 1k, 14k, 114k (défaut: tous)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    keys = parse_encoder_keys(args.encoders)
    failures = 0
    for key in keys:
        hf_id, expected = ENCODER_CATALOG[key]
        failures += smoke_encoder(key, hf_id, expected)

    if failures:
        print(f"\nÉchec: {failures} encodeur(s) sur {len(keys)}.", file=sys.stderr)
        return 1
    print(f"\nTous les encodeurs demandés ({len(keys)}) sont chargeables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
