"""
Utilitaires légers pour configs YAML et artefacts JSON (sans PyTorch).

Utilisé par les scripts d'évaluation, de tracking CSV et d'export relecture
qui doivent tourner avec seulement PyYAML (pas de dépendance torch/numpy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def deep_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Lire une valeur de config imbriquée en notation pointée (ex. ``train.batch_size``).

    Paramètres :
        config : Dict imbriqué chargé depuis YAML.
        key : Chemin séparé par des points.
        default : Valeur renvoyée si un segment du chemin est absent.

    Retour :
        Valeur à ``key`` ou ``default``.
    """
    cursor: Any = config
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def load_yaml_config(path: Path) -> dict[str, Any]:
    """
    Charger une config d'expérience YAML dans un dict simple.

    Paramètres :
        path : Chemin vers ``base.yaml`` ou équivalent.

    Retour :
        Mapping YAML parsé.

    Lève :
        ValueError: Si la racine YAML n'est pas un objet.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Écrire un artefact JSON en créant les répertoires parents si nécessaire.

    Paramètres :
        path : Chemin du fichier de sortie.
        payload : Dict sérialisable (métriques, rapports, etc.).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
