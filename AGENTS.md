# AGENTS.md — Instructions pour agents et contributeurs

Ce fichier définit les conventions du dépôt **S3T**. Tout agent ou contributeur doit les suivre.

---

## Langues

| Zone | Langue |
|------|--------|
| Communication avec l'utilisateur | **Français** |
| Code Python (noms, docstrings courtes, commentaires) | **Anglais** |
| Documentation projet (`README.md`, `PRD.md`, `README_experiments.md`, ce fichier) | **Français** |
| Messages de commit | **Français** (phrases complètes, orientées « pourquoi ») |

---

## Architecture pipeline

- Un module par stage : `scripts/0_preflight.py` … `scripts/6_infer.py`
- `scripts/pipeline.py` = **routeur CLI uniquement** (pas de logique métier inline)
- `scripts/bootstrap.sh` = installation environnement Phase 1

Voir [PRD.md](PRD.md) §2.3 et [README.md](README.md) §Architecture.

---

## Qualité obligatoire avant commit

Avant **chaque commit**, exécuter dans l'ordre :

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt   # si pas déjà fait
ruff check .
ruff format --check .
pytest
```

Ou via hooks automatiques :

```bash
pre-commit install
pre-commit run --all-files
```

**Règle :** ne pas committer si `ruff` ou `pytest` échoue.

Outils retenus :
- **Ruff** : lint + format (remplace black + flake8/isort pour ce projet)
- **pytest** : tests unitaires et d'intégration légers

---

## Mise à jour documentation avant commit

**Obligatoire** : mettre à jour [PRD.md](PRD.md) et [README.md](README.md) dans le même commit lorsque le changement affecte :

- le comportement d'un stage du pipeline
- les prérequis, commandes CLI, ou chemins de sortie
- le statut d'implémentation d'un module (`implémenté` / `à implémenter`)
- les critères go/no-go ou la structure du dépôt

Checklist rapide avant commit :
- [ ] `ruff check .` et `ruff format --check .` OK
- [ ] `pytest` OK
- [ ] `PRD.md` à jour si le périmètre ou les exigences ont changé
- [ ] `README.md` à jour si l'usage ou l'architecture a changé
- [ ] `AGENTS.md` à jour si les règles de contribution changent

---

## Conventions de code Python

- Python 3.10+
- Typage progressif (`from __future__ import annotations`)
- Fonctions courtes, noms explicites en anglais
- Chaque stage expose `main()` et/ou `run_from_namespace(args: argparse.Namespace) -> int`
- Codes de sortie documentés (0 = succès, non-zero = erreur)
- Pas de secrets dans le dépôt (`.env` ignoré)

Structure tests :

```text
tests/
  conftest.py
  test_0_preflight.py
  ...
```

---

## Ajout d'un nouveau stage

1. Créer `scripts/N_<stage>.py` avec CLI autonome
2. Brancher le subcommand dans `scripts/pipeline.py` (délégation, pas de logique dupliquée)
3. Ajouter tests dans `tests/test_N_<stage>.py`
4. Mettre à jour `PRD.md` §2.3 et phases associées
5. Mettre à jour `README.md` (table architecture + section pipeline)
6. Lancer ruff + pytest avant commit

---

## Fichiers de référence

| Fichier | Rôle |
|---------|------|
| [PRD.md](PRD.md) | Exigences, hyperparamètres, plan de projet |
| [README.md](README.md) | Point d'entrée, quickstart, pipeline |
| [README_experiments.md](README_experiments.md) | Runbook expérimental détaillé |
| [requirements.txt](requirements.txt) | Dépendances runtime Phase 1 |
| [requirements-dev.txt](requirements-dev.txt) | Ruff, pytest, pre-commit |
| [pyproject.toml](pyproject.toml) | Configuration ruff / pytest |
