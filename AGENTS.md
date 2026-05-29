# AGENTS.md — Instructions pour agents et contributeurs

Ce fichier définit les conventions du dépôt **S3T**. Tout agent ou contributeur doit les suivre.

---

## Langues

| Zone | Langue |
|------|--------|
| Communication avec l'utilisateur | **Français** |
| Code Python (noms de symboles) | **Anglais** |
| Code Python (docstrings, commentaires) | **Français** |
| Documentation projet (`README.md`, `docs/PRD.md`, ce fichier) | **Français** |
| Messages de commit | **Français** (phrases complètes, orientées « pourquoi ») |

---

## Architecture pipeline

| Zone | Dossier | Rôle |
|------|---------|------|
| Commun | `scripts_communs/` | Étapes **0–2** : `0_preflight.py`, `1_download.py`, `2_prepare.py`, `st_common.py`, `pipeline.py`, `bootstrap.sh` |
| Variante **1** | `1_Transformer/` | Étapes **3–6** baseline ST : SPM, train, evaluate, infer, `configs/` |
| Variante **2** | `2_speechLLM/` | Projecteur B1 : train, evaluate, infer |
| Variante **3** | `3_Gemini/` | Baseline API Gemini |
| Variante **4** | `4_cascade/` | Baseline ASR→MT (squelette) |

- Les routeurs `*/pipeline.py` sont des **délégations CLI uniquement** (pas de logique métier inline).
- Les dossiers préfixés par un chiffre ne sont pas des packages Python importables ; les imports logiques (`speechLLM`, `Gemini`, `Cascade`) passent par `scripts_communs/variant_bootstrap.py`.

Voir [docs/PRD.md](docs/PRD.md) §2.3 et [README.md](README.md) §Architecture.

---

## Qualité obligatoire avant commit

Avant **chaque commit**, exécuter dans l'ordre :

```bash
source .venv/bin/activate
pip install -r requirements.txt   # si pas déjà fait (runtime + outils dev)
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

**Obligatoire** : mettre à jour [docs/PRD.md](docs/PRD.md) et [README.md](README.md) dans le même commit lorsque le changement affecte :

- le comportement d'un stage du pipeline
- les prérequis, commandes CLI, ou chemins de sortie
- le statut d'implémentation d'un module (`implémenté` / `à implémenter`)
- les critères go/no-go ou la structure du dépôt

Checklist rapide avant commit :
- [ ] `ruff check .` et `ruff format --check .` OK
- [ ] `pytest` OK
- [ ] `docs/PRD.md` à jour si le périmètre ou les exigences ont changé
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

### Commentaires et documentation du code

Le code doit être **bien commenté** (docstrings et commentaires en **français** ; noms de variables/fonctions en anglais) :

1. **En-tête de fichier** — Chaque module Python (`.py`) commence par un docstring de module qui explique :
   - le rôle du fichier dans le pipeline S3T ;
   - les entrées / sorties principales (manifests, checkpoints, métriques, etc.) ;
   - les dépendances ou conventions particulières si nécessaire.

2. **Docstrings sur les fonctions** — Toute fonction (y compris méthodes de classe) expose une docstring décrivant :
   - ce qu'elle fait ;
   - les paramètres importants (`Args`) ;
   - la valeur de retour (`Returns`) ;
   - les exceptions ou effets de bord notables, le cas échéant.

3. **Commentaires inline** — Tout bloc de logique **non trivial** (algorithmes, choix de design, contraintes PRD, interactions GPU/AMP, anti-fuite données, etc.) est accompagné de commentaires courts qui expliquent le **pourquoi**, pas seulement le quoi.

**Trivial** = affectation évidente, import standard, appel unique à une API bien nommée — pas besoin de commentaire redondant.

**Non trivial** = boucles d'entraînement, gel d'encodeur, décodage, normalisation texte, règles de manifest, gestion de chemins de run — commentaire attendu.

Les nouveaux fichiers et les modifications substantielles sur un module existant doivent respecter ces trois points avant commit.

Structure tests :

```text
tests/
  conftest.py
  test_0_preflight.py
  ...
```

---

## Ajout d'un nouveau stage

1. Créer le module dans `scripts_communs/` (commun) ou `1_Transformer/` / `N_<variante>/`
2. Brancher le subcommand dans le `pipeline.py` du dossier concerné
3. Ajouter tests dans `tests/test_N_<stage>.py`
4. Mettre à jour `docs/PRD.md` §2.3 et phases associées
5. Mettre à jour `README.md` (table architecture + section pipeline)
6. Lancer ruff + pytest avant commit

---

## Fichiers de référence

| Fichier | Rôle |
|---------|------|
| [docs/PRD.md](docs/PRD.md) | Exigences, hyperparamètres, ablations, protocole runs, template YAML |
| [README.md](README.md) | Point d'entrée, quickstart, pipeline, expérimentation |
| [requirements.txt](requirements.txt) | Dépendances runtime + dev (Ruff, pytest, pre-commit) |
| [pyproject.toml](pyproject.toml) | Configuration ruff / pytest |
