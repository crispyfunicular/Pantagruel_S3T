# S3T — Speech Translation (Pantagruel replication)

Réplication du système de **traduction de la parole end-to-end** décrit dans *Pantagruel* (encodeur SSL + décodeur Transformer), évalué sur **m-TEDx** (`fr-en`, `fr-pt`, `fr-es`) avec **SacreBLEU**.

| Document | Rôle |
|----------|------|
| [PRD.md](PRD.md) | Vision, exigences, hyperparamètres, risques |
| [AGENTS.md](AGENTS.md) | Conventions agents, qualité, workflow avant commit |
| [README_experiments.md](README_experiments.md) | Runbook détaillé, ablations, tracking |
| [requirements.txt](requirements.txt) | Dépendances Phase 1 |
| [requirements-dev.txt](requirements-dev.txt) | Ruff, pytest, pre-commit |

---

## Prérequis

- Python 3.10+
- GPU CUDA recommandé
- Accès réseau (OpenSLR, Hugging Face)
- Espace disque ≥ 200 GB (corpus + runs)

---

## Développement et qualité

**Langues :** code en anglais, documentation projet en français (voir [AGENTS.md](AGENTS.md)).

Installation des outils dev :

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

Avant **chaque commit** (obligatoire) :

```bash
ruff check .
ruff format --check .
pytest
# ou : pre-commit run --all-files
```

Mettre à jour [PRD.md](PRD.md) et [README.md](README.md) dans le même commit si le comportement CLI, l'architecture ou les prérequis changent.

---

## Architecture du projet

Convention : **un fichier Python par stage**, plus un orchestrateur CLI.

| Étape | Module | Commande `pipeline.py` | Statut |
|-------|--------|------------------------|--------|
| Bootstrap | [`scripts/bootstrap.sh`](scripts/bootstrap.sh) | — | implémenté |
| 0 — Preflight | [`scripts/0_preflight.py`](scripts/0_preflight.py) | `preflight` | implémenté |
| 1 — Download | [`scripts/1_download.py`](scripts/1_download.py) | `download` | implémenté |
| 2 — Prepare | [`scripts/2_prepare.py`](scripts/2_prepare.py) | `prepare` | implémenté |
| 3 — SPM | `scripts/3_spm.py` | `spm` | implémenté |
| 4 — Train | `scripts/4_train.py` | `train` | implémenté |
| 5 — Evaluate | `scripts/5_evaluate.py` | `evaluate` | implémenté |
| 6 — Infer | `scripts/6_infer.py` | `infer` | implémenté |
| Orchestrateur | [`scripts/pipeline.py`](scripts/pipeline.py) | `run` (+ toutes les étapes) | routeur actif |

Chaque module stage est exécutable **directement** (`python scripts/N_*.py ...`) ou via `pipeline.py <subcommand>`.  
`pipeline.py` ne contient pas la logique métier : il route vers le module correspondant.

**Stack d'entraînement :** SpeechBrain (pas fairseq). Le dépôt ne suit pas le schéma minimal `recipes/<task>/train.py` + `hparams/*.yaml` : stages numérotés, contrat d'artifacts par run, SacreBLEU externe obligatoire. Voir [PRD.md §2.5](PRD.md#25-source-de-vérité-historique-et-transposition-speechbrain) pour le tableau des divergences et l'alignement avec l'historique fairseq (`pantagruel_uni`, lecture seule).

---

## Quickstart

```bash
# 0) Bootstrap — venv + dépendances Phase 1
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
source .venv/bin/activate

# Optionnel : PyTorch avec CUDA
# ./scripts/bootstrap.sh --with-cuda-index-url https://download.pytorch.org/whl/cu124 --lock

# 1) Vérifier l'environnement
python scripts/pipeline.py preflight

# 2) Pipeline complet (fr-en exemple)
python scripts/pipeline.py run --langpair fr-es --run-id run_001 \
  --config configs/fr-es/base.yaml \
  --from-stage preflight --to-stage evaluate
```

---

## Pipeline

```mermaid
flowchart TD
  bootstrap["bootstrap.sh"] --> preflight["0_preflight.py"]
  preflight --> download["1_download.py"]
  download --> prepare["2_prepare.py"]
  prepare --> spm["3_spm.py"]
  spm --> train["4_train.py"]
  train --> evaluate["5_evaluate.py"]
  evaluate --> infer["6_infer.py"]
  orchestrator["pipeline.py run"] -.-> preflight
  orchestrator -.-> download
  orchestrator -.-> prepare
  orchestrator -.-> spm
  orchestrator -.-> train
  orchestrator -.-> evaluate
  orchestrator -.-> infer
```

### 0) Bootstrap (`scripts/bootstrap.sh`)
- **But**: préparer un environnement Python reproductible pour la phase 1 du PRD.
- **Entrées**: `requirements.txt`, optionnel `--with-cuda-index-url`.
- **Actions**: crée le venv, met à jour `pip`, installe les dépendances, vérifie `torch`/CUDA.
- **Sorties**: `.venv/`, optionnel `requirements.lock.txt` avec `--lock`.
- **Validation**: le script termine sans erreur et affiche l’état CUDA (`cuda available: True/False`).

### 1) Preflight (`scripts/0_preflight.py`)
- **But**: vérifier qu’une machine distante Linux + CUDA est prête avant download/train.
- **Module**: [`scripts/0_preflight.py`](scripts/0_preflight.py) (appelable directement ou via `pipeline.py preflight`).
- **Politique**: `strict_critical` — seuls les checks critiques font échouer le script (exit `1`). Les warnings n’empêchent pas de continuer.
- **Checks critiques (fail)**:
  - Python >= 3.10
  - `torch` importable
  - CUDA disponible si `--check-gpu` (défaut: activé)
  - espace disque libre >= `--min-disk-gb` (défaut: 200 GB)
- **Checks non critiques (warn)**:
  - VRAM GPU >= `--min-vram-gb` (défaut: 8 GB)
  - `nvidia-smi` présent
  - connectivité OpenSLR + Hugging Face (`--check-network`)
  - dossiers `datasets/` et `scripts/` présents
- **Sorties**: `artifacts/preflight_report.json` (résumé + détail de chaque check).
- **Validation**: `summary.passed == true` dans le rapport JSON.

Exemple sur machine distante (après `bootstrap.sh` + activation du venv) :

```bash
python scripts/0_preflight.py --check-gpu --min-disk-gb 200 --min-vram-gb 8
# ou via l’orchestrateur :
python scripts/pipeline.py preflight --check-gpu --min-disk-gb 200 --min-vram-gb 8
```

Lecture rapide du rapport :

```bash
python -c "import json; r=json.load(open('artifacts/preflight_report.json')); print(r['summary'])"
```

### 2) Download (`scripts/1_download.py`)
- **But**: récupérer les corpus m-TEDx depuis OpenSLR-100.
- **Module**: [`scripts/1_download.py`](scripts/1_download.py) (direct ou `pipeline.py download`).
- **Défaut**: `--langpairs fr-en` (une seule paire si non précisé).
- **Paires supportées**: `fr-en`, `fr-pt`, `fr-es`.
- **Entrées**: `--langpairs`, `--output-root`, `--resume` / `--no-resume`, `--extract` / `--no-extract`.
- **Actions**: téléchargement HTTP (reprise simple si possible) + extraction `.tgz` optionnelle.
- **Sorties**: `datasets/raw/mtedx_<pair>.tgz`, dossiers extraits, `artifacts/download_manifest.json`.
- **Validation**: manifest sans erreur (`exit_code == 0`), archives présentes pour chaque paire demandée.

Exemple :

```bash
python scripts/1_download.py
# équivalent :
python scripts/pipeline.py download

# plusieurs paires
python scripts/1_download.py --langpairs fr-en,fr-pt,fr-es
```

### 3) Prepare (`scripts/2_prepare.py` / `pipeline.py prepare`)
- **Module**: [`scripts/2_prepare.py`](scripts/2_prepare.py) (direct ou `pipeline.py prepare`).
- **But**: transformer les données brutes en données entraînables conformes PRD.
- **Entrées**: `datasets/raw/mtedx_<pair>/` (après download), `--sample-rate`, durées min/max, `--text-norm`, `--lowercase`.
- **Actions**:
  - découpage des FLAC m-TEDx en WAV mono 16 kHz PCM16,
  - filtrage segments invalides (texte vide, durées hors borne, audio manquant),
  - manifests TSV (`id`, `audio`, `n_frames`, `tgt_text`, …) + `*.target.txt` par split,
  - contrôle anti-fuite train vs valid/test (`--fail-on-leak`, défaut activé).
- **Sorties**: `datasets/processed/<pair>/`, `datasets/manifests/<pair>/*.tsv`, `artifacts/prepare_<pair>.json`.
- **Validation**: `exit_code == 0`, rapport sans fuite, au moins un segment par split attendu.

```bash
python scripts/2_prepare.py --langpair fr-en
python scripts/pipeline.py prepare --langpair fr-en --min-duration 1.0 --max-duration 30.0
```

**Reprise (run long, safe to relaunch):** `--resume` est activé par défaut — les WAV déjà valides sont ignorés.

```bash
source .venv/bin/activate
./scripts/prepare_status.sh fr-en          # état actuel
./scripts/resume_prepare.sh fr-en          # reprise + vérif finale (log: logs/prepare_fr-en.log)
# ou manuellement :
python scripts/2_prepare.py --langpair fr-en --resume --verbose
python scripts/2_prepare.py --langpair fr-en --verify-only
```

Prérequis : download terminé (`datasets/raw/fr-en/` ou `artifacts/download_manifest.json` avec `exit_code: 0`).
Format WAV produit : **16 kHz, mono, PCM_16** ; progression dans `artifacts/prepare_<pair>.progress.json`.

### 4) SPM (`scripts/3_spm.py` / `pipeline.py spm`)
- **Module**: `scripts/3_spm.py` (implémenté).
- **But**: entraîner le tokenizer SentencePiece sur la cible textuelle.
- **Entrées**: `--langpair`, `--vocab-size`, `--model-type`, optionnel `--train-text`.
- **Actions**: entraînement SPM sur `datasets/manifests/<pair>/train.target.txt` (par défaut).
- **Sorties**: `datasets/processed/spm/<pair>_<vocab>.model` et `.vocab`, rapport `artifacts/spm_<pair>_<vocab>.json`.
- **Validation**: modèles SPM générés et chargeables sans erreur.

### 5) Train (`scripts/4_train.py` / `pipeline.py train`)
- **Module**: `scripts/4_train.py` (implémenté).
- **But**: entraîner le modèle ST (encodeur SSL + décodeur Transformer).
- **Entrées**: `--config` (hyperparamètres/run), `--run-id`, optionnel `--output-dir`.
- **Actions**:
  - lecture config run,
  - boucle d’entraînement (HF encoder + décodeur Transformer) avec logs/checkpoints,
  - sélection du meilleur checkpoint (PRD: priorité `BLEU dev`).
- **Sorties**: `runs/<langpair>/<run_id>/checkpoints/{best,last}.pt`, `train.log`, `metrics.json`, copie `config.yaml`.
- **Validation**: courbe loss descendante, checkpoints présents, run traçable (config + logs).

### 6) Evaluate (`scripts/5_evaluate.py` / `pipeline.py evaluate`)
- **Module**: `scripts/5_evaluate.py` (implémenté).
- **But**: mesurer objectivement la qualité de traduction.
- **Entrées**: `--config`, `--run-id`, `--checkpoint`, `--beam-size`.
- **Actions**:
  - décodage `valid`/`test`,
  - calcul SacreBLEU (et métriques associées) avec protocole fixe.
- **Sorties**: `runs/<pair>/<run_id>/eval/{dev,test}_predictions.txt`, `sacrebleu_{dev,test}.txt`, `metrics.json`.
- **Validation**: métriques produites et comparables entre runs (même commande/protocole).

### 7) Infer (`scripts/6_infer.py` / `pipeline.py infer`)
- **Module**: `scripts/6_infer.py` (implémenté).
- **But**: traduire de nouveaux audios hors dataset d’entraînement.
- **Entrées**: `--checkpoint`, `--input-audio`, optionnel `--config`, `--beam-size`.
- **Actions**: chargement du checkpoint, décodage greedy des audios fournis.
- **Sorties**: `inference/predictions.jsonl` (ou chemin `--output`), une ligne JSON par audio.
- **Validation**: prédictions générées pour chaque entrée audio, format de sortie exploitable.

> Statut actuel: les étapes **`preflight` à `infer` sont implémentées**. Le pipeline est exécutable de bout en bout avec config de run.

---

## Commandes par étape

```bash
python scripts/0_preflight.py --min-disk-gb 200 --check-gpu
# équivalent :
python scripts/pipeline.py preflight --min-disk-gb 200 --check-gpu

python scripts/1_download.py
python scripts/pipeline.py download --langpairs fr-en,fr-pt

python scripts/pipeline.py prepare --langpair fr-es \
  --sample-rate 16000 --min-duration 1.0 --max-duration 30.0

python scripts/pipeline.py spm --langpair fr-es --vocab-size 1000

python scripts/pipeline.py train --config configs/fr-es/base.yaml --run-id run_001

python scripts/pipeline.py evaluate --config configs/fr-es/base.yaml --run-id run_001

python scripts/pipeline.py infer \
  --checkpoint runs/fr-es/run_001/checkpoints/best.pt \
  --input-audio path/to/audio.wav
```

Options communes : `--verbose`, `--dry-run`, `--log-file`.

---

## Structure du dépôt (cible)

```text
S3T/
  AGENTS.md
  tests/
  pyproject.toml
  .pre-commit-config.yaml
  scripts/
    bootstrap.sh        # Bootstrap environnement
    pipeline.py           # Orchestrateur CLI (routeur)
    0_preflight.py        # Stage 0 — preflight
    1_download.py         # Stage 1 — download
    2_prepare.py          # Stage 2 — prepare
    3_spm.py              # Stage 3 — tokenization
    4_train.py            # Stage 4 — train
    5_evaluate.py         # Stage 5 — evaluate
    6_infer.py            # Stage 6 — infer
    st_common.py          # Utilitaires partagés train/evaluate/infer
  configs/              # YAML par langpair (à créer)
  datasets/
    raw/
    processed/
    manifests/
  runs/                 # checkpoints, logs, eval
  artifacts/            # rapports preflight, stats data
  inference/
```

---

## Jalons go/no-go (résumé PRD)

| Phase | Critère |
|-------|---------|
| Bootstrap | venv OK, `torch.cuda.is_available()` si GPU |
| Preflight | rapport JSON sans blocage |
| Prepare | 0 fuite train/valid/test, manifests propres |
| Train | loss ↓, `BLEU dev` > baseline |
| Evaluate | signature SacreBLEU loggée, artifacts reproductibles |

Détails : [PRD.md](PRD.md) §4 et [README_experiments.md](README_experiments.md).

---

## Codes de sortie (`pipeline.py`)

| Code | Signification |
|------|----------------|
| 0 | Succès |
| 2 | Erreur d’arguments |
| 4 | Erreur d'exécution stage (ex: download/train) |

---

## Prochaines étapes de développement

1. Ajouter des `configs/fr-en/base.yaml`, `configs/fr-pt/base.yaml`, `configs/fr-es/base.yaml`
2. Brancher une stratégie de décodage beam search complète (actuellement greedy baseline)
3. Ajouter le tracking systématique dans `runs/experiments_tracking.csv`
4. Renforcer l'évaluation (ablations freeze/vocab/beam) et comparaison Table 8
5. Industrialiser l'intégration des checkpoints Pantagruel HF (cache + vérifs preflight)
