# Cascade — baseline ASR→MT (fr→en)

Traduction de la parole en **deux étages**, comme les systèmes cascade cités dans l’article Pantagruel :

1. **ASR** — audio français → transcription française  
2. **MT** — transcription française → texte anglais  

Les métriques finales utilisent le **même protocole SacreBLEU** que la baseline ST, `speechLLM` et Gemini (`runs/<langpair>/<run_id>/eval/`).

## Statut

| Composant | Statut |
|-----------|--------|
| Routeur `4_cascade/pipeline.py` | implémenté |
| Configs YAML (`configs/fr-en/`) | implémenté |
| `--dry-run` (evaluate / infer) | implémenté |
| Backends ASR (Whisper, etc.) | **à implémenter** (`cascade_common.transcribe_french`) |
| Backends MT (Marian, NLLB, etc.) | **à implémenter** (`cascade_common.translate_french_to_english`) |
| Évaluation SacreBLEU complète | bloquée tant que ASR+MT ne sont pas câblés |

## Prérequis

- Étapes communes `0`–`2` du dépôt ([README](../README.md)) : manifests TSV + WAV 16 kHz mono.
- Dépendances : mêmes que le projet (`pip install -r requirements.txt`) ; les backends ASR/MT ajouteront probablement `transformers` / modèles HF (déjà présents pour Pantagruel).

## Usage CLI

Depuis la racine du dépôt, venv activé :

```bash
# Plan d'évaluation (sans charger les modèles)
python 4_cascade/pipeline.py evaluate \
  --config 4_cascade/configs/fr-en/cascade.yaml \
  --run-id run_001_cascade_utterance \
  --dry-run

# Inférence (plan)
python 4_cascade/pipeline.py infer \
  --config 4_cascade/configs/fr-en/cascade.yaml \
  --input-audio datasets/processed/fr-en/valid/EXAMPLE.wav \
  --dry-run
```

Segments **sentence_like** : utiliser [`configs/fr-en/cascade_sentence.yaml`](configs/fr-en/cascade_sentence.yaml).

Quand les backends seront branchés :

```bash
python 4_cascade/pipeline.py evaluate \
  --config 4_cascade/configs/fr-en/cascade.yaml \
  --run-id run_001_cascade_utterance -v

python 4_cascade/pipeline.py infer \
  --config 4_cascade/configs/fr-en/cascade.yaml \
  --input-audio path/to/audio.wav
```

## Config YAML

| Section | Rôle |
|---------|------|
| `experiment` | Nom, paire, `output_dir` par défaut |
| `data` | `valid_manifest`, `test_manifest`, `segment_mode` (traçabilité) |
| `asr` | `backend`, `model_id` |
| `mt` | `backend`, `model_id` |
| `decode` | `asr_language`, `mt_max_length` |

## Codes de sortie

| Code | Signification |
|------|----------------|
| `0` | Succès |
| `2` | Config ou fichiers manquants |
| `3` | Pipeline non câblé (ASR/MT non implémentés) |

## Backends prévus (implémentation)

| Étape | Backend par défaut (config) | Notes |
|-------|----------------------------|--------|
| ASR | `whisper` / `openai/whisper-large-v3` | Alternative : encodeur Pantagruel / Whisper fine-tuné |
| MT | `marian` / `Helsinki-NLP/opus-mt-fr-en` | Alternative : NLLB, mBART |

La normalisation du texte intermédiaire (FR) et de la sortie (EN) devra rester **alignée** sur `2_prepare` (`text_norm`, casse) pour une comparaison équitable avec les autres variantes.

## Artefacts

Même contrat que Gemini :

```text
runs/fr-en/<run_id>/eval/
  dev_predictions.txt
  test_predictions.txt
  sacrebleu_dev.txt
  sacrebleu_test.txt
  metrics.json    # pipeline: cascade_asr_mt
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `pipeline.py` | Routeur CLI (`evaluate`, `infer`) |
| `cascade_common.py` | Settings, enchaînement ASR→MT (stubs) |
| `evaluate_cascade.py` | SacreBLEU valid/test |
| `infer_cascade.py` | JSONL inférence |

Voir aussi [docs/PRD.md §2.3.3](../docs/PRD.md#233-baseline-cascade-asrmt).
