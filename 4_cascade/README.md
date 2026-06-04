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
| Backends ASR (`whisper`) | implémenté (`cascade_common.transcribe_french`) |
| Backends MT (`marian`) | implémenté (`cascade_common.translate_french_to_english`) |
| Évaluation SacreBLEU complète | implémenté (GPU recommandé ; smoke `--limit 5`) |

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

```bash
# Smoke (5 segments)
python 4_cascade/pipeline.py evaluate \
  --config 4_cascade/configs/fr-en/cascade_sentence.yaml \
  --run-id run_000_cascade_smoke5 --limit 5 -v

# Run complet sentence_like
python 4_cascade/pipeline.py evaluate \
  --config 4_cascade/configs/fr-en/cascade_sentence.yaml \
  --run-id run_001_cascade_sentence_like -v

python 4_cascade/pipeline.py infer \
  --config 4_cascade/configs/fr-en/cascade_sentence.yaml \
  --input-audio path/to/audio.wav -v
```

**VRAM :** Whisper large-v3 + Marian — prévoir ~6–10 Go ; ne pas lancer en parallèle d'un long `1_Transformer/train` sur la même GPU.

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
| `3` | Backend ASR/MT non supporté (autre que whisper / marian) |

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
| `cascade_common.py` | Settings, Whisper ASR, Marian MT, cache modèles |
| `evaluate_cascade.py` | SacreBLEU valid/test |
| `infer_cascade.py` | JSONL inférence |

Voir aussi [docs/PRD.md §2.3.3](../docs/PRD.md#233-baseline-cascade-asrmt).
