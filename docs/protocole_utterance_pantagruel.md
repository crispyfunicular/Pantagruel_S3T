# Protocole utterance — comparaison Table 8 Pantagruel

Ce document décrit comment obtenir des **SacreBLEU** sur la **même segmentation** que l’article Pantagruel (2026) : segments **utterance** natifs m-TEDx, pas la fusion `sentence_like`.

## Segmentation

| Mode | Chemins | Usage |
|------|---------|--------|
| **utterance** (Pantagruel / Table 8) | `datasets/manifests/fr-en/`, `datasets/processed/fr-en/` | Comparaison papier (~17,5 BLEU Pantagruel-B-1k) |
| **sentence_like** (S3T expérimental) | `datasets/manifests_sentence/fr-en/`, `datasets/processed_sentence/fr-en/` | Runs déjà reportés dans `rapport.md` |

Préparation utterance (si manifests absents sur la machine GPU) :

```bash
python scripts_communs/pipeline.py prepare --langpair fr-en
# équivalent : --segment-mode utterance (défaut)
```

## Runs cibles (fr→en)

| Variante | Run ID | Config | Entraînement | Durée ordre de grandeur |
|----------|--------|--------|--------------|-------------------------|
| ST Table 8 | `run_002_transformer_baseline_utterance` | `1_Transformer/configs/fr-en/base_utterance.yaml` | **80k updates** | ~8 h GPU |
| Gemini Flash | `run_002_gemini_flash_utterance` | `3_Gemini/configs/fr-en/gemini_flash_utterance.yaml` | aucun (API) | ~1–2 h + coût API |
| Cascade | `run_001_cascade_utterance` | `4_cascade/configs/fr-en/cascade.yaml` | aucun | ~3–5 h GPU |
| speechLLM B1 | `run_003_speechllm_b1_utterance_long` | `2_speechLLM/configs/fr-en/b1_utterance_long.yaml` | **20k updates** | ~3–4 h GPU |
| ST **L-14k** | `run_010_transformer_baseline_utterance_large_14k` | `1_Transformer/configs/fr-en/base_utterance_large_14k.yaml` | **80k updates** | ~12–18 h GPU |
| ST **L-114k** | `run_011_transformer_baseline_utterance_large_114k` | `1_Transformer/configs/fr-en/base_utterance_large_114k.yaml` | **80k updates** | ~12–18 h GPU |
| speechLLM **L-14k** | `run_012_speechllm_b1_utterance_large_14k` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml` | **20k updates** | ~4–6 h GPU |
| speechLLM **L-114k** | `run_013_speechllm_b1_utterance_large_114k` | `2_speechLLM/configs/fr-en/b1_utterance_large_114k.yaml` | **20k updates** | ~4–6 h GPU |

**14k / 114k** = heures de pré-entraînement Pantagruel (LeBenchmark / INA), pas un autre corpus m-TEDx : mêmes manifests `datasets/manifests/fr-en/`. Encodeurs HF : `PantagrueLLM/speech-large-14K`, `PantagrueLLM/speech-large-114K` (Table 8 : ~24,0 et ~25,2 BLEU test).

Référence papier (Table 8, fr→en) : **Pantagruel-B-1k ≈ 17,5 ± 0,4 BLEU** ; **L-14k ≈ 24,0** ; **L-114k ≈ 25,2** (utterance, protocole LeBenchmark / fairseq).

## Commandes

### Évaluations rapides (sans train)

```bash
source .venv/bin/activate
export GEMINI_API_KEY=...   # optionnel pour Gemini

bash scripts/run_utterance_pantagruel_metrics.sh          # cascade + gemini
bash scripts/run_utterance_pantagruel_metrics.sh cascade  # cascade seul
```

### Baseline ST (réplication Table 8)

```bash
nohup bash 1_Transformer/scripts/run_002_baseline_utterance_nohup.sh \
  > logs/run_002_transformer_utterance_nohup_wrapper.log 2>&1 &
```

### speechLLM sur utterance

```bash
python 2_speechLLM/pipeline.py run \
  --config 2_speechLLM/configs/fr-en/b1_utterance_long.yaml \
  --run-id run_003_speechllm_b1_utterance_long
```

### Encodeurs Large 14k / 114k (Table 8)

Vérifier HF + VRAM avant les runs longs :

```bash
python scripts/smoke_pantagruel_encoders.py --encoders 14k,114k
bash scripts/run_pantagruel_encoder_scale_utterance.sh dry-run
```

Entraînements (orchestrateur ou nohup ST) :

```bash
bash scripts/run_pantagruel_encoder_scale_utterance.sh smoke
# ST L-14k / L-114k (tour GPU) :
nohup bash 1_Transformer/scripts/run_010_baseline_utterance_14k_nohup.sh \
  > logs/run_010_wrapper.log 2>&1 &
nohup bash 1_Transformer/scripts/run_011_baseline_utterance_114k_nohup.sh \
  > logs/run_011_wrapper.log 2>&1 &
# speechLLM :
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-14k
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-114k
# Ré-évaluer si checkpoints déjà présents :
bash scripts/run_pantagruel_encoder_scale_utterance.sh eval-all
```

## Règles de comparaison

1. Ne pas évaluer un checkpoint **entraîné sur sentence_like** sur des manifests **utterance** (train/eval incohérent).
2. Même **SacreBLEU** signé pour tous les runs ([protocole_evaluation.md](protocole_evaluation.md)) ; noter `segment_mode` dans `experiments_tracking.csv`.
3. ST : éval v1 en **greedy** (objectif papier : beam 5 — voir protocole §4.1) ; comparer au papier en utterance pour la ligne « Table 8 ».

## Suivi

Après chaque run :

```bash
python scripts_communs/update_experiments_tracking.py --run-dir runs/fr-en/<run_id>
```

Mettre à jour `rapport.md` et `README.md` lorsque les scores utterance sont figés.
