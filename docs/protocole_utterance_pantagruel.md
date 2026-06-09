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
| ST Table 8 | `run_002_transformer_baseline_utterance` | `1_Transformer/configs/fr-en/base_utterance.yaml` | **80k updates** | ~8 h GPU — **échec** (collapse, BLEU test 3,79) |
| ST Table 8 **v2** | `run_004_transformer_baseline_utterance_v2` | `1_Transformer/configs/fr-en/base_utterance_v2.yaml` | early stop @20k + gel 5k | **ok** — 16,84 / 16,68 (tour, 2026-06-05) |
| Gemini Flash | `run_002_gemini_flash_utterance` | `3_Gemini/configs/fr-en/gemini_flash_utterance.yaml` | aucun (API) | ~1–2 h + coût API |
| Cascade | `run_001_cascade_utterance` | `4_cascade/configs/fr-en/cascade.yaml` | aucun | ~3–5 h GPU |
| speechLLM B1 | `run_003_speechllm_b1_utterance_long` | `2_speechLLM/configs/fr-en/b1_utterance_long.yaml` | **20k updates** | **ok** — 10,00 / 7,47 (tour, 2026-06-05) |
| ST **L-14k** | `run_010_transformer_baseline_utterance_large_14k` | `1_Transformer/configs/fr-en/base_utterance_large_14k.yaml` | **80k updates** | **échec** — 0,00 / 0,00 (tour, 2026-06-09, **~10 h 23** train + **~11 min** éval) |
| ST **L-14k v2** | `run_014_transformer_baseline_utterance_large_14k_v2` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v2.yaml` | early stop (prévu) | **à lancer** — gel 5k, LR 1e-4 ; ~4–8 h GPU estimées |
| ST **L-114k** | `run_011_transformer_baseline_utterance_large_114k` | `1_Transformer/configs/fr-en/base_utterance_large_114k.yaml` | **80k updates** | ~10–12 h GPU (ordre de grandeur, d’après run_010) |
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

**run_002** (juin 2026, tour) : entraînement terminé mais **mode collapse** (~26k updates) — BLEU dev/test **3,90 / 3,79** ; hypothèses répétitives (`iveive…`, `me me me…`). Non comparable au papier (~17,5).

**run_004 v2** (correctifs : `freeze_encoder_updates: 5000`, `early_stopping_patience: 2`, `learning_rate_peak: 1e-4`) — **terminé 2026-06-05** (tour, ~1 h 15) : early stop à **20 000** updates (meilleur checkpoint @16k) ; SacreBLEU corpus **dev 16,84 / test 16,68** (`eval/sacrebleu_*.txt`). Proche Table 8 (~17,5) ; écart ~0,8 BLEU (greedy v1 vs beam 5 papier, stack PyTorch/HF).

```bash
# ThinkPad → tour (avant lancement)
rsync -avz 1_Transformer/4_train.py mpellissier@10.8.0.2:~/S3T/1_Transformer/
rsync -avz 1_Transformer/scripts/run_004_baseline_utterance_v2_nohup.sh \
  mpellissier@10.8.0.2:~/S3T/1_Transformer/scripts/
rsync -avz 1_Transformer/configs/fr-en/base_utterance_v2.yaml \
  mpellissier@10.8.0.2:~/S3T/1_Transformer/configs/fr-en/
rsync -avz scripts/run_night_end_of_day.sh mpellissier@10.8.0.2:~/S3T/scripts/

# Tour
cd ~/S3T && source .venv/bin/activate
chmod +x 1_Transformer/scripts/run_004_baseline_utterance_v2_nohup.sh
mkdir -p logs
nohup bash 1_Transformer/scripts/run_004_baseline_utterance_v2_nohup.sh \
  > logs/run_004_transformer_utterance_v2_wrapper.log 2>&1 &
echo $! > logs/run_004_transformer_utterance_v2.pid
tail -f logs/run_004_transformer_baseline_utterance_v2_spm_train_eval.log
```

Contrôle qualité après ~30 min :

```bash
head -3 runs/fr-en/run_004_transformer_baseline_utterance_v2/eval/test_predictions.txt
grep Early logs/run_004_transformer_baseline_utterance_v2_spm_train_eval.log
python -c "import json; print(json.loads(open('runs/fr-en/run_004_transformer_baseline_utterance_v2/metrics.json').read())['early_stopped'])"
```

Legacy run_002 (ne pas relancer sauf ablation) :

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

**run_010** (juin 2026, tour Modyco) : entraînement **80k updates** terminé mais **mode collapse** — BLEU dev/test **0,00** (meilleur dev en cours de train ~0,025) ; hypothèses répétitives (`I I I…`). Durée mesurée (`metrics.json`) : **10 h 23 min** train GPU (37 380 s) + **10 min** éval (626 s) ; fenêtre **2026-06-08 22h17 → 2026-06-09 08h53**. Même cause probable qu’en B-1k (`run_002`) : gel encodeur trop court (1k), LR 2e-4, pas d’early stop.

**run_014 v2** (retry) : correctifs calqués sur `run_004` (`freeze_encoder_updates: 5000`, `early_stopping_patience: 2`, `learning_rate_peak: 1e-4`). Lancement **nocturne** sur tour partagée (vérifie qu’aucun `pipeline.py train` n’est actif) :

```bash
# Tour Modyco (soirée, GPU partagé)
cd ~/S3T && source .venv/bin/activate
mkdir -p logs
nohup bash scripts/run_modyco_night_st_large_14k_v2.sh \
  > logs/run_014_st_large_14k_v2_chain_wrapper.log 2>&1 &
tail -f logs/run_014_transformer_baseline_utterance_large_14k_v2_spm_train_eval.log
```

Entraînements (orchestrateur ou nohup ST) :

```bash
bash scripts/run_pantagruel_encoder_scale_utterance.sh smoke
# ST L-114k (tour GPU, après run_014 v2) :
nohup bash 1_Transformer/scripts/run_011_baseline_utterance_114k_nohup.sh \
  > logs/run_011_wrapper.log 2>&1 &
# speechLLM (OVH ou tour selon planning) :
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-14k
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-114k
# Ré-évaluer si checkpoints déjà présents :
bash scripts/run_pantagruel_encoder_scale_utterance.sh eval-all
```

Legacy **run_010** (ne pas relancer — échec documenté) :

```bash
nohup bash 1_Transformer/scripts/run_010_baseline_utterance_14k_nohup.sh \
  > logs/run_010_wrapper.log 2>&1 &
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
