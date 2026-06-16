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
| ST **L-14k v2** | `run_014_transformer_baseline_utterance_large_14k_v2` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v2.yaml` | early stop | **ok** — 17,12 / 17,21 (Modyco, juin 2026) |
| speechLLM **L-14k unfreeze** | `run_015_speechllm_b1_utterance_large_14k_unfreeze` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k_unfreeze.yaml` | **20k updates** (early stop) | **ok** — 3,90 / 3,65 (Modyco, juin 2026) — sous run_012 gelé |
| ST **L-114k** | `run_011_transformer_baseline_utterance_large_114k` | `1_Transformer/configs/fr-en/base_utterance_large_114k.yaml` | **80k updates** | ~10–12 h GPU (ordre de grandeur, d’après run_010) |
| speechLLM **L-14k** | `run_012_speechllm_b1_utterance_large_14k` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml` | **20k updates** | ~4–6 h GPU |
| speechLLM **L-114k** | `run_013_speechllm_b1_utterance_large_114k` | `2_speechLLM/configs/fr-en/b1_utterance_large_114k.yaml` | **20k updates** | **ok** — 15,92 / 15,24 (OVH, juin 2026) |
| ST **L-114k v2** | `run_016_transformer_baseline_utterance_large_114k_v2` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v2.yaml` | early stop | **ok** — 20,30 / 19,63 (OVH, juin 2026, ~9,1 h GPU, early stop @~21k) |
| ST **L-14k v3** | `run_020_transformer_baseline_utterance_large_14k_v3` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v3.yaml` | early stop | **ok** — 22,05 / **21,22** (Modyco, juin 2026, eval dev complet) |
| ST **L-114k v3** | `run_019_transformer_baseline_utterance_large_114k_v3` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v3.yaml` | early stop | **ok** — 21,09 / **20,19** (OVH, juin 2026) |
| ST **L-14k v4** (batch 64) | `run_024_transformer_baseline_utterance_large_14k_v4` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v4.yaml` | early stop @2,5k | **échec** — 0,20 / **0,35** (Modyco, collapse) |
| ST **L-14k v5** (SpecAugment) | `run_026_transformer_baseline_utterance_large_14k_v5` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v5.yaml` | early stop @~55k | **ok** — 26,57 / **26,12** (Modyco, ~7,6 h GPU, 14 juin 2026) |
| ST **L-114k v4** (batch 64) | `run_025_transformer_baseline_utterance_large_114k_v4` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v4.yaml` | early stop @~2,6k | **échec** — 0,24 / **0,31** (OVH, collapse) |
| speechLLM **L-114k v2** | `run_017_speechllm_b1_utterance_large_114k_v2` | `2_speechLLM/configs/fr-en/b1_utterance_large_114k_v2.yaml` | **20k updates** (early stop) | **échec** — 6,56 / 5,60 (OVH, max **128 tok**) |
| speechLLM **L-14k v3** | `run_021_speechllm_b1_utterance_large_14k_v3` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k_v3.yaml` | early stop | **échec** — 5,84 / 5,48 (Modyco, max **128 tok**) |
| speechLLM **L-14k replicate** | `run_023_speechllm_b1_utterance_large_14k_replicate` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k_replicate.yaml` | **20k updates** | **ok** — 15,26 / **14,23** (Modyco, **48 tok**) |
| speechLLM **L-114k v3** | `run_022_speechllm_b1_utterance_large_114k_v3` | `2_speechLLM/configs/fr-en/b1_utterance_large_114k_v3.yaml` | **20k updates** | **échec** — 5,28 / **4,78** (OVH, max **128 tok**) |
| speechLLM **L-14k + Qwen2.5-3B** | `run_018_speechllm_b2bis_utterance_large_14k_qwen25_3b` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k_qwen25_3b.yaml` | **20k updates** (early stop) | **ok** — 13,96 / 12,95 (Modyco) — sous run_012 Phi-2 |
| ST **L-14k v6 long** | `run_027_transformer_baseline_utterance_large_14k_v6_long` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v6_long.yaml` | early stop | **ok** — 26,37 / **25,12** (Modyco, 14 juin 2026 — sous run_026) |
| ST **L-114k v5** (SpecAugment) | `run_028_transformer_baseline_utterance_large_114k_v5` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v5.yaml` | early stop @~31k | **ok** — 24,08 / **23,51** (OVH, 14 juin 2026) — meilleur L-114k local |
| ST **L-114k v6 long** | `run_030_transformer_baseline_utterance_large_114k_v6_long` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v6_long.yaml` | 120k updates | **non planifié** (waiter obsolète retiré sur OVH, 15 juin) |
| ST **L-14k v7** (SPM 5k) | `run_031_transformer_baseline_utterance_large_14k_v7_spm5k` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v7_spm5k.yaml` | early stop | **ok** — 24,24 / **24,02** (Modyco, 14 juin) — sous run_026 |
| speechLLM **L-114k replicate** | `run_032_speechllm_b1_utterance_large_114k_replicate` | `2_speechLLM/configs/fr-en/b1_utterance_large_114k_replicate.yaml` | **20k updates** | **ok** — 15,14 / **14,15** (OVH, **48 tok** — sous run_013 **15,24**) |
| ST **L-114k v7** (SPM 5k) | `run_033_transformer_baseline_utterance_large_114k_v7_spm5k` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v7_spm5k.yaml` | 80k updates | **en cours** (OVH, @ ~36,5k/80k, best dev **23,40** @ 32k — early stop probable sous ~2–4 h) |
| ST **L-14k v8** (SPM 8k) | `run_034_transformer_baseline_utterance_large_14k_v8_spm8k` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v8_spm8k.yaml` | early stop | **ok** — 23,36 / **22,24** (Modyco, 14 juin) — sous run_031 et run_026 |
| ST **B-1k v5** (SpecAugment) | `run_035_transformer_baseline_utterance_b1k_v5` | `1_Transformer/configs/fr-en/base_utterance_b1k_v5.yaml` | 80k updates | **ok** — 20,18 / **19,75** (Modyco, 15 juin 2026) |
| ST **L-14k v9** (warmup 10k) | `run_036_transformer_baseline_utterance_large_14k_v9_warmup10k` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v9_warmup10k.yaml` | early stop | **interrompu** (Modyco, @ ~5k — reprise `--resume` possible) |
| ST **L-14k v9** (SpecAugment fort) | `run_037_transformer_baseline_utterance_large_14k_v9_specaug_strong` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v9_specaug_strong.yaml` | early stop | **non lancé** |
| ST **L-114k v9** (SpecAugment freq) | `run_038_transformer_baseline_utterance_large_114k_v9_specaug_freq` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v9_specaug_freq.yaml` | early stop | **en file** (OVH, après run_033 — ~9–12 h GPU) |
| ST **L-114k v10** (warmup 10k) | `run_042_transformer_baseline_utterance_large_114k_v10_warmup10k` | `1_Transformer/configs/fr-en/base_utterance_large_114k_v10_warmup10k.yaml` | early stop | **en file** (OVH, après run_038 — ~10–12 h GPU) |
| ST **L-14k v5 replicate** | `run_043_transformer_baseline_utterance_large_14k_v5_replicate` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v5_replicate.yaml` | early stop | **prêt** (Modyco — validation run_026 @ 26,12) |
| speechLLM **L-14k v5** (SpecAugment) | `run_039_speechllm_b1_utterance_large_14k_v5_specaug` | `2_speechLLM/configs/fr-en/b1_utterance_large_14k_v5_specaug.yaml` | **20k updates** | **ok** — 14,59 / **13,84** (Modyco, 16 juin — sous run_023 **14,23**) |
| Speech_Text **utterance v2** | `run_040_pantagruel_multimodal_utterance_v2` | `5_Pantagruel_multimodal/configs/fr-en/base_utterance_v2.yaml` | early stop | **échec** (Modyco — HF `PantagrueLLM/Speech_Text_Base_fr_1K_4GB` 404) |
| ST **L-14k v10** (finetune freq) | `run_041_transformer_finetune_utterance_large_14k_v10_specaug_freq_from_run026` | `1_Transformer/configs/fr-en/base_utterance_large_14k_v10_specaug_freq_finetune.yaml` | finetune 69k max | **en cours** (Modyco, depuis run_026 @ ~44k, best dev hérité **25,64**) |

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
# speechLLM L-14k replicate 48 tok (Modyco, run_023 — après échec run_021 128 tok) :
nohup bash scripts/run_modyco_speechllm_14k_replicate.sh \
  > logs/run_023_speechllm_chain_wrapper.log 2>&1 &
# ST L-14k v5 SpecAugment (Modyco, run_026) — terminé 26,12 test ; meilleur ST local :
# ST L-14k v6 long (Modyco, run_027) — terminé 25,12 test (sous run_026) :
# ST L-14k v7 SPM 5k — terminé 24,02 test (Modyco, run_031) :
# ST B-1k v5 SpecAugment (Modyco, run_035) — terminé ; rappatrier eval/ :
bash scripts/pull_remote_results.sh run_035_transformer_baseline_utterance_b1k_v5
# Amélioration run_026 (Modyco) — chaîne nocturne 16 juin : eval run_036 → run_039 ok → run_040 échec HF → run_037 non lancé.
# Finetune run_041 (SpecAugment freq depuis run_026, ~3,5 h) :
nohup bash scripts/run_modyco_st_14k_v10_specaug_freq_finetune.sh \
  > logs/run_041_modyco_wrapper.log 2>&1 &
# OVH (16 juin 2026) : **run_033** ST L-114k SPM 5k **en cours** (@ ~36,5k/80k, GPU ~79 %, best dev **23,40**) ;
#   waiters actifs → **run_038** puis **run_042** (`chain_038_042_ovh_wait.log`) ; doublon possible sur run_038 (`run_038_ovh_wait_chain.log`).
#   `run_032` terminé (14,15 test) ; `run_028` meilleur L-114k terminé (**23,51**).
#   Suivi : tail -f ~/S3T/logs/run_033_*_spm_train_eval.log
# Terminés récents : run_039 (13,84), run_032 (14,15), run_035 (19,75), run_026 (26,12)
# speechLLM legacy (run_012/013 déjà terminés sur OVH) :
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
