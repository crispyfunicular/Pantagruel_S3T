# Plan d'amélioration pipeline S3T — cibles Table 8

Document de référence pour rapprocher les scores du pipeline S3T de la **Table 8** du papier Pantagruel ([`docs/Pantagruel_2026.pdf`](Pantagruel_2026.pdf)) : ST fr→en/pt/es sur m-TEDx (utterance), NER, SLU, SER.

**Baseline actuelle (juin 2026)** : meilleur ST local `run_026` v5 SpecAugment — **26,12 BLEU** test (Modyco, vocab 1k). L-114k : `run_028` **23,51** test (OVH). Piste 3 SPM L-14k : **sous-performe** vocab 1k (`run_031` 24,02 ; `run_034` 22,24). **Améliorations run_026** : `run_036`–`run_040` **en cours / en file** (Modyco + OVH). OVH : `run_032` speechLLM **en cours** ; `run_030`/`run_033` **en file** ; `run_038` SpecAugment freq **en file**.

**Runs piste 1 (batch 64)** :

- Modyco L-14k : `run_024` — **échec** (collapse @ 0,35 BLEU test)
- OVH L-114k : `run_025` — **échec** (collapse @ **0,31** BLEU test, early stop @~2,6k)

**Runs piste 2 (SpecAugment)** :

- Modyco L-14k : `run_026` — **ok** — 26,57 / **26,12** test ; `run_027` v6 long **ok** — 26,37 / **25,12** (sous run_026)
- OVH L-114k : `run_028` v5 **ok** — 24,08 / **23,51** test ; `run_030` v6 long **en file** (bloqué par run_032)

---

## État des lieux — écarts identifiés

| Élément | Papier / LeBenchmark 2.0 | S3T actuel | Impact estimé |
|---------|--------------------------|------------|---------------|
| Batch effectif | 64 – 256 tokens·s | **8** (`grad_accum=8`) | ★★★ |
| SpecAugment | oui (time + feature masks) | **implémenté** (temporel + fréquentiel ST ; temporel speechLLM) | ★★★ |
| Vocab SPM | ≈8 k (LeBenchmark 2.0) | **1 k** | ★★ |
| Warmup | 10 000 steps (PRD) | **configurable** (`warmup_updates` lu par `4_train.py`) | ★★ |
| Décodage eval | beam 5 | **beam 5** (`5_evaluate.py`) | ★ |
| Sélection ckpt | beam BLEU dev | **greedy** en train (beam à l’éval finale) | ★ |
| fr→es / fr→pt | runs L-14k + L-114k | partiels ou absents | ★ |
| Tâches NER / SLU / SER | tête downstream dédiée | **non implémenté** | — |

Autres écarts structurels (stack fairseq vs PyTorch/HF, pas de speed perturbation m-TEDx) : voir [`docs/protocole_utterance_pantagruel.md`](protocole_utterance_pantagruel.md) et [`web/variantes.md`](../web/variantes.md).

---

## Piste 1 — Batch effectif (impact ★★★)

**Problème** : `batch_size=1 × gradient_accumulation=8` → batch effectif **8**, contre 64–256 dans fairseq. Instabilité du gradient et variabilité entre runs.

**Action cible** (configs `_v4.yaml` dans [`1_Transformer/configs/fr-en/`](../1_Transformer/configs/fr-en/)) :

```yaml
train:
  batch_size: 1
  gradient_accumulation: 64    # batch effectif 64
  learning_rate_peak: 0.0002   # montée linéaire avec le batch
  warmup_updates: 1000           # 10 % de max_updates (ratio fairseq)
  max_updates: 10000           # même micro-batches que v3 : 80k×8 = 10k×64
  freeze_encoder_updates: 625  # même ratio que v3 (5k/80k)
```

**Adaptation budget GPU (~10 h Modyco)** : `max_updates: 10000` conserve le volume de micro-batches de `run_020` (80k updates × grad_accum 8). Une recette « full fairseq » avec `max_updates: 100000` et `warmup_updates: 10000` est réservée à OVH / runs longs.

**Implémenté** :

- Config : [`base_utterance_large_14k_v4.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v4.yaml)
- Run : `run_024_transformer_baseline_utterance_large_14k_v4`
- Script Modyco : [`scripts/run_modyco_st_14k_v4.sh`](../scripts/run_modyco_st_14k_v4.sh)

**OVH L-114k** :

- Config : [`base_utterance_large_114k_v4.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_114k_v4.yaml)
- Run : `run_025_transformer_baseline_utterance_large_114k_v4`
- Script OVH : [`scripts/run_ovh_st_114k_v4.sh`](../scripts/run_ovh_st_114k_v4.sh)

**Gain estimé** : +1 à 2 BLEU.

---

## Piste 2 — SpecAugment (impact ★★★)

**Problème** : LeBenchmark 2.0 applique du masquage temporel (et parfois fréquentiel) sur les features de l'encodeur SSL (`apply_mask=True` côté fairseq).

**Action** :

- Flag YAML `spec_augment` lu par [`1_Transformer/4_train.py`](../1_Transformer/4_train.py) et [`2_speechLLM/train.py`](../2_speechLLM/train.py).
- Masquage temporel : [`apply_waveform_time_mask`](../scripts_communs/st_common.py) sur les formes d'onde.
- Masquage fréquentiel : [`apply_feature_freq_mask`](../scripts_communs/st_common.py) sur les features encodeur (run_038).

Exemple de flag YAML :

```yaml
spec_augment:
  enabled: true
  mask_time_prob: 0.05
  mask_time_length: 10
```

**Implémenté (v5)** :

- Config : [`base_utterance_large_14k_v5.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v5.yaml) (recette `run_020` v3 + SpecAugment)
- Run : `run_026_transformer_baseline_utterance_large_14k_v5` — **ok** — 26,57 / **26,12** test (Modyco, ~7,6 h GPU, early stop @~55k)
- Run long suivant : `run_027_transformer_baseline_utterance_large_14k_v6_long` — **ok** — 26,37 / **25,12** test (Modyco, 14 juin — sous run_026)
- Script Modyco : [`scripts/run_modyco_st_14k_v6_long.sh`](../scripts/run_modyco_st_14k_v6_long.sh) (+ waiter [`run_modyco_wait_st_then_st_14k_v6_long.sh`](../scripts/run_modyco_wait_st_then_st_14k_v6_long.sh))

**OVH L-114k (SpecAugment + long)** :

- Config v5 : [`base_utterance_large_114k_v5.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_114k_v5.yaml) → `run_028`
- Config v6 long : [`base_utterance_large_114k_v6_long.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_114k_v6_long.yaml) → `run_030`
- Chaîne OVH : [`scripts/run_ovh_wait_st_chain_114k_v5_v6_long.sh`](../scripts/run_ovh_wait_st_chain_114k_v5_v6_long.sh)

**Gain estimé** : +1 à 2 BLEU (validé L-14k : run_026 **26,12** vs run_020 **21,22**).

**Améliorations run_026 (juin 2026)** :

| Run | Changement | Machine | Statut |
|-----|------------|---------|--------|
| `run_036` | `warmup_updates: 10000` | Modyco | **en cours** |
| `run_037` | `mask_time_prob: 0.10` | Modyco | **en file** |
| `run_038` | SpecAugment temporel + fréquentiel L-114k | OVH | **en file** |
| `run_039` | SpecAugment speechLLM L-14k | Modyco | **en file** |
| `run_040` | Speech_Text utterance (recette run_026) | Modyco | **en file** |

Scripts : [`run_modyco_wait_chain_post_035_improve_run026.sh`](../scripts/run_modyco_wait_chain_post_035_improve_run026.sh), [`run_ovh_wait_chain_post_032_033_st_specaug_freq.sh`](../scripts/run_ovh_wait_chain_post_032_033_st_specaug_freq.sh).

---

## Piste 3 — Vocabulaire SPM plus grand (impact ★★)

**Problème** : vocabulaire **1 k** sous-mots vs ≈**8 k** dans LeBenchmark 2.0 pour ST.

**Action** :

- Tester `--vocab-size 5000` puis `8000` via [`1_Transformer/3_spm.py`](../1_Transformer/3_spm.py).
- Configs v7 SPM 5k (recette run_026 v5 + SpecAugment) :
  - L-14k : [`base_utterance_large_14k_v7_spm5k.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v7_spm5k.yaml) → `run_031` (**ok** — 24,02 test, sous run_026)
  - L-114k : [`base_utterance_large_114k_v7_spm5k.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_114k_v7_spm5k.yaml) → `run_033` (**en file**, OVH)
- Config v8 SPM 8k :
  - L-14k : [`base_utterance_large_14k_v8_spm8k.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v8_spm8k.yaml) → `run_034` (**ok** — 22,24 test, sous run_031)
  - Scripts : [`run_modyco_wait_st_then_st_14k_v8_spm8k.sh`](../scripts/run_modyco_wait_st_then_st_14k_v8_spm8k.sh)
- **Attention** : changer le vocab invalide les checkpoints existants → run séparé obligatoire.

**Gain estimé** : incertain ; résultats juin 2026 L-14k : SPM 5k (**24,02**) et 8k (**22,24**) **sous** vocab 1k + SpecAugment (**26,12**, run_026).

---

## Piste 4 — Cohérence greedy / beam pour `best.pt` (impact ★)

**Problème** : [`1_Transformer/4_train.py`](../1_Transformer/4_train.py) sélectionne `best.pt` sur le BLEU **greedy** en cours d'entraînement ; [`5_evaluate.py`](../1_Transformer/5_evaluate.py) rapporte le BLEU **beam 5** (implémenté).

**Action restante** :

- Ajouter `eval_beam_during_training: true` dans les configs, ou
- Réévaluer les N derniers checkpoints au beam en fin de run.

**Gain estimé** : marginal.

---

## Piste 5 — Extensions multilingues fr→es et fr→pt (impact ★)

La Table 8 couvre trois directions ST. Le pipeline supporte déjà m-TEDx ; les paires es/pt sont les plus simples à ajouter.

**Action** : cloner les configs `fr-en` vers `fr-es` / `fr-pt` (L-14k et L-114k) et lancer les entraînements.

| Direction | Données train m-TEDx | Cible papier (L-14k / L-114k) |
|-----------|----------------------|-------------------------------|
| fr→en | 50 h | 24,0 / 25,2 BLEU |
| fr→es | 38 h | 25,5 / 25,4 BLEU |
| fr→pt | 25 h | 21,9 / 24,5 BLEU |

---

## Piste 6 — Tâches non-ST (NER, SLU, SER)

Ces benchmarks de la Table 8 ne sont **pas encore** dans le pipeline S3T :

| Tâche | Corpus | Métrique | Architecture papier |
|-------|--------|----------|---------------------|
| NER speech | PxCorpus, ETAPE | F1↑ / NEER↓ | encodeur SSL + sonde linéaire 3 couches |
| SLU | MEDIA | CER↓ | Transformer seq2seq concepts |
| SER | AlloSat | CCC↑ | encodeur gelé + 5-BiLSTM + régression |

**Action** : nouvelle variante ou module `5_downstream/` (hors scope ST actuel). Détails protocole : Annexe B.2 du papier Pantagruel.

---

## Ordre de priorité recommandé

1. **Piste 1** — batch ↑ → **échec** L-14k (`run_024`) et L-114k (`run_025`, 0,31 test)
2. **Piste 2** — SpecAugment → **succès L-14k** (`run_026`, **26,12** test) ; L-114k **ok** (`run_028`, **23,51** test OVH) ; longs L-14k **ok** (`run_027`, 25,12 — sous run_026) ; `run_030` **en file**
3. **Piste 3** — SPM 5k/8k L-14k **ok** mais **sous vocab 1k** (`run_031` 24,02 ; `run_034` 22,24) ; `run_033` en file OVH
4. **Piste 4** — sélection checkpoint au beam
5. **Piste 5** — configs et runs fr→es / fr→pt
6. **Piste 6** — NER / SLU / SER (développement nouveau)

---

## Checklist de suivi

| ID | Action | Statut |
|----|--------|--------|
| batch-grad-accum | Configs v4 batch 64, LR 2e-4 ; run L-14k Modyco | **échec** (`run_024`, 0,35 BLEU) |
| batch-grad-accum | Config v4 L-114k + run OVH | **échec** (`run_025`, 0,31 BLEU) |
| spec-augment | run_026 Modyco | **ok** — **26,12** test |
| spec-augment | run_028 OVH | **ok** — **23,51** test (meilleur L-114k local) |
| spec-augment | run_030 OVH v6 long | **en file** (bloqué par run_032) |
| spec-augment-long | run_027 (14k 120k) | **ok** — **25,12** test (sous run_026) |
| spm-vocab | SPM 5k run_031 Modyco | **ok** — **24,02** test (sous run_026) |
| spm-vocab | SPM 8k run_034 Modyco | **ok** — **22,24** test (sous run_031) |
| spm-vocab | SPM 5k run_033 OVH | **en file** |
| speechllm-114k-replicate | run_032 OVH (48 tok) | **en cours** (@ ~10k/20k, best dev ~10,6) |
| st-b1k-specaugment | run_035 Modyco (B-1k v5) | **terminé** (résultats à rappatrier) |
| improve-run026-warmup | run_036 Modyco (warmup 10k) | **en cours** |
| improve-run026-specaug-strong | run_037 Modyco (mask 0.10) | **en file** |
| improve-run026-specaug-freq | run_038 OVH (time + freq) | **en file** |
| improve-run026-sllm-specaug | run_039 Modyco (speechLLM) | **en file** |
| improve-run026-speechtext | run_040 Modyco (Speech_Text utterance) | **en file** |
| beam-consistency | `eval_beam_during_training` ou rééval beam | à faire (beam eval **ok**) |
| multilingual-runs | Configs fr→es / fr→pt L-14k / L-114k | à faire |
| downstream-tasks | NER / SLU / SER (Table 8 hors ST) | à planifier |

---

## Références

- Papier Pantagruel : [`docs/Pantagruel_2026.pdf`](Pantagruel_2026.pdf) — Table 8 (ST, NER, SLU, SER)
- Protocole utterance et runs : [`docs/protocole_utterance_pantagruel.md`](protocole_utterance_pantagruel.md)
- Exigences pipeline : [`docs/PRD.md`](PRD.md) §4 (hyperparamètres ST)
- Estimation GPU : [`docs/estimation_ressources_fr_en.md`](estimation_ressources_fr_en.md)
