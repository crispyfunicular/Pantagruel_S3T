# Recommandations et pistes d'amélioration — pipeline S3T

Document de synthèse unique regroupant toutes les pistes d'amélioration du projet S3T, pour les deux variantes prioritaires (ST end-to-end et speechLLM). Mis à jour en juin 2026 après consultation de Gemini.

**Sources fusionnées :** `plan_amelioration_table8.md`, `plan_migration_speechllm.md`, `documentation/speechllm.md §2.6`, recommandations Gemini (17 juin 2026).

---

## Résultats de référence (fr→en, utterance, juin 2026)

| Variante | Run | BLEU test | Statut |
|----------|-----|-----------|--------|
| **ST L-14k v5 SpecAugment** | `run_026` | **26,12** | ok — **meilleur ST local**, dépasse Table 8 L-14k (~24,0) |
| ST L-14k v10 finetune freq | `run_041` | **25,95** | ok — sous run_026 |
| ST L-14k v5 replicate | `run_043` | **24,78** | ok — écart ~1,3 vs run_026 (réplicabilité partielle) |
| ST L-114k v5 SpecAugment | `run_028` | **23,51** | ok — sous papier ~25,2 |
| ST L-114k SPM 5k | `run_033` | best dev **25,53** @ 70k | **en cours** OVH (~71k/80k, fin ~1–2 h) |
| ST L-114k freq | `run_038` | — | **en file** OVH (waiter actif, après run_033) |
| ST L-114k warmup 10k | `run_042` | — | **en file** OVH (après run_038) |
| speechLLM L-14k SpecAugment fort | `run_045` | **13,69** | ok (Modyco, 17 juin) — sous run_023 **14,23** |
| speechLLM L-114k SpecAugment | `run_044` | — | **échec** (Modyco, 17 juin — HF `speech-large-114K` gated) |
| speechLLM B1 L-14k gelé | `run_012` | **15,03** | ok |
| speechLLM B1 L-114k gelé | `run_013` | **15,24** | ok |
| speechLLM B1 L-14k replicate | `run_023` | **14,23** | ok |
| speechLLM B1 L-14k dégelé | `run_015` | **3,65** | ok — **sous** gelé (dégel trop brutal) |

Référence papier (Table 8, fr→en, utterance) : B-1k **17,5 ± 0,4** ; L-14k **24,0 ± 0,4** ; L-114k **25,2 ± 0,4**.

---

## Situation des ressources GPU (17 juin 2026, ~10h)

- **Modyco** : **libre** depuis 09:35 (run_045 terminé ; aucun run en file)
- **OVH** : **run_033** en cours (~71k/80k, GPU 87 %, best dev **25,53** @ 70k) → **run_038** → **run_042** (waiter `run_ovh_wait_spm5k_then_st_114k_v10.sh` actif, poll 5 min)

**Blocage Modyco** : les modèles HF `speech-large-114K` sont **gated** — speechLLM L-114k impossible sur Modyco sans token HF (run_044 échoué le 17 juin). Réserver L-114k à OVH.

---

## Piste A — Stabiliser L-114k à l'échelle *(OVH — file déjà en place)*

### Contexte

Le modèle L-114k (23,51 BLEU) reste **2 pts sous le papier** (~25,2). Hypothèse principale : un encodeur de cette taille nécessite un warmup plus long pour éviter que les gradients du décodeur — initialisé aléatoirement — ne perturbent ses poids pré-entraînés.

> **Gemini :** « Un warmup long (10 000 pas) est crucial pour éviter que le décodeur n'envoie des gradients destructeurs à un encodeur d'une telle capacité. Le régime d'entraînement PyTorch est probablement trop agressif au démarrage par rapport aux schedulers fairseq. »

### Runs planifiés (ne rien changer)

| Run | Config | Changement vs run_028 | Statut |
|-----|--------|-----------------------|--------|
| `run_038` | [`base_utterance_large_114k_v9_specaug_freq.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_114k_v9_specaug_freq.yaml) | + SpecAugment fréquentiel (`mask_freq_prob: 0.05`) | **en file** OVH |
| `run_042` | [`base_utterance_large_114k_v10_warmup10k.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_114k_v10_warmup10k.yaml) | warmup 4k → **10k** | **en file** OVH (après run_038) |

**Action : surveiller run_033** (fin ~1–2 h) **puis confirmer le démarrage automatique de run_038 et run_042.**

```bash
ssh ubuntu@145.239.52.158 'tail -f ~/S3T/logs/run_033_*_spm_train_eval.log'
ssh ubuntu@145.239.52.158 'tail -f ~/S3T/logs/chain_038_042_ovh_wait.log'
```

---

## Piste B — Batch effectif intermédiaire *(Modyco — à implémenter)*

### Contexte

Le batch 64 (`gradient_accumulation: 64`) a provoqué un collapse systématique (run_024 : 0,35 BLEU ; run_025 : 0,31 BLEU). La recette actuelle utilise `gradient_accumulation: 8` (batch effectif 8). Le papier utilise 64–256 séquences.

> **Gemini :** « En environnement PyTorch/HF, une forte accumulation de gradients (64) combinée à un LR de 2e-4 peut provoquer des gradients explosifs. Testez un batch intermédiaire (32) avec LR conservateur (1e-4). Cela permet de lisser le gradient sans heurter les limites de stabilité numérique de l'AMP. »

### Run à créer

> **Note ID** : `run_044` est déjà consommé par un speechLLM L-114k (échec gated HF, 17 juin). Utiliser **`run_046`** pour le ST batch-32.

**Run ID suggéré :** `run_046_transformer_baseline_utterance_large_14k_v11_batch32`

**Fichier à créer :** [`1_Transformer/configs/fr-en/base_utterance_large_14k_v11_batch32.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v11_batch32.yaml)

Seuls changements vs [`base_utterance_large_14k_v5.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v5.yaml) (run_026) :

```yaml
experiment:
  name: "fr-en_baseline_utterance_large_14k_v11_batch32"
  output_dir: "runs/fr-en/run_046_transformer_baseline_utterance_large_14k_v11_batch32"

train:
  gradient_accumulation: 32   # batch effectif 32 (vs 8 run_026, vs 64 run_024 collapse)
  learning_rate_peak: 0.0001  # LR inchangé (2e-4 avec batch 64 = collapse)
  # tout le reste identique à v5 : SpecAugment, warmup 4k, freeze 5k, early_stop 4
```

**Durée estimée :** ~8–10 h GPU (Modyco). Script à créer : `scripts/run_modyco_st_14k_v11_batch32.sh` (dupliquer `scripts/run_modyco_st_14k_v5_replicate.sh`).

### Interprétation attendue

- Si stable et > 26,12 → le batch effectif était le facteur limitant principal.
- Si stable mais ≈ 26,12 → le batch 8 n'était pas le goulot d'étranglement.
- Si collapse → le problème de stabilité est lié à autre chose (LR, AMP) ; tester `5e-5`.

---

## Piste C — SpecAugment fort *(Modyco — run_037 déjà configuré)*

### Contexte

run_026 utilise `mask_time_prob: 0.05`. La piste d'augmentation plus agressive (0.10) n'a jamais été lancée suite à la chaîne abandonnée 036→037.

### Run à lancer

**Run ID :** `run_037_transformer_baseline_utterance_large_14k_v9_specaug_strong`

Config existante : [`1_Transformer/configs/fr-en/base_utterance_large_14k_v9_specaug_strong.yaml`](../1_Transformer/configs/fr-en/base_utterance_large_14k_v9_specaug_strong.yaml)

Seul changement vs run_026 : `mask_time_prob: 0.10` (vs 0.05).

**Action :** script `scripts/run_modyco_st_14k_v9_specaug_strong.sh` existe — lancer sur Modyco (GPU libre). Distinct de **`run_045`** (speechLLM SpecAugment fort, déjà terminé).

---

## Piste D — Cohérence greedy / beam pour `best.pt` *(coût faible)*

### Contexte

`4_train.py` sélectionne `best.pt` sur le BLEU **greedy** en cours d'entraînement ; `5_evaluate.py` rapporte le BLEU **beam 5**. Le vrai meilleur checkpoint pourrait ne pas être le même avec beam.

> **Gemini :** « Plutôt qu'implémenter `eval_beam_during_training: true` (ralentirait les runs de 10h), décodez simplement les 3–5 meilleurs checkpoints sauvegardés par le greedy, puis sélectionnez le vainqueur ex post. C'est un gain marginal, mais gratuit en temps GPU d'entraînement, et cela vous aligne sur la philosophie d'évaluation du papier. »

### Options

**Option A (immédiate, sans modifier le code) :** `best.pt` et `last.pt` sont tous deux disponibles. Comparer en relançant `5_evaluate.py` avec `--checkpoint last.pt` sur les meilleurs runs terminés :

```bash
source .venv/bin/activate
python 1_Transformer/pipeline.py evaluate \
  --config 1_Transformer/configs/fr-en/base_utterance_large_14k_v5.yaml \
  --checkpoint runs/fr-en/run_026_transformer_baseline_utterance_large_14k_v5/checkpoints/last.pt \
  --run-id run_026_eval_lastpt
```

**Option B (dev ~2 h) :** ajouter `train.checkpoint_keep_n_best: 3` dans `4_train.py` pour conserver les 3 meilleurs checkpoints. Utile si l'option A révèle un écart significatif.

**Recommandation : commencer par l'option A** sur run_026 et run_028.

---

## Piste E — Vocabulaire SPM avec gel encodeur prolongé *(impact incertain)*

### Contexte

SPM 5k (run_031 : 24,02) et 8k (run_034 : 22,24) sont **sous** vocab 1k + SpecAugment (run_026 : 26,12). Résultat contre-intuitif vs LeBenchmark (~8k).

> **Gemini :** « Un vocabulaire plus large implique une matrice d'embedding du décodeur beaucoup plus vaste à initialiser. Si vous relancez SPM 5k ou 8k, augmentez drastiquement la durée du gel de l'encodeur (`freeze_encoder_updates > 5k`, peut-être 10k ou 15k). Le décodeur a besoin de plus d'étapes pour structurer un espace de 8000 sous-mots avant que les gradients n'atteignent l'encodeur. »

### Hypothèse à tester

Config : recette run_026 (SpecAugment) + SPM 5k + `freeze_encoder_updates: 15000`.

**Priorité basse** — à lancer seulement après run_046 (batch 32 ST) et run_037 (SpecAugment fort ST), car le résultat run_033 (L-114k SPM 5k) donnera déjà une indication.

---

## Piste F — Réplicabilité et seeds multiples *(priorité scientifique)*

### Contexte

run_026 (26,12) vs run_043 (24,78) : écart ~1,3 BLEU avec seed et config identiques. L'écart suggère une variabilité résiduelle GPU (fp16, CuDNN, machine partagée).

Le PRD §6 recommande ≥ 2 seeds avant de promouvoir une variante. **Non fait** pour les runs récents.

### Action

Lancer run_026 avec `seed: 1` (ou 2) après les run_044 et run_037 sur Modyco. Durée : ~8 h.

Config à créer : `base_utterance_large_14k_v5_seed2.yaml` (seul changement : `seed: 1`, `deterministic: true`).

---

## Piste G — Extensions multilingues fr→es et fr→pt *(Table 8 complète)*

La Table 8 couvre trois directions. Les données sont téléchargeables via `1_download.py`.

| Direction | Données m-TEDx | Cible papier L-14k / L-114k |
|-----------|----------------|------------------------------|
| fr→en | ~50 h | 24,0 / 25,2 — **dépassé** (26,12) |
| fr→es | ~38 h | 25,5 / 25,4 |
| fr→pt | ~25 h | 21,9 / 24,5 |

**Action :** cloner les configs `fr-en/base_utterance_large_14k_v5.yaml` vers `fr-es/` et `fr-pt/`, lancer `2_prepare` pour ces paires, puis entraîner. Effort ~20 min de setup par paire + ~8 h GPU chacune.

---

## Piste H — speechLLM : suite des ablations B1 / B2

### État (juin 2026)

| Run | Encodeur | Gel | BLEU test | Segmentation |
|-----|----------|-----|-----------|--------------|
| `run_012` | L-14k | gelé | **15,03** | utterance |
| `run_013` | L-114k | gelé | **15,24** | utterance |
| `run_023` | L-14k | gelé | **14,23** | utterance (replicate) |
| `run_039` | L-14k | gelé + SpecAugment (0.05) | **13,84** | utterance — sous run_023 |
| `run_045` | L-14k | gelé + SpecAugment fort (0.10) | **13,69** | utterance — sous run_039 et run_023 |
| `run_044` | L-114k | gelé + SpecAugment | — | **échec** Modyco (HF gated) — relancer sur OVH |
| `run_005` | B-1k | **dégelé** | **18,83** | sentence_like |
| `run_015` | L-14k | dégelé | **3,65** | utterance — **sous** gelé |

### Prochaines étapes speechLLM par priorité

| Priorité | Action | Justification |
|----------|--------|---------------|
| **P0** | Relecture qualitative `run_003` (`eval/dev_predictions.txt`) | Comprendre les 7,47 BLEU utterance : boucles ? longueur ? |
| **P1** | Ablation dégel utterance (`run_006`) | run_015 a collapse ; à reprendre avec LR `5e-5`, WD `0.01` |
| **P1** | Confirmer meilleur encodeur (L-14k vs L-114k) | run_012 (15,03) ≈ run_013 (15,24) : pas de gain net ; L-14k suffisant ? |
| **P2** | 2e seed sur meilleur run (run_012 ou run_013) | Exigence PRD §6 avant conclusion forte |
| **P3** | B2 — autre LLM (Llama-3.2-3B, Mistral-7B) | Projecteur à ré-entraîner par couple (encodeur, LLM) |
| **P3** | B2 — SpecAugment speechLLM | run_039 (0.05) vs run_045 (0.10) vs run_023 | SpecAugment **n'aide pas** en B1 gelé (13,84 → 13,69) |

**Règle B2 :** changer une seule chose par rapport au run de référence ; relire `plan_migration_speechllm.md` §B2bis avant de lancer.

---

## Piste I — Tâches downstream non-ST *(Table 8, hors scope actuel)*

La Table 8 couvre aussi NER speech, SLU (MEDIA) et SER (AlloSat). Ces tâches ne font pas partie du pipeline S3T actuel.

| Tâche | Corpus | Métrique | Architecture papier |
|-------|--------|----------|---------------------|
| NER speech | PxCorpus, ETAPE | F1↑ / NEER↓ | encodeur SSL + sonde 3 couches |
| SLU | MEDIA | CER↓ | Transformer seq2seq concepts |
| SER | AlloSat | CCC↑ | encodeur gelé + 5-BiLSTM |

**Action :** nouvelle variante `6_downstream/` ou module dédié. Hors scope ST actuel. Détails : Annexe B.2 du papier Pantagruel.

---

## Checklist de suivi

| ID | Action | Statut |
|----|--------|--------|
| A — OVH file | Surveiller run_033 (~71k/80k) → run_038 → run_042 | **en cours / en file** |
| B — batch 32 | Créer config v11 + script, lancer **run_046** ST (Modyco) | **à faire** — `run_044` ID pris par speechLLM |
| C — SpecAugment fort ST | Lancer **run_037** (config + script existent, Modyco) | **à faire** — distinct de run_045 speechLLM |
| H-run_044 | speechLLM L-114k SpecAugment sur OVH (Modyco gated) | **à relancer sur OVH** |
| H-run_045 | speechLLM L-14k SpecAugment fort | **ok** — 13,69 test |
| D — option A beam | Réévaluer run_026 et run_028 avec `last.pt` | **à faire** |
| E — SPM + gel long | Créer config + lancer (après B et C) | **à planifier** |
| F — seed 2 | Créer config v5_seed2, lancer run_026 seed 2 | **à planifier** |
| G — fr→es / fr→pt | Cloner configs, `2_prepare`, entraîner | **à planifier** |
| H-P0 | Relecture qualitative run_003 predictions | **à faire** |
| H-P1 | Ablation dégel utterance run_006 (LR 5e-5) | **à faire** |
| H-P2 | 2e seed speechLLM run_012 ou run_013 | **à planifier** |
| H-P3 | B2 autre LLM (Llama-3.2-3B) | **à planifier** |
| I | NER / SLU / SER — nouvelle variante | **hors scope actuel** |

---

## Références

| Document | Rôle |
|----------|------|
| [`documentation/PRD.md`](PRD.md) §5–6 | Hyperparamètres cibles LeBenchmark et ablations obligatoires |
| [`documentation/protocole_utterance_pantagruel.md`](protocole_utterance_pantagruel.md) | Tous les runs utterance avec configs, statuts et commandes |
| [`documentation/protocole_evaluation.md`](protocole_evaluation.md) | Protocole SacreBLEU figé |
| [`documentation/estimation_ressources_fr_en.md`](estimation_ressources_fr_en.md) | Budget GPU par run |
| [`1_Transformer/configs/fr-en/`](../1_Transformer/configs/fr-en/) | Configs YAML par run |
| [`rapport.md §5`](../rapport.md#5-résultats) | Tableaux complets tous pipelines |
| [`runs/experiments_tracking.csv`](../runs/experiments_tracking.csv) | Agrégat chiffré |
