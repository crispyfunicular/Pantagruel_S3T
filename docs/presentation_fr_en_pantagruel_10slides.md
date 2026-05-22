# Présentation synthétique (10 diapos) — ST fr→en avec Pantagruel

**Public :** collègues et encadrants  
**Durée cible :** ~12–15 min (+ questions)  
**Version longue :** [presentation_fr_en_pantagruel.md](presentation_fr_en_pantagruel.md) (détail CE, beam, ablations, annexes)

**Projet :** S3T — réplication Pantagruel 2026, PyTorch/HF + pipeline `scripts/0…6` (pas fairseq)

---

## Slide 1 — Titre et objectif

**Ajouter un modèle ST fr→en à la famille Pantagruel**

- Réplication de l’expérience *Speech Translation* du papier Pantagruel (2026)
- Première direction : **français → anglais** (m-TEDx, ~50 h)
- Livrable : pipeline **reproductible** (S3T) + comparaison **SacreBLEU** / Table 8

*Orateur (~30 s) :* même science que fairseq/`pantagruel_uni`, stack modernisée et artifacts traçables (`runs/<pair>/<run_id>/`).

---

## Slide 2 — Pantagruel et question scientifique

**Pantagruel** = encodeurs SSL unifiés texte + parole (JEPA / data2vec 2.0), checkpoints **Hugging Face**.

**Notre cas :** fine-tune **ST end-to-end** = encodeur parole Pantagruel (pré-entraîné HF) + **décodeur Transformer 6 couches** → anglais tokenisé (SPM). Pas de re-prétrain multimodal dans S3T Temps A.

**Question :** ce couple atteint-il un **BLEU** comparable au protocole LeBenchmark / **Table 8** ?

| Engagement | Détail |
|------------|--------|
| Encodeur | checkpoint HF suffit (pas de prétrain recodé) |
| Éval | **SacreBLEU** figé + signature |
| Écarts | fairseq → S3T **documentés** (greedy vs beam, warmup, etc.) |

---

## Slide 3 — Architecture et stratégie d’entraînement

```text
Audio FR 16 kHz  →  Encodeur Pantagruel (HF)  →  memory
                              ↑ cross-attention
Préfixe anglais  →  Décodeur 6 couches  →  tokens EN (SPM)
```

| Composant | Choix |
|-----------|--------|
| Encodeur | Pantagruel-Base (768) ou Large |
| Décodeur | 6 layers, cross-attention |
| Cible | anglais SPM (vocab 1k–5k) |

**Freeze (~5k–10k updates) :** encodeur **figé** au début → le décodeur (poids aléatoires) apprend sans **catastrophic forgetting** sur l’SSL français ; puis dégel + LR encodeur plus faible.

**Teacher forcing (train) :** le modèle voit les vrais tokens anglais précédents ; il n’**invente** pas encore la phrase (génération = éval / inférence).

---

## Slide 4 — Pourquoi S3T (pas fairseq) et pipeline

| | fairseq (`pantagruel_uni`) | S3T |
|---|---------------------------|-----|
| Rôle | laboratoire article | réplication **ops-friendly** |
| Orchestration | Hydra | `scripts/0…6` + `pipeline.py` |
| Données / éval | prep m-TEDx, generate | manifests TSV + **SacreBLEU** externe |

**On transpose les invariants** (hyperparams, protocole m-TEDx, beam 5 à l’éval) — **on ne copie pas** le code fairseq ni le prétrain multimodal+MLM.

```text
preflight → download → prepare → spm → train → evaluate → infer
```

Stages **implémentés** ; checkpoint utile = `runs/.../checkpoints/best.pt` (meilleur **BLEU dev**).

---

## Slide 5 — Données et tokenisation

**m-TEDx fr→en** (~50 h train) — OpenSLR-100, WAV **16 kHz mono**, filtres durée/texte, manifests `train/valid/test.tsv`, **anti-fuite** train/dev/test.

**SentencePiece** (`3_spm.py`) : entraîné **uniquement** sur l’anglais du train ; vocab cible **1k** (baseline) ou **5k** (ablation).

```bash
python scripts/pipeline.py prepare --langpair fr-en
python scripts/pipeline.py spm --langpair fr-en --vocab-size 1000
```

---

## Slide 6 — Comment le modèle apprend (loss)

**Objectif :** prédire le **prochain token anglais** conditionné par l’audio (+ préfixe connu).

| Étape | Idée |
|-------|------|
| Logits | score par token du vocabulaire |
| Softmax | `exp` + normalisation → probas qui somment à 1 |
| CE | `CE = −log(p_corr)` sur le **bon** token ; faible si confiant (ex. p=0,66 → CE≈0,41) |
| Batch | moyenne des CE ; **label smoothing 0.1** (petit corpus) |

**Mini-exemple (4 tokens fictifs, cible token 0) :** logits `[2, 1, 0.1, −1]` → bon token **66 %** → CE **≈ 0,41** ; si la cible était un token à **2 %** → CE **≈ 3,9**.

**PPL = exp(CE)** : diagnostic dans `train.log` ; **critère de sélection = BLEU dev**, pas la loss seule.

---

## Slide 7 — Trois choses à ne pas confondre

| Outil | Niveau | Quand ? |
|-------|--------|---------|
| **Cross-entropy** | token | entraînement (`4_train.py`) |
| **BLEU (SacreBLEU)** | phrase | choix `best.pt` + Table 8 |
| **Décodage** | phrase | génération à l’éval |

**Greedy vs beam (inférence seulement) :**

| | Greedy | Beam (ex. 5) |
|---|--------|--------------|
| Principe | meilleur token **à chaque pas** | **K** hypothèses de phrase, garde les K meilleures |
| S3T **aujourd’hui** | `5_evaluate.py`, BLEU dev en train | `--beam-size` loggé, **pas encore codé** |
| Papier / Table 8 | — | **beam 5** (cible comparaison) |

```text
TRAIN : teacher forcing + CE
VALIDATION S3T : greedy → BLEU dev → best.pt
COMPARAISON OFFICIELLE : beam 5 (à implémenter)
```

---

## Slide 8 — Évaluation et état du dépôt

```bash
python scripts/pipeline.py evaluate \
  --config configs/fr-en/base.yaml --run-id run_001_fr-en
```

**Sorties :** `dev_predictions.txt`, `test_predictions.txt`, `sacrebleu_*.txt` (avec **signature**), `metrics.json`.

| Fait | À venir / écart documenté |
|------|---------------------------|
| Pipeline 0→6, train HF + décodeur | `configs/fr-en/base.yaml` |
| CE + freeze + greedy + SacreBLEU | **Beam 5**, warmup scheduler, SpecAugment |
| Manifests fr-en | Runs GPU + chiffres Table 8 |

**Promotion d’un run :** gain **BLEU dev** stable (plusieurs seeds), pas loss seule — noter greedy vs beam.

---

## Slide 9 — Exécution et ressources

```bash
source .venv/bin/activate
python scripts/pipeline.py preflight --check-gpu
python scripts/pipeline.py download --langpair fr-en
python scripts/pipeline.py prepare --langpair fr-en
python scripts/pipeline.py spm --langpair fr-en --vocab-size 1000
python scripts/pipeline.py train --config configs/fr-en/base.yaml --run-id run_001_fr-en
python scripts/pipeline.py evaluate --config configs/fr-en/base.yaml --run-id run_001_fr-en
```

| Ressource | Ordre de grandeur |
|-----------|-------------------|
| Disque fr→en | 80–120 GB |
| GPU | 16 GB VRAM recommandé |
| 1 run (~120k updates) | ~25–45 h GPU (calibrer run pilote 2k updates) |

**Jalons go :** preflight OK → manifests sans fuite → train loss ↓ + BLEU dev \> baseline → eval avec signature SacreBLEU.

Détail budget : [estimation_ressources_fr_en.md](estimation_ressources_fr_en.md).

---

## Slide 10 — Message de clôture et livrables

> Chaîne **reproductible** pour un modèle **fr→en** comparable au protocole Pantagruel : données m-TEDx, encodeur HF, décodeur entraîné, **SacreBLEU** traçable — avant extension fr→es / fr→pt.

**Livrables cibles :** `best.pt` + config figée + prédictions + métriques ; plus tard checkpoint / model card HF alignés famille Pantagruel.

**Questions fréquentes :**

- Pourquoi pas fairseq ? → même science, maintenance et traçabilité S3T.
- Pourquoi freeze ? → protéger l’encodeur SSL.
- Beam ? → protocole papier ; greedy = étape intermédiaire actuelle.

**Références détaillées :** [presentation_fr_en_pantagruel.md](presentation_fr_en_pantagruel.md), [PRD.md](PRD.md), [rapport_pantagruel_uni_vers_article.md](rapport_pantagruel_uni_vers_article.md).

---

## Annexe orateur (hors diapo) — Glossaire une ligne

| Terme | Définition |
|-------|------------|
| SSL | prétrain sans labels de tâche |
| ST | audio → texte dans une autre langue |
| SPM | sous-mots SentencePiece |
| SacreBLEU | BLEU standardisé + signature reproductible |
| Greedy / beam | stratégies de **génération** à l’inférence uniquement |
