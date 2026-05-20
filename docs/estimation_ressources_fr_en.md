# Estimation des ressources — pipeline ST fr→en (machine distante)

Document de planification pour dimensionner **disque**, **GPU (VRAM + temps)**, **RAM CPU** et **réseau** avant de lancer les runs S3T / Pantagruel.

**Hypothèses de référence** (alignées PRD + `README_experiments.md`) :

| Paramètre | Valeur |
|-----------|--------|
| Corpus | m-TEDx **fr→en**, ~**50 h** train (+ valid/test) |
| Audio | 16 kHz, mono, PCM_16, segments 1–30 s |
| Encodeur | Pantagruel-**Base** (~93 M paramètres, sortie 768) |
| Décodeur | Transformer 6 couches |
| Entraînement cible | jusqu’à **120 000 updates**, batch **8** × accum **8** (effectif 64) |
| Freeze encodeur | 5 000 updates puis fine-tune joint |
| Précision | AMP **bf16** / fp16 |

Les ordres de grandeur ci-dessous sont des **fourchettes** ; un run pilote de 500–2 000 updates permet de calibrer sur *votre* GPU.

---

## 1. Synthèse rapide (à mettre dans une slide)

| Ressource | Minimum viable (fr→en) | Recommandé | Confort / multi-runs |
|-----------|----------------------|------------|----------------------|
| **Disque libre** | 50 GB | **80–120 GB** | **200 GB** (PRD, 3 paires + marge) |
| **VRAM GPU** | 8 GB (serré) | **16 GB** | 24 GB+ |
| **Temps GPU** (1 run complet) | — | **25–45 h** | +15–25 h par série d’ablations |
| **RAM CPU** | 16 GB | 32 GB | 64 GB (prepare parallèle) |
| **Réseau** | ~12 GB download | + cache HF ~1–3 GB | — |

**Message pour les encadrants :** un **premier modèle fr→en crédible** ≈ **1 GPU 16 GB pendant ~1–2 jours** + **~100 GB disque** si on garde raw, processed, checkpoints et eval.

---

## 2. Stockage disque (détail)

### 2.1 Par étape du pipeline

| Étape | Contenu | Estimation fr→en | Supprimable après ? |
|-------|---------|------------------|---------------------|
| Download | `mtedx_fr-en.tgz` | **~9–10 GB** | Oui, si extract OK |
| Raw extrait | FLAC + métadonnées | **~8–12 GB** | Oui, après `prepare` |
| Processed WAV | segments 16 kHz PCM16 | **~6–8 GB** | Non (entrée train) |
| Manifests + SPM | TSV, `.model`, `.vocab` | **< 0,1 GB** | Non |
| Cache Hugging Face | poids Pantagruel-Base | **~0,4–1,5 GB** | Regénérable |
| Environnement `.venv` | torch, SB, transformers… | **~5–10 GB** | Non |
| 1 run `train` + `eval` | checkpoints, logs, preds | **~1–3 GB** | Non (traçabilité) |
| 5 ablations + 2 seeds | 5–10 runs | **~5–15 GB** | Archiver ailleurs |

### 2.2 Formule audio (processed)

```
taille_WAV (GB) ≈ heures_audio × 3600 × 32000 / 1e9
                ≈ heures_audio × 0,115
```

Exemples :

| Heures (PCM16 mono 16 kHz) | Taille WAV |
|----------------------------|------------|
| 50 h (train seul) | **~5,8 GB** |
| 55 h (train+valid+test) | **~6,3 GB** |
| Pic prepare (raw FLAC + WAV) | **×1,5 à 2** temporaire |

### 2.3 Scénarios disque

| Scénario | Contenu conservé | Total estimé |
|----------|------------------|--------------|
| **A — Minimal fr→en** | WAV + manifests + 1 run + venv + HF | **~35–50 GB** |
| **B — Standard fr→en** | + raw/archive + 3 runs + logs | **~70–100 GB** |
| **C — PRD complet (3 paires)** | fr-en + fr-pt + fr-es + ablations | **~150–220 GB** |

Le preflight exige **≥ 200 GB** : cohérent avec le scénario C et une marge pour checkpoints intermédiaires.

---

## 3. GPU — mémoire (VRAM)

### 3.1 Ordre de grandeur du modèle

| Composant | Paramètres (ordre de grandeur) | Poids bf16 |
|-----------|----------------------------------|------------|
| Encodeur Pantagruel-Base | ~93 M | ~190 MB |
| Décodeur 6L (768 dim) | ~25–45 M | ~50–90 MB |
| **Total poids** | ~120–140 M | **~0,3–0,4 GB** |

La VRAM est surtout dominée par :

- **activations** (longueur audio dans le batch, souvent pire que la longueur texte) ;
- **états optimiseur AdamW** (≈ 2–3× les paramètres **entraînables** en fp32) ;
- **éval périodique** (decode sur valid, greedy ou beam).

### 3.2 Facteur critique : longueur des segments

Le collateur padde les waveforms à la **longueur max du batch**. Avec `--max-duration 30.0` :

- 30 s × 16 kHz = **480 000** échantillons par utterance ;
- batch 8 au pire cas → risque **OOM** sur GPU 8 GB.

**Leviers pour réduire la VRAM :**

| Levier | Effet |
|--------|--------|
| `batch_size: 4` ou `2` | ↓ VRAM linéaire |
| `gradient_accumulation: 16–32` | garde le batch effectif |
| `max_duration: 20` ou `15` | ↓ fortement le pire cas |
| `amp_dtype: bf16` | ↓ ~30–50 % vs fp32 |
| `freeze_encoder_updates` | ↓ gradients encodeur au début |
| `max_eval_batches: 20` (défaut code) | ↓ pic VRAM à l’éval |

### 3.3 Matrice VRAM indicative

| GPU | Config typique | Verdict |
|-----|----------------|---------|
| **8 GB** (T4, RTX 3060) | batch 2–4, accum 16–32, max_dur 20 s, bf16 | **Pilote OK**, run long possible mais lent |
| **16 GB** (V100 16G, RTX 4080, A10) | batch 8, accum 8, max_dur 30 s, bf16 | **Recommandé** (config README_experiments) |
| **24 GB+** (A100 40G, RTX 4090) | batch 8–16, beam eval confortable | **Confort** + marge ablations parallèles |

**Preflight** : VRAM ≥ 8 GB en *warning* ; viser **16 GB** pour l’expérience de référence.

---

## 4. GPU — temps de calcul

### 4.1 Nombre d’updates vs passages sur les données

Estimation du nombre de segments train (TEDx, segments ~5–8 s en moyenne) :

```
N_segments ≈ heures_train × 3600 / durée_moyenne_s
           ≈ 50 × 3600 / 6 ≈ 30 000
```

Avec `batch_size = 8` (sans accum pour l’intuition) :

```
steps_par_epoch ≈ N_segments / 8 ≈ 3 750
```

Pour **120 000 updates** :

```
passages_sur_train ≈ 120 000 / 3 750 ≈ 32 epochs (ordre de grandeur)
```

### 4.2 Temps par update (fourchettes)

| Phase | Secondes / update (indicatif) | GPU type |
|-------|-----------------------------|----------|
| Encodeur **gelé** (début) | 0,2 – 0,6 s | V100 / RTX 3090 |
| Encodeur **dégelé** | 0,5 – 1,5 s | V100 / RTX 3090 |
| Encodeur dégelé | 0,3 – 0,8 s | A100 |

### 4.3 Durée d’un run complet (120k updates)

```
T_train ≈ N_updates × t_update_moyen
```

| Hypothèse `t_update_moyen` | Temps GPU train seul |
|--------------------------|----------------------|
| 0,5 s | **~17 h** |
| 0,8 s | **~27 h** |
| 1,2 s | **~40 h** |

**+ Évaluations** (`eval_every_updates: 1000`, decode partiel valid) :

- ~120 évals × 2–5 min → **+4 à 10 h GPU**

| **Total 1 run fr→en (référence)** | **~25 – 45 h GPU** |
|-----------------------------------|---------------------|

### 4.4 Runs plus courts (calibration)

| Objectif | `max_updates` | Temps GPU estimé |
|----------|---------------|------------------|
| Smoke test (pipeline OK) | 500 | **0,5 – 2 h** |
| Baseline exploitable | 10 000 – 20 000 | **4 – 10 h** |
| Comparaison Table 8 (cible PRD) | 80 000 – 120 000 | **20 – 45 h** |

Early stopping (`patience` 8 sur BLEU dev) peut **réduire** la facture de 20–40 % si le dev plafonne tôt.

### 4.5 Plan d’expériences (ablations PRD §6)

| Série | Runs | Updates / run (hyp.) | GPU total (fourchette) |
|-------|------|----------------------|-------------------------|
| Baseline + 4 ablations | 5 | 60k–120k | **~80 – 200 h** |
| + 2 seeds sur meilleure config | 2 | idem | **+30 – 80 h** |
| **Total recherche fr→en** | 7–10 | — | **~120 – 280 h GPU** |

En pratique : commencer par **1 run pilote** (10k updates) avant de lancer toute la grille.

---

## 5. CPU et RAM (hors GPU)

| Étape | CPU | RAM | Durée indicative |
|-------|-----|-----|------------------|
| `prepare` fr→en (~55 h audio) | 4–8 cœurs utiles | 8–16 GB | **2–8 h** (I/O + resample) |
| `spm` | 1 cœur | < 4 GB | **< 5 min** |
| `download` | réseau | — | **30 min – 2 h** (débit) |
| `preflight` | léger | — | **< 1 min** |

Prepare peut tourner **sans GPU** ; seuls train/eval/infer nécessitent CUDA.

---

## 6. Réseau

| Flux | Volume |
|------|--------|
| OpenSLR `mtedx_fr-en.tgz` | **~9–10 GB** |
| Hugging Face Pantagruel-Base (+ deps tokenizer) | **~0,5–2 GB** |
| pip install torch/CUDA (si venv vide) | **~2–5 GB** |
| **Total premier setup** | **~15–20 GB** |

---

## 7. Estimation « coût cloud » (indicatif)

Prix publics variables selon hébergeur/région — **fourchettes 2025–2026** pour budgétiser :

| Type GPU | €/h (indicatif) | 30 h (1 run) | 150 h (grille ablations) |
|----------|-----------------|--------------|---------------------------|
| T4 / L4 16 GB | 0,3 – 0,8 | 10 – 25 € | 45 – 120 € |
| V100 16 GB | 0,6 – 1,5 | 18 – 45 € | 90 – 225 € |
| A100 40/80 GB | 1,5 – 3,5 | 45 – 105 € | 225 – 525 € |

**Machine labo / perso** : compter surtout **électricité** + amortissement GPU ; le poste dominant reste le **temps humain** et le **disque**.

---

## 8. Checklist machine distante

### Minimum pour démarrer fr→en

- [ ] Linux, Python ≥ 3.10, `nvidia-smi` OK  
- [ ] **≥ 50 GB** disque libre (≥ 100 GB conseillé)  
- [ ] GPU **≥ 8 GB** VRAM (16 GB recommandé)  
- [ ] Accès Internet OpenSLR + Hugging Face  
- [ ] `python scripts/pipeline.py preflight --check-gpu --min-disk-gb 100`

### Avant le long run

- [ ] Run pilote : `max_updates: 2000`, mesurer `s/update` dans `train.log`  
- [ ] Extrapoler : `T_estimé = s/update × max_updates`  
- [ ] Vérifier pic VRAM avec `nvidia-smi dmon` pendant 100 updates  
- [ ] Ajuster `batch_size` / `max_duration` si OOM  

### Commande pilote (extrait config)

```yaml
train:
  max_updates: 2000
  eval_every_updates: 500
  batch_size: 4          # si 8 GB VRAM
  gradient_accumulation: 16
  max_eval_batches: 10
```

---

## 9. Formule de calcul personnalisée

Copier dans un tableur :

```
# Entrées
heures_train = 50
duree_moy_s = 6
batch_size = 8
grad_accum = 8
max_updates = 120000
sec_per_update = 0.8        # mesuré en pilote
eval_every = 1000
sec_per_eval = 180          # mesuré en pilote

# Dérivés
n_segments = heures_train * 3600 / duree_moy_s
steps_per_epoch = n_segments / batch_size
epochs = max_updates / steps_per_epoch

disk_gb = 10 + 6 + 8 + 2 + (n_runs * 2)   # archive + wav + venv/HF + runs
gpu_hours = max_updates * sec_per_update / 3600 \
          + (max_updates / eval_every) * sec_per_eval / 3600
```

---

## 10. Lien avec la présentation

| Slide présentation | Ressource à annoncer |
|--------------------|----------------------|
| Données fr→en | ~50 h, ~10 GB download, ~7 GB WAV |
| Entraînement | 120k updates, batch effectif 64 |
| Go/no-go | 16 GB GPU, 100+ GB disque pour série complète |
| Clôture | **~1–2 jours GPU** pour baseline ; **~1–2 semaines GPU** pour grille complète |

---

## Références projet

- [PRD.md](../PRD.md) §1.3 (volumes m-TEDx), §5 (hyperparamètres), §7 (risques)
- [README.md](../README.md) prérequis 200 GB
- [README_experiments.md](../README_experiments.md) template config
- [presentation_fr_en_pantagruel.md](presentation_fr_en_pantagruel.md)

*Mettre à jour ce document après le premier run pilote avec les valeurs réelles `sec/update`, VRAM max et `bleu_dev`.*
