# Audit du dépôt `../fairseq` (focus `origin/main`)

Date d’analyse: 2026-05-18  
Référentiel analysé: `/home/morgane/git/GETALP/fairseq`  
Remote principal: `origin = git@github.com:formiel/fairseq.git`

## 1) Résumé exécutable

- La branche `origin/main` contient **2325 commits** (dont **2303 non-merge** et **22 merges**), de **2017-09-14** à **2024-01-22**.
- Le dernier commit de `origin/main` est un merge de synchro upstream:
  - `d4360598` (2024-01-22) `Merge remote-tracking branch 'upstream/main'` (auteur: Hang Le).
- Point critique pour la reproductibilité:
  - Dans ce clone, `origin/main` est **en retard** par rapport à `origin/master`.
  - `git rev-list --left-right --count origin/master...origin/main` retourne `132 0`, donc **132 commits présents sur `origin/master` et absents de `origin/main`**.
  - Cet écart inclut des ajouts substantiels (siamese/dual-decoder/speech-text), très susceptibles d’expliquer des divergences de résultats.

## 2) État du dépôt inspecté

- Branche locale courante: `master` (suivi: `origin/master`).
- Arborescence racine standard fairseq observée (`fairseq/`, `examples/`, `fairseq_cli/`, `tests/`, etc.).
- Fichier non versionné détecté localement: `Pantagruel_2026.pdf` (non commité).

## 3) Historique global de `origin/main`

## Fenêtre temporelle

- Premier commit (racine): `e734b0fa` (2017-09-14) `Initial commit` (Sergey Edunov)
- Dernier commit: `d4360598` (2024-01-22) merge upstream (Hang Le)

## Volume par année (commits)

- 2017: 103
- 2018: 367
- 2019: 590
- 2020: 631
- 2021: 397
- 2022: 191
- 2023: 44
- 2024: 2

## Contributeurs principaux (top observé)

- Myle Ott: 722 commits
- alexeib: 139 commits
- Changhan Wang: 50 commits
- Alexei Baevski: 38 commits
- Ning Dong: 36 commits
- Yuqing Tang: 36 commits
- Alex Xiao: 34 commits

Remarque: l’historique est majoritairement celui de fairseq OSS/upstream.

## 4) Zones du code les plus touchées sur `origin/main`

## Fichiers les plus fréquemment modifiés (extraits)

- `fairseq/trainer.py` (201 touches)
- `fairseq/options.py` (165)
- `fairseq/utils.py` (162)
- `fairseq/models/transformer.py` (160)
- `fairseq/sequence_generator.py` (136)
- `train.py` (130)
- `fairseq/checkpoint_utils.py` (105)
- `README.md` (103)
- `tests/test_binaries.py` (97)

## Répartition par dossiers (top-level)

- `fairseq/`: 5594 touches
- `examples/`: 2001
- `tests/`: 535
- `fairseq_cli/`: 221
- puis scripts/fichiers de support (`scripts/`, `docs/`, `setup.py`, etc.)

## Thèmes fréquents dans les messages de commit (approximation par mots-clés)

- `LM`: 107
- `checkpoint`: 97
- `transformer`: 96
- `bert`: 54
- `translation`: 53
- `speech`: 48
- `wav2vec`: 41
- `multilingual`: 37
- `fp16`: 33
- `hydra`: 30
- `roberta`: 30

Conclusion: `origin/main` reflète une base fairseq généraliste (NLP + speech), pas un historique spécialisé unique.

## 5) Commits récents notables sur `origin/main` (2023-2024)

- `d4360598` (2024-01-22) Merge remote-tracking branch 'upstream/main'
- `fad2c4d1` (2024-01-08) Update README.md (#5407)
- `da8fb630` (2023-10-10) Change Meta AI to FAIR (#5346)
- `c7c478b9` (2023-10-09) fix iterator when loading from checkpoint (#5344)
- `b5d89cdd` (2023-09-07) Update align_and_segment.py (#5317)
- `4db26494` (2023-08-18) hubert/wav2vec2 positional conv option
- `31fba013` (2023-06-23) SinusoidalPositionalEmbedding buffer non-persistant
- Série de commits MMS (ASR/TTS/LID/docs) en 2023-05/06

## Commits `origin/main` signés Hang Le

- `d4360598` (2024-01-22) merge upstream/main
- `c0d64a1e` (2023-08-19) merge upstream/main
- `622d12f0` (2023-06-26) merge upstream/main
- `8e9b99fd` (2023-06-16) merge upstream/main
- `3f2f45c6` (2023-04-03) update gitignore
- `8e370d8d` (2022-07-11) update .gitignore

Lecture: sur `origin/main`, les contributions spécifiques de Hang Le sont quasi uniquement des merges/synchronisations et des ajustements `.gitignore`.

## 6) Écart déterminant: `origin/main` vs `origin/master`

Même si la demande cible `origin/main`, ce point est central pour expliquer une divergence expérimentale.

## Données de divergence

- `origin/master` contient **132 commits** absents de `origin/main`.
- `origin/main` n’a **aucun commit** absent de `origin/master` (dans ce clone).
- Diff HEAD `origin/main..origin/master`:
  - **59 fichiers modifiés**
  - **10193 insertions**, **530 suppressions**

## Nature des changements absents de `origin/main`

Les commits absents de `origin/main` (mais présents sur `origin/master`) incluent notamment:

- ajout et évolution de modèles/critères pour:
  - dual-decoder
  - siamese speech-text
  - dual-beam decoding
  - OT/Wasserstein losses
- extensions dans:
  - `examples/speech_text_siamese/...`
  - `examples/speech_text_joint_to_text/...`
  - `fairseq/models/speech_to_text/s2t_dd_transformer.py`
  - `fairseq/models/transformer_dd.py`
  - `fairseq/sequence_generator_dd.py`
  - `fairseq/scoring/wer_bleu.py`

Exemples de commits présents sur `origin/master` mais pas sur `origin/main`:

- `7be2260a` merge code dual decoder transformer with master
- `bc76d930` Merge siamese_pt code with master
- `a32ceb8c` Add single decoder arch
- `6211402b` First commit for dual beam search
- `e925d707` Add clonefuse
- `09306a17` Fix decoding for siamese_pt when merging
- `716c719d`/`dd08d862`/`8368f7e2`/`331b978a`/`6fe14840` (2026-05-01): updates README + visualisations liées à Pantagruel

## 7) Hypothèse directe pour votre divergence de résultats

Si vous comparez vos expériences à "ses résultats" et que cette personne a travaillé sur la ligne `master` (ou sur un checkout contenant les commits listés ci-dessus), alors:

- exécuter vos expériences depuis `origin/main` **ne reproduira pas** la même base de code;
- les différences portent précisément sur les composants speech-text/siamese/dual-decoder/scoring, donc sur des zones à fort impact métrique.

## 8) Recommandation pratique immédiate

Pour une comparaison juste:

1. figer le commit exact de référence de l’autre personne (`git rev-parse HEAD` chez elle/lui),
2. checkout ce commit dans votre environnement,
3. comparer ensuite uniquement:
   - données/manifests
   - paramètres CLI/hydra
   - seed et version CUDA/PyTorch
   - checkpoints utilisés

---

## 9) Inventaire complet des branches distantes (hors `origin/main`)

Date d’analyse complémentaire : 2026-05-19

### 9.1 Vue d’ensemble

| Branche | Commits totaux | Commits absents de `origin/main` | Auteurs principaux | Période |
|---------|:--------------:|:---------------------------------:|-------------------|---------|
| `origin/master` | 2457 | 132 | Hang Le (+ upstream) | 2017–2026-05 |
| `origin/pantagruel_uni` | 3147 | **822** | Hang Le | 2024-02–2026-05 |
| `origin/naive_beamsearch` | 904 | 904 | Myle Ott, équipe FB | 2019 (branche morte) |
| `origin/bi_trans_lm` | 763 | 763 | (fairseq upstream ancien) | 2019 (branche morte) |
| `origin/siamese_s2t` | 2070 | 77 | Hang Le | 2021-03–2023-05 |
| `origin/adapters` | 1929 | 19 | Hang Le | 2021-01–2021-06 |
| `origin/LeBenchmark` | 1948 | 16 | Hang Le | 2021-06–2021-09 |
| `origin/blockbert` | 249 | 15 | Jiezhong Qiu | 2019 (externe, mort) |
| `origin/seq_task` | 721 | 8 | Myle Ott | 2019 (mort) |
| `origin/xlmr_benchmark` | 986 | 1 | Naman Goyal | 2019 (mort) |
| `origin/classic_seqlevel` | 47 | 2 | Myle Ott, Edunov | 2018 (mort) |

---

### 9.2 `origin/pantagruel_uni` — branche centrale pour les expériences actuelles

C’est de loin la branche la plus active et la plus pertinente : **822 commits uniques** par rapport à `origin/main`, couvrant la période **2024-02 à 2026-05** (derniers commits : 2026-05-13).

#### Fichiers et dossiers clés absents de `origin/main`

- `examples/pantagruel/` — nouveau répertoire dédié Pantagruel :
  - `configs/speech/finetuning/` contient les configs Hydra pour les expériences de **speech translation sur mTEDx** :
    - `base_mtedx_fr2en_wdecay0.1-ls.yaml`
    - `base_mtedx_fr2es.yaml`
    - `base_mtedx_fr2pt.yaml`
    - `large_mtedx_fr2en_lr3e-5.yaml`
    - `large_mtedx_fr2es.yaml`
    - `large_mtedx_fr2pt.yaml`
    - `base_commonvoice.yaml`, `base_10h.yaml`, etc.
- `examples/data2vec/models/data2vec2_st.py` — modèle `data2vec2_st`
- `examples/data2vec/tasks/s2t_finetuning.py` — task `data2vec2_st_finetuning`
- `examples/data2vec/models/modalities/audio.py`, `text.py` — modalités unimodales
- `examples/data2vec/tasks/multimodal.py` — task multimodale
- `examples/speech_to_text/prep_mtedx_data.py` — script de préparation des données mTEDx

Diff global `origin/main..origin/pantagruel_uni` : **50 fichiers modifiés**, **+2742 / −170 lignes**.

#### Architecture de fine-tuning (extrait des configs mTEDx)

Les configs fr→en/es/pt partagent toutes la même structure de modèle `data2vec2_st` :

```yaml
task:
  _name: data2vec2_st_finetuning
  normalize: true

criterion:
  _name: label_smoothed_cross_entropy
  label_smoothing: 0.1

lr_scheduler:
  _name: tri_stage
  phase_ratio: [0.1, 0.4, 0.5]
  final_lr_scale: 0.05

model:
  _name: data2vec2_st
  w2v_path: ???          # chemin vers le checkpoint Pantagruel pré-entraîné
  autoregressive: true
  feature_grad_mult: 0.1
  layerdrop: 0.05
  decoder_embed_dim: 512
  decoder_ffn_embed_dim: 2048
  decoder_layers: 6
  decoder_attention_heads: 8
```

Différences entre les configs par paire de langues :

| Config | `max_update` | `weight_decay` | LR | Variante |
|--------|:-----------:|:--------------:|-----|---------|
| `base_mtedx_fr2en_wdecay0.1-ls` | 750 000 | 0.01 | 1e-4 | base |
| `base_mtedx_fr2es` | 550 000 | 0.01 | 1e-4 | base |
| `base_mtedx_fr2pt` | 550 000 | 0.01 | 1e-4 | base |
| `large_mtedx_fr2en_lr3e-5` | (large) | — | 3e-5 | large |

Le serveur de calcul de référence est Jean-Zay / Adastra (chemin `/lustre/fsn1/projects/rech/oou/ucy22cr/pantagruel/`).

#### Timeline des commits notables sur `origin/pantagruel_uni`

| Période | Thème principal |
|---------|----------------|
| 2024-02 | Implémentation pytorch2 SDPA dans data2vec2, optimisation mémoire |
| 2024-03–04 | Pré-entraînement speech-text joint (data2vec2 multimodal), token_type_embeddings |
| 2024-05 | Préparation manifests LeBenchmark 14K, nettoyage données audio |
| 2026-01–03 | Configs pour fine-tuning unimodal speech (ASR + ST), MLM loss, adastra |
| 2026-03 | `add mlm loss to pantagruel multimodal`, correction bugs finetuning |
| 2026-05 | Nettoyage `.gitignore`, suppression configs sensibles |

---

### 9.3 `origin/master` — branche d’intégration (complément section 6)

Rappel : 132 commits uniques vs `origin/main`. Les commits de Hang Le se répartissent en deux périodes :

**2021 — développement dual-decoder** (commits fonctionnels) :
- `2021-03-16` : `Initial commit ind decoders for dd_transformer`
- `2021-04-14` : `First commit for dual beam search`
- `2021-04-27` : `Add wait-k in training for dual-decoder transf.`
- `2021-05-06` : `Add wer_bleu scorer`
- Série de corrections du dual-beam decoding jusqu’en 2021-05-25.

**2023 — intégration siamese + nettoyage** :
- `2023-06-12` : `Merge siamese_pt code with master`
- `2023-06-13` : `merge code dual decoder transformer with master`
- `2023-06-14` : READMEs (siamese, adapters, URLs)

**2026-05-01** : 5 commits de mise à jour du README + visualisations t-SNE des représentations encodeurs (auteur `formiel`).

---

### 9.4 `origin/siamese_s2t` — développement siamese speech-text (77 commits uniques)

Branche de **travail en cours** pour le pré-entraînement Siamese CTC+OT (Ch. 5 de la thèse). Deux phases :

- **2021-11 à 2023-02** : développement progressif du modèle `siamese_st2t` (CTC, OT loss, RoBERTa text encoder, mBART decoder, adversarial regularizer). Commits fonctionnels.
- **2023-05** : nettoyage et ajout de figures/README (12 commits, purement documentaires).

Le code correspondant est aujourd’hui intégré dans `origin/master` (via merge de juin 2023).

---

### 9.5 `origin/adapters` — adapters multilingues (19 commits uniques, 2021)

Développement des adapters pour S2T (Ch. 4 de la thèse) :
- adapters parallèles (`xattn` + encoder)
- chargement depuis mBART (`mbart50`) pour initialiser le décodeur S2T
- identifiants de langue pour la ST multilingue

Dernier commit : `2021-06-04`. Branche archivée, code intégré dans `origin/master`.

---

### 9.6 `origin/LeBenchmark` — features wav2vec2 pour mTEDx (16 commits uniques, 2021)

Scripts de prétraitement pour extraire des features wav2vec2 fines-tunées depuis LeBenchmark et les utiliser pour la ST sur **mTEDx** :
- `prep_ft_w2v2.py` : préparation features w2v2 pour ST
- `prep_mtedx_data.py` : prétraitement src/tgt par paire de langues
- option `--do-lower`, suppression SpecAugment si features pré-extraites

Dernier commit : `2021-09-17`. Branche archivée.

---

### 9.7 Branches externes / mortes (à ignorer pour les expériences)

| Branche | Raison d’ignorer |
|---------|-----------------|
| `origin/naive_beamsearch` | Branche 2019 fairseq amont, divergée avant tous les travaux de Hang Le |
| `origin/bi_trans_lm` | Idem, 2019, pas de contribution GETALP |
| `origin/blockbert` | Contribution externe (Jiezhong Qiu, 2019), sans rapport |
| `origin/seq_task` | Structured prediction upstream (Myle Ott, 2019) |
| `origin/classic_seqlevel` | Classical seq-level losses (Edunov, 2018) |
| `origin/xlmr_benchmark` | XLM-R benchmark (Goyal, 2019) |

---

## 10) Synthèse : quelle branche utiliser pour reproduire les expériences

| Objectif | Branche recommandée | Commit de référence |
|---------|--------------------|--------------------|
| Reproduire ST fr→en/es/pt avec Pantagruel (data2vec2_st) | **`origin/pantagruel_uni`** | dernier HEAD ou commit avant 2026-05-13 (nettoyage) |
| Reproduire dual-decoder Transformer (Ch. 3 thèse) | `origin/master` | `7be2260a` (merge DD, 2023-06-13) |
| Reproduire CTC + OT siamese (Ch. 5 thèse) | `origin/master` | `bc76d930` (merge siamese_pt, 2023-06-12) |
| Reproduire adapters (Ch. 4 thèse) | `origin/master` ou `origin/adapters` | dernier commit adapters |

**Point critique** : pour les expériences fr→{en,es,pt} sur mTEDx avec Pantagruel, la branche `origin/pantagruel_uni` est indispensable — elle contient les configs Hydra, le modèle `data2vec2_st`, la task `data2vec2_st_finetuning` et les scripts de préparation mTEDx, tous absents de `origin/main` et de `origin/master`.

---

## Annexe A — Commandes utilisées pour cet audit

```bash
git -C /home/morgane/git/GETALP/fairseq status --short --branch
git -C /home/morgane/git/GETALP/fairseq branch -r
git -C /home/morgane/git/GETALP/fairseq rev-list --count origin/main
git -C /home/morgane/git/GETALP/fairseq rev-list --count --no-merges origin/main
git -C /home/morgane/git/GETALP/fairseq rev-list --count --merges origin/main
git -C /home/morgane/git/GETALP/fairseq shortlog -sne origin/main
git -C /home/morgane/git/GETALP/fairseq log --date=short --pretty='%ad' origin/main | cut -d- -f1 | sort | uniq -c
git -C /home/morgane/git/GETALP/fairseq log --pretty=format: --name-only origin/main | awk 'NF' | sort | uniq -c | sort -nr
git -C /home/morgane/git/GETALP/fairseq rev-list --left-right --count origin/master...origin/main
git -C /home/morgane/git/GETALP/fairseq diff --shortstat origin/main..origin/master
git -C /home/morgane/git/GETALP/fairseq diff --name-only origin/main..origin/master
git -C /home/morgane/git/GETALP/fairseq log --date=short --pretty='%h | %ad | %an | %s' origin/main..origin/master
```

## Annexe B — Commandes ajoutées pour l’audit des branches non-`main` (2026-05-19)

```bash
# Inventaire et comptage des commits uniques par branche
for branch in origin/master origin/LeBenchmark origin/adapters origin/bi_trans_lm \
              origin/blockbert origin/classic_seqlevel origin/naive_beamsearch \
              origin/pantagruel_uni origin/seq_task origin/siamese_s2t origin/xlmr_benchmark; do
  count=$(git rev-list --count "$branch")
  unique=$(git rev-list --count "origin/main".."$branch")
  echo "$branch : total=$count unique_vs_main=$unique"
done

# Log des commits uniques par branche
git log --date=short --pretty='%h | %ad | %an | %s' origin/main..origin/pantagruel_uni
git log --date=short --pretty='%h | %ad | %an | %s' origin/main..origin/siamese_s2t
git log --date=short --pretty='%h | %ad | %an | %s' origin/main..origin/adapters
git log --date=short --pretty='%h | %ad | %an | %s' origin/main..origin/LeBenchmark

# Diff pantagruel_uni vs main
git diff --name-only origin/main..origin/pantagruel_uni
git diff --shortstat origin/main..origin/pantagruel_uni

# Lire un fichier sur une branche distante sans checkout
git show origin/pantagruel_uni:examples/pantagruel/configs/speech/finetuning/base_mtedx_fr2en_wdecay0.1-ls.yaml
git show origin/pantagruel_uni:examples/pantagruel/
```
