# Product Requirement Document (PRD) & Plan de Projet
## Réplication du Système de Traduction de la Parole (ST) End-to-End (Pantagruel)

---

## 1. Vue d'Ensemble & Objectifs

### 1.1 Contexte
Ce projet vise à répliquer les expériences de **Traduction de la Parole (Speech-to-Text Translation - ST)** décrites dans l'article scientifique *Pantagruel: Unified Self-Supervised Encoders for French Text and Speech* (2026) (`Pantagruel_2026.pdf`). L'étude évalue la capacité des modèles auto-superveillés (SSL) français à projeter des représentations acoustiques continues directement vers du texte cible multilingue de manière *end-to-end* (E2E).

### 1.2 Objectif Principal
Construire, entraîner et évaluer un système de traduction de la parole de bout en bout en connectant un décodeur Transformer à 6 couches sur l'encodeur SSL pré-entraîné **Pantagruel** (ou ses équivalents LeBenchmark 2.0).

### 1.3 Périmètre d'Évaluation
Le système sera évalué sur les sous-ensembles de source française (`fr`) du corpus **multilingual TEDx (m-TEDx)** pour trois directions de traduction :
* **Français -> Anglais (fr-en)** : environ 50 heures de données d'entraînement.
* **Français -> Portugais (fr-pt)** : environ 38 heures de données d'entraînement.
* **Français -> Espagnol (fr-es)** : environ 25 heures de données d'entraînement.

La métrique de validation finale est le **score BLEU** calculé via *SacreBLEU*, à comparer directement avec les résultats de la Table 8 de l'article Pantagruel.

---

## 2. Architecture Technique du Modèle

Le modèle suit une architecture de type Séquence-à-Séquence (Seq2Seq) *end-to-end* :

> [ Audio (16 kHz) ] -> [ Encodeur SSL Pantagruel (Figé/Ajusté) ]
>                                            | (Représentations Continues)
>                                            v
> [ Texte Cible (Précédent) ] -> [ Décodeur Transformer (6 Couches) ] -> [ Token Cible Suivant ]

### 2.1 L'Encodeur Acoustique (SSL)
* **Modèle de base :** `Pantagruel-Base` (768 dimensions de sortie) ou `Pantagruel-Large` (1024 dimensions).
* **Entrée :** Formes d'onde brutes (raw waveforms) échantillonnées à 16 kHz, mono.
* **Comportement :** Extrait des caractéristiques contextuelles denses dans un espace latent continu (JEPA/data2vec 2.0).

### 2.2 Le Décodeur Textuel
* **Type :** Décodeur Transformer standard autoregressif.
* **Configuration :** 6 couches, 4 à 8 têtes d'attention (selon la dimension de l'encodeur), dimension cachée alignée sur la sortie de l'encodeur (768 ou 1024).
* **Mécanisme :** L'attention croisée (Cross-Attention) fait correspondre les requêtes textuelles (Queries) aux clés/valeurs acoustiques (Keys/Values) générées par Pantagruel.

### 2.3 Architecture logicielle du pipeline

Le code d'exécution est organisé en **un module Python par étape**, orchestré par une CLI unique :

| Étape | Fichier | Subcommand `pipeline.py` | Rôle | Statut |
| :--- | :--- | :--- | :--- | :--- |
| Bootstrap | `scripts_communs/bootstrap.sh` | — | Installation venv + dépendances Phase 1 | implémenté |
| 0 — Preflight | `scripts_communs/0_preflight.py` | `preflight` | Validation machine distante (Linux + CUDA) | implémenté |
| 1 — Download | `scripts_communs/1_download.py` | `download` | Téléchargement m-TEDx (OpenSLR-100), défaut `fr-en` | implémenté |
| 2 — Prepare | `scripts_communs/2_prepare.py` | `prepare` | Audio 16 kHz, manifests, normalisation texte | implémenté |
| 3 — SPM | `1_Transformer/3_spm.py` | `spm` | Tokenizers SentencePiece (train uniquement) | implémenté |
| 4 — Train | `1_Transformer/4_train.py` | `train` | Entraînement ST (encodeur + décodeur) | implémenté |
| 5 — Evaluate | `1_Transformer/5_evaluate.py` | `evaluate` | Décodage + métriques SacreBLEU | implémenté |
| 6 — Infer | `1_Transformer/6_infer.py` | `infer` | Inférence sur nouveaux audios | implémenté |
| Orchestrateur | `scripts_communs/pipeline.py` | `run` (+ toutes les étapes) | Routeur CLI, enchaînement `--from-stage` / `--to-stage` | routeur actif |

**Règles d'architecture :**
* Chaque stage expose un point d'entrée (`main()` / `run_from_namespace(args)`) et peut être exécuté **directement** ou via `pipeline.py`.
* `pipeline.py` ne contient pas la logique métier des stages : il délègue aux modules numérotés.
* Les options CLI communes (`--verbose`, `--dry-run`, `--log-file`) sont homogènes entre stages.

### 2.3.1 Pipeline speechLLM (ligne prioritaire fr→en, B1)

Alternative au décodeur Transformer : **Pantagruel gelé** → downsampling → **projecteur entraînable** → **LLM causal gelé** (inspiré SLAM-ASR). Les stages données `0`–`3` restent dans `scripts/`.

| Étape | Fichier | Subcommand `2_speechLLM/pipeline.py` | Rôle | Statut |
| :--- | :--- | :--- | :--- | :--- |
| Train B1 | `2_speechLLM/train.py` | `train` | Loss masquée sur tokens `ASSISTANT` | implémenté |
| Evaluate | `2_speechLLM/evaluate.py` | `evaluate` | SacreBLEU valid/test (texte brut, pas SPM) | implémenté |
| Infer | `2_speechLLM/infer.py` | `infer` | WAV arbitraire → traduction anglaise | implémenté |
| Common | `2_speechLLM/speechllm_common.py` | — | Modèle, collate, checkpoints projecteur | implémenté |
| Config | `2_speechLLM/configs/fr-en/b1.yaml` | — | Pilote Phi-2 ; B2bis Qwen/Mistral dans `b2bis_*.yaml` | implémenté |
| Orchestrateur | `2_speechLLM/pipeline.py` | `run` | `train` → `evaluate` | routeur actif |

**Artifacts :** même contrat que §2.3 (`runs/<lang_pair>/<run_id>/`, `eval/sacrebleu_*.txt`, signature SacreBLEU). Checkpoints : `trainable_state` (projecteur ; + tenseurs `encoder.*` si `freeze_encoder: false`). Voir [plan_migration_speechllm.md](plan_migration_speechllm.md) et [2_speechLLM/README.md](../2_speechLLM/README.md).

### 2.3.2 Baseline API — Gemini ST (audio → texte)

Baseline externe destinée à fournir un **point de comparaison rapide** (zéro entraînement) pour
les résultats obtenus avec les pipelines locaux (baseline ST et `speechLLM`).

Principes :
- **Entrées** : manifests `valid.tsv` / `test.tsv` produits par `2_prepare.py` (WAV 16 kHz mono + `tgt_text`).
- **Prompt** : instruction textuelle (champ `prompt.template`) conçue pour être réutilisable par la partie LLM de `speechLLM`.
- **Sorties** : même contrat d’artefacts d’évaluation (`runs/.../eval/` + signature SacreBLEU) avec `pipeline = gemini_st`.
- **Tracking coût/temps** : `eval/metrics.json` inclut `runtime.elapsed_minutes` et `gemini_cost_estimate_usd` (estimé via `pricing.*` dans la config Gemini).
- **Décodage Gemini 3.x** : champ optionnel `decode.thinking_level` (`minimal`, `low`, `medium`, `high`) — configs `gemini_flash_35_*_v2.yaml` (`max_output_tokens: 8192`, `thinking_level: minimal`).
- **Post-traitement réponse** (`gemini_common.sanitize_gemini_translation`) : exclusion des parts `thought` à l’extraction ; troncature des boucles de répétition (artefact `MAX_TOKENS`) avant SacreBLEU. Run de référence utterance v2 : `run_005_gemini_35_flash_utterance_v2` (41,42 / 41,09 BLEU).
- **Authentification** : variable d’environnement `GEMINI_API_KEY` (aucun secret versionné).

CLI (routeur `3_Gemini/pipeline.py`) :
- `evaluate` : décodage API + SacreBLEU valid/test
- `infer` : traduction d’un WAV arbitraire + append JSONL

### 2.3.3 Baseline cascade — ASR→MT

Baseline en **deux étages** (comparaison papier / pratique industrie) : reconnaissance du français
puis traduction texte vers l’anglais. Pas d’entraînement end-to-end ST dans cette piste.

Principes :
- **Entrées** : mêmes manifests `valid.tsv` / `test.tsv` que `2_prepare` (utterance ou `sentence_like`).
- **Chaîne** : `audio (FR)` → ASR → `texte FR` → MT → `texte EN` ; métrique finale **SacreBLEU** sur l’anglais.
- **Sorties** : contrat `runs/.../eval/` identique ; `pipeline = cascade_asr_mt` dans `metrics.json`.
- **Configs** : `4_cascade/configs/<langpair>/cascade.yaml` (utterance) et `cascade_sentence.yaml` (option).

CLI (routeur `4_cascade/pipeline.py`) :
- `evaluate` : décodage valid/test + SacreBLEU (ASR Whisper + MT Marian ; `--dry-run` et `--limit` disponibles).
- `infer` : WAV arbitraire → JSONL.

Backends cibles (YAML `asr` / `mt`) : Whisper + Marian par défaut ; extensible (Pantagruel ASR, NLLB, etc.).
Détail : [4_cascade/README.md](../4_cascade/README.md).

### 2.3.4 Variante Pantagruel multimodale (`speech_text`) — expérimentale

Piste dédiée aux essais avec un checkpoint Pantagruel **multimodal récent** (famille
`Speech_Text_*`) en mode full model, séparée des runs prioritaires `2_speechLLM`.

Principes :
- **Entrées** : mêmes manifests `valid.tsv` / `test.tsv` issus de `2_prepare`.
- **Objectif** : tester l'apport du modèle multimodal tel quel, sans réécrire la piste B1.
- **Sorties cibles** : contrat `runs/.../eval/` identique aux autres variantes (SacreBLEU).
- **Statut** : **implémenté** (délégation `1_Transformer` 3–6) — encodeur `Speech_Text_*` via `model.encoder_api: speech_text` + `trust_remote_code: true` ; décodeur Transformer + SPM ; données `sentence_like` par défaut.
- **Limite** : pas de perte multimodale speech+text du pretrain ; fine-tuning ST sur encodeur audio uniquement.

CLI (routeur `5_Pantagruel_multimodal/pipeline.py`) :
- `spm` : SentencePiece sur `manifests_sentence` (section `spm` du YAML)
- `train` : fine-tuning ST (`4_train.py`, encodeur Speech_Text)
- `evaluate` : SacreBLEU dev/test + mise à jour `experiments_tracking.csv`
- `infer` : WAV arbitraire (`6_infer.py`)

### 2.4 Qualité logicielle et workflow de contribution

* **Langues :** code et commentaires en **anglais** ; documentation projet (`README.md`, `PRD.md`, etc.) en **français**.
* **Lint / format :** **Ruff** obligatoire avant chaque commit (`ruff check`, `ruff format --check`).
* **Tests :** **pytest** obligatoire avant chaque commit.
* **Hooks :** configuration `pre-commit` recommandée (voir `.pre-commit-config.yaml`).
* **Documentation :** toute évolution fonctionnelle du pipeline doit mettre à jour **PRD.md** et **README.md** dans le même commit.

### 2.5 Source de vérité historique et transposition SpeechBrain

**Contexte :** l'expérience ST du papier Pantagruel a été réalisée initialement avec un pipeline de recherche basé sur **fairseq** (dépôt historique `../fairseq/`, branche de référence `origin/pantagruel_uni`). Ce projet S3T vise la **même expérience scientifique** (m-TEDx, encodeur Pantagruel, décodeur Transformer, BLEU SacreBLEU) mais avec **SpeechBrain** comme stack d'entraînement — **sans utiliser fairseq**.

**Plan en deux temps :**
* **Temps A — Réplication fidèle :** reproduire le protocole expérimental observable (données, fine-tuning ST, évaluation BLEU) avec artefacts complets et comparables.
* **Temps B — Améliorations :** optimisations et ablations après baseline stabilisée, sans casser la traçabilité des runs A.

#### Tableau : alignement des étapes

| Étape S3T | Équivalent historique (fairseq, lecture seule) | Équivalent SpeechBrain typique | Statut S3T |
| :--- | :--- | :--- | :--- |
| `bootstrap.sh` | setup env + deps | install SB + deps | implémenté |
| `0_preflight` | checks machine implicites | rarement un stage dédié | implémenté |
| `1_download` | téléchargement OpenSLR m-TEDx | données supposées prêtes | implémenté |
| `2_prepare` | `prep_mtedx_data.py` (+ aug. speed pert. sur branche historique) | `DynamicItemDataset` / CSV recipe | implémenté |
| `3_spm` | vocab BPE fairseq | tokenizer intégré ou SPM externe | implémenté |
| `4_train` | `fairseq-hydra-train` / scripts Pantagruel | `Brain` + `hparams/train.yaml` | implémenté |
| `5_evaluate` | `fairseq-generate` + scorers | `decode` puis métriques (souvent séparés) | implémenté |
| `6_infer` | génération hors split | inférence recipe dédiée | implémenté |

#### Divergences explicites par rapport à un pipeline SpeechBrain « recette standard »

Les points suivants **ne suivent pas** le schéma minimal `recipes/<task>/train.py` + `hparams/*.yaml` tel quel. Ils sont **volontaires** pour ce dépôt (traçabilité, reproductibilité distante, alignement papier).

| Sujet | Pipeline SpeechBrain typique | Choix S3T (divergence) | Impact |
| :--- | :--- | :--- | :--- |
| **Point d'entrée** | Une recette SB par tâche (`python train.py hparams/...`) | CLI modulaire `scripts/N_*.py` + routeur `pipeline.py` | Plus verbeux, mais meilleur contrôle ops/CI |
| **Preflight** | Pas de stage standard | `0_preflight.py` obligatoire avant jobs distants | Étape supplémentaire hors SB |
| **Download** | Données fournies en amont | `1_download.py` intégré (défaut `fr-en` seul) | Diverge des tutos SB ; défaut ≠ les 3 paires |
| **Préparation données** | CSV/DynamicItem via API SB | Manifests TSV maison + règles PRD (NFKC, durées, anti-fuite) | Format et filtres à mapper vers loaders SB |
| **Tokenisation** | Souvent dans la recette data | Stage dédié `3_spm.py` (train only) | Séparation explicite, pas le flux SB par défaut |
| **Modèle** | `speechbrain.lobes` + interfaces HF | Encodeur Pantagruel (HF) + décodeur PyTorch custom 6 couches | Pas un modèle SB pré-packagé S2T |
| **Entraînement** | `Brain.fit()` + checkpoints SB | `4_train.py` + config YAML versionnée par run | API d'entraînement à encapsuler (SB ou hybride) |
| **Évaluation** | `Evaluator` / scripts decode SB | `5_evaluate.py` = decode **puis** SacreBLEU externe figé | Decode et scoring explicitement séparés dans les artifacts |
| **Inférence** | Même recette avec mode test | `6_infer.py` dédié (fichiers WAV arbitraires) | Chemin production distinct du eval dev/test |
| **Tracking** | Checkpoints SB + logs internes | `runs/<pair>/<run_id>/` + `experiments_tracking.csv` + manifest JSON | Contrat d'artifacts plus strict que la doc SB minimale |
| **Métriques** | BLEU via utils SB possibles | SacreBLEU en CLI canonique + signature obligatoire | Alignement papier LeBenchmark, pas défaut SB |

> **Divergence SpeechBrain — pré-entraînement multimodal :** le fil historique fairseq inclut des étapes de **prétrain Pantagruel / data2vec multimodal** et des pertes combinées (`pantagruel_multi_loss`, MLM speech-text) absentes du pipeline SB minimal. **Temps A** part du **checkpoint Pantagruel déjà entraîné** (Hugging Face) ; le prétrain multimodal complet n'est **pas** recodé dans S3T sauf décision explicite ultérieure.

> **Divergence SpeechBrain — augmentation m-TEDx :** la branche historique mentionne une **speed perturbation** sur le prétraitement m-TEDx. Le PRD S3T fixe pour l'instant un prétraitement plus simple (SpecAugment à l'entraînement). Toute réintroduction de speed pert. sera une **extension Temps B**, documentée à part.

> **Divergence SpeechBrain — hyperparamètres encodeur :** fairseq historique utilise parfois `encoder_grad_multi` et des schedulers spécifiques Hydra. S3T retient la matrice LeBenchmark (§5) et le gel d'encodeur (RF-11) ; les noms et mécanismes SB (`freeze`, param groups) peuvent différer tant que le comportement est documenté par run.

> **Divergence SpeechBrain — métriques annexes :** fairseq historique expose aussi **WER+BLEU** et **ASR-BLEU**. Ce ne sont **pas** des métriques natives SpeechBrain. Elles restent **hors scope Temps A** sauf ajout explicite au PRD ; le critère principal reste **SacreBLEU** (RF-14 à RF-19).

#### Contrat d'artifacts par run (commun Temps A et B)

Chaque run doit contenir au minimum :

```text
runs/<langpair>/<run_id>/
  config.yaml              # copie figée des hyperparamètres
  train.log
  checkpoints/
    best.pt
  eval/
    dev_predictions.txt
    test_predictions.txt
    sacrebleu_dev.txt      # avec signature
    protocol.json          # version figée du protocole (protocole_evaluation.md)
    sacrebleu_test.txt
    metrics.json
```

> **Divergence SpeechBrain :** SpeechBrain stocke souvent les checkpoints dans un dossier d'expérience interne sans manifeste JSON/SacreBLEU externe obligatoire. S3T **impose** ce paquet pour la reproductibilité et la comparaison Table 8.

---

## 3. Besoins Fonctionnels & Pipeline de Données

### 3.1 Ingestion et Prétraitement des Données

> **Divergence SpeechBrain :** les recettes SB supposent en général des CSV/`DynamicItemDataset` prêts ; S3T impose des **manifests TSV** (`2_prepare.py`), un stage **download** séparé et des filtres RF-03 à RF-05 documentés ici — voir aussi §2.5.

* **RF-01 :** Téléchargement automatique ou scripté du corpus m-TEDx (OpenSLR-100).
* **RF-02 :** Normalisation audio : Conversion systématique de tous les segments audio en format WAV, 16 kHz, mono, 16-bit PCM.
* **RF-03 :** Filtrage des données : Élimination des segments vides ou dont la transcription textuelle est manquante.
* **RF-04 :** Filtrage de durée : Exclure les segments hors bornes (ex: `<1s` ou `>30s`) pour stabiliser l'entraînement.
* **RF-05 :** Règle anti-fuite de données : Respect strict des splits `train/valid/test` d'origine; aucune phrase de test ne doit apparaître dans les corpus de tokenisation ou d'entraînement.

#### Extension optionnelle — segmentation \"phrase-like\" (unités syntaxiques)

Pour maximiser la qualité de traduction (et la stabilité des métriques), on peut regrouper des segments contigus
à l'intérieur de chaque split afin d'approcher des unités de type phrase.

Implémentation (opt-in) : `scripts_communs/2_prepare.py`
- `--segment-mode utterance` (défaut) : segmentation m-TEDx native (utterances).
- `--segment-mode sentence_like` : fusion contiguë **dans un même talk** (et speaker si possible), avec :
  - `--sentence-target-duration` (défaut 10s) : durée cible d'un segment fusionné,
  - `--sentence-max-duration` (défaut 15s) : borne dure,
  - `--sentence-require-punctuation/--no-sentence-require-punctuation` : couper préférentiellement sur `.?!`.

Recommandation de reproductibilité : conserver les manifests \"référence\" et produire les manifests phrase-like dans
un répertoire séparé (ex. `datasets/manifests_sentence/` + `datasets/processed_sentence/`).

### 3.2 Tokenisation (Texte Cible)
* **RF-06 :** Entraînement de trois tokeniseurs distincts via **SentencePiece** (algorithme Unigram ou BPE) sur les textes cibles (anglais, portugais, espagnol) du jeu d'entraînement.
* **RF-07 :** Taille du vocabulaire ciblée : Entre 1 000 et 5 000 sous-mots (Subwords) pour s'adapter à la faible quantité de données par couple de langues.
* **RF-08 :** Normalisation texte explicite et figée avant tokenisation : `NFKC`, suppression d'espaces redondants, normalisation de casse (stratégie documentée), traitement stable de la ponctuation et des chiffres.

### 3.3 Entraînement et Optimisation

> **Divergence SpeechBrain :** l'entraînement cible un modèle **Pantagruel (HF) + décodeur custom**, pas un lobes S2T SB préfabriqué ; le gel encodeur (RF-11) et les schedulers peuvent être implémentés via `Brain` ou couche PyTorch, avec noms de config différents de fairseq — voir §2.5.

* **RF-09 :** Calcul de la perte via une entropie croisée (Cross-Entropy Loss) standard avec lissage d'étiquette (*Label Smoothing* = 0.1).
* **RF-10 :** Implémentation d'un planificateur de taux d'apprentissage (Learning Rate Scheduler) avec une phase de montée linéaire (*Warmup*) et une décroissance inverse de la racine carrée (Inverse Square Root Decay).
* **RF-11 :** Stratégie de gel des poids (*Freezing*) : Possibilité de geler l'encodeur Pantagruel pendant les N premières étapes (ex: 5 000 à 10 000 updates) pour stabiliser le décodeur initialisé aléatoirement.
* **RF-12 :** Stabilité d'entraînement : clipping du gradient (ex: `max_norm=1.0`), `gradient_accumulation`, et entraînement mixte (`fp16`/`bf16`) pour atteindre un batch effectif reproductible sur GPU limité.

### 3.4 Inférence et Évaluation

> **Divergence SpeechBrain :** `5_evaluate.py` sépare explicitement **décodage** et **SacreBLEU CLI** (artifacts texte + signature) ; `6_infer.py` est un chemin hors splits m-TEDx — ce n'est pas le flux unique « eval recipe » SB — voir §2.5.

* **RF-13 :** Implémentation d'une recherche par faisceau (**Beam Search Decoding**) avec une largeur de faisceau (*Beam Width*) de 5.
* **RF-14 :** Évaluation via la bibliothèque officielle **SacreBLEU** avec la signature standard pour garantir la reproductibilité et la comparaison équitable avec LeBenchmark 2.0. Protocole opérationnel figé : [protocole_evaluation.md](protocole_evaluation.md) (`2026-06-02-v1`, module `scripts_communs/eval_protocol.py`, artefact `eval/protocol.json` par run).
* **RF-15 :** Critère de sélection du meilleur checkpoint : `BLEU dev` prioritaire; `loss dev` utilisée comme signal secondaire en cas d'ambiguïté.

### 3.5 Protocole d'Évaluation Reproductible (Obligatoire)
* **RF-16 :** Fixer et documenter la version de `sacrebleu` utilisée dans les expériences.
* **RF-17 :** Enregistrer la commande canonique d'évaluation BLEU (mêmes options pour tous les runs et toutes les langues).
* **RF-18 :** Logger la signature SacreBLEU complète dans les artifacts de run.
* **RF-19 :** Appliquer exactement les mêmes règles de normalisation/dé-tokenisation au couple `hypotheses/references` avant calcul BLEU.

### 3.6 Reproductibilité Expérimentale (Obligatoire)
* **RF-20 :** Fixer les seeds globales (`python`, `numpy`, `torch`, `cuda`) et les logger.
* **RF-21 :** Documenter les paramètres de déterminisme (`cudnn.deterministic`, `cudnn.benchmark`) et le compromis performance/reproductibilité.
* **RF-22 :** Versionner l'environnement d'exécution (fichier de dépendances, version CUDA, commit Git du code).
* **RF-23 :** Sauvegarder une configuration complète par run (YAML/JSON) avec chemins des checkpoints et paramètres de décodage.

---

## 4. Plan de Projet & Étapes d'Exécution (Gantt Conceptuel)

Ce plan est conçu pour être exécuté de manière itérative. Chaque phase doit être validée avant de passer à la suivante.

### Phase 1 : Configuration de l'Environnement & Outillage (Jours 1-2)
* Scripts : `scripts_communs/bootstrap.sh`, `scripts_communs/0_preflight.py`, orchestration via `scripts_communs/pipeline.py preflight`.
* Créer un environnement virtuel isolé (`conda` ou `venv` avec Python 3.10+).
* Installer les dépendances : `pip install -r requirements.txt` puis `pre-commit install` (voir `scripts_communs/bootstrap.sh` ; optionnel `pip freeze > requirements.lock.txt`).
* Valider la chaîne qualité : `ruff check .`, `ruff format --check .`, `pytest`.
* Configurer les accès au GPU (Vérification de `cuda.is_available()`).
* Télécharger les poids pré-entraînés du modèle Pantagruel sur Hugging Face (`PantagrueLLM/`).
* Fixer les seeds globales et préparer un template de configuration de run versionné.
* **Jalon go/no-go :** environnement reproductible validé (dépendances figées + script de seed unique + test GPU OK).

### Phase 2 : Ingestion et Tokenisation m-TEDx (Jours 3-5)
* Scripts : `scripts_communs/1_download.py`, `scripts_communs/2_prepare.py` (implémentés), `1_Transformer/3_spm.py`.
* Télécharger les partitions m-TEDx via `python scripts_communs/1_download.py` (défaut: `fr-en`; option multi-paires: `fr-en,fr-pt,fr-es`).
* Écrire le script de prétraitement audio (vérification/conversion du taux d'échantillonnage à 16 kHz via `torchaudio` ou `pydub`).
* Générer les fichiers de manifestes (`train.tsv`, `valid.tsv`, `test.tsv`) contenant les chemins des fichiers audio, la durée et les textes cibles.
* Entraîner les modèles SentencePiece pour chaque langue cible et sauvegarder les fichiers de vocabulaire (`.model` et `.vocab`).
* Documenter explicitement les règles de normalisation texte et de filtrage de durée.
* **Jalon go/no-go :** manifests contrôlés (0 segment vide, 0 fuite entre splits, stats de durée/longueur textuelle exportées).

### Phase 3 : Construction du Modèle Seq2Seq (Jours 6-7)
* Initialiser l'encodeur Pantagruel via la classe adéquate (ex: `Wav2Vec2Model` ou `Data2Vec2Model` selon l'export Hugging Face).
* Configurer et instancier le décodeur Transformer à 6 couches (`nn.TransformerDecoder`).
* Implémenter la couche de projection linéaire (Linear Layer) pour mapper les sorties cachées du décodeur vers la taille du vocabulaire SentencePiece.
* Écrire un test unitaire avec un batch fictif (dummy batch) pour valider le passage des données (*Forward Pass*) et la dimension des tenseurs.
* **Jalon go/no-go :** `forward` et `backward` valides sur batch fictif sans erreur mémoire.

### Phase 4 : Pipeline d'Entraînement & Alignement Hyperparamétrique (Jours 8-12)
* Script : `1_Transformer/4_train.py`.
* Implémenter la boucle d'entraînement principale conforme aux paramètres de *LeBenchmark 2.0* (Parcollet et al., 2024).
* Configurer l'optimiseur AdamW (`weight_decay=0.01`) et le scheduler de Learning Rate.
* Intégrer les mécanismes de régularisation : Dropout (0.1) et SpecAugment (masquage temporel et fréquentiel sur les features acoustiques).
* Configurer `gradient_accumulation`, AMP (`fp16`/`bf16`) et clipping du gradient.
* Lancer les scripts d'entraînement pour le premier couple (recommandation : `fr-es` car il fait 25h, idéal pour un prototypage rapide).
* Surveiller la convergence de la courbe de perte (Loss) et du `BLEU dev` sur TensorBoard ou Weights & Biases.
* **Jalon go/no-go :** perte en baisse + `BLEU dev` supérieur à la baseline minimale sur `fr-es`.

### Phase 5 : Inférence, Génération et Évaluation (Jours 13-15)
* Scripts : `1_Transformer/5_evaluate.py`, `1_Transformer/6_infer.py`.
* Charger le meilleur checkpoint basé sur `BLEU dev` (loss comme signal secondaire).
* Écrire le script de décodage autoregressif en mode *Beam Search*.
* Traduire l'ensemble de test m-TEDx et sauvegarder les prédictions textuelles brutes.
* Appliquer la dé-tokenisation pour retrouver le texte sous forme de phrases lisibles.
* Exécuter le calcul SacreBLEU sur les fichiers de sortie avec commande canonique figée.
* Consigner les résultats dans un tableau comparatif face à la Table 8 du papier Pantagruel.
* Logger la signature SacreBLEU et les paramètres de décodage dans les artifacts finaux.
* **Jalon go/no-go :** rapport final reproductible (commande d'éval + signature + config + commit).

---

## 5. Matrice des Hyperparamètres (Référence LeBenchmark 2.0)

Pour assurer une comparaison équitable, utilisez les valeurs cibles suivantes lors de la configuration de vos scripts :

| Paramètre | Valeur Recommandée | Note |
| :--- | :--- | :--- |
| **Optimiseur** | AdamW | Beta_1 = 0.9, Beta_2 = 0.98 |
| **Taux d'apprentissage max (LR)** | 1e-4 à 3e-4 | Ajuster selon la taille du Batch |
| **Warmup updates** | 10 000 | Augmentation progressive du LR |
| **Label Smoothing** | 0.1 | Évite la sur-confiance du décodeur |
| **Largeur du faisceau (Beam)**| 5 | Paramètre de génération au test |
| **Ajustement de l'encodeur** | Figé puis Dégelé | Geler les 5000 premières étapes |
| **Gradient Clipping** | `max_norm=1.0` | Limite les gradients explosifs |
| **Batch effectif** | 64 à 256 séquences | Via `batch_size x gradient_accumulation` |
| **Précision mixte** | `fp16` ou `bf16` | Réduit mémoire/temps d'entraînement |
| **Cadence d'évaluation** | Tous les N updates | Ex: 1000 updates |
| **Early stopping** | Patience 5 à 10 évaluations | Basé sur `BLEU dev` |
| **Seeds** | Fixes et loggées | `python`, `numpy`, `torch`, `cuda` |
| **Déterminisme CuDNN** | Documenté par run | `deterministic` + `benchmark` |
| **Versionnement run** | Obligatoire | commit Git + config YAML/JSON + versions libs |

---

## 6. Baselines & Mini-Ablations (Obligatoire)

### 6.1 Baseline minimale
* **BL-01 :** Entraîner une baseline avec encodeur totalement figé et décodage `beam=1`.
* **BL-02 :** Consigner `BLEU dev`, `BLEU test`, temps d'entraînement, et mémoire GPU.

### 6.2 Ablations courtes
* **AB-01 :** `freeze=5k` vs `freeze=10k`.
* **AB-02 :** vocabulaire `1k` vs `5k`.
* **AB-03 :** décodage `beam=1` vs `beam=5`.
* **Règle de décision :** promouvoir uniquement les variantes qui améliorent `BLEU dev` de manière stable (au moins 2 runs cohérents).

---

## 7. Risques Connus et Atténuation

1. **Divergence ou Oubli Catastrophique de l'Encodeur**
   * *Risque :* Le décodeur, initialisé aléatoirement, envoie des gradients chaotiques à l'encodeur au début de l'entraînement, détruisant ses connaissances linguistiques pré-entraînées.
   * *Atténuation :* Appliquer un gel strict de l'encodeur (`requires_grad=False`) durant la phase de Warmup ou utiliser un taux d'apprentissage 10x plus faible pour l'encodeur que pour le décodeur.
2. **Surapprentissage (Overfitting) sur m-TEDx**
   * *Risque :* Les volumes de données sont faibles (surtout les 25h de `fr-es`). Le modèle apprend par cœur les voix du jeu d'entraînement.
   * *Atténuation :* Augmenter la probabilité de masquage de SpecAugment et utiliser un dropout plus agressif (jusqu'à 0.15) dans les couches du décodeur.
3. **Non-reproductibilité des expériences**
   * *Risque :* Des différences de seed, de version de bibliothèques, ou de paramètres CUDA empêchent de reproduire les scores BLEU.
   * *Atténuation :* Journaliser systématiquement seeds, versions, commit Git, configuration complète et environnement matériel.
4. **Écart de protocole BLEU avec le papier**
   * *Risque :* Une différence de normalisation/dé-tokenisation ou d'options SacreBLEU rend la comparaison avec Pantagruel invalide.
   * *Atténuation :* Suivre [protocole_evaluation.md](protocole_evaluation.md) (version figée, `eval/protocol.json`, signature SacreBLEU) ; incrémenter la version de protocole si le décodage ou la normalisation change.

---

## 8. Protocole opérationnel des runs

Complète [README.md](../README.md) (usage CLI des stages). Points obligatoires pour chaque expérience :

### 8.1 Convention de nommage

`run_<id>_<langpair>_seed<seed>_freeze<updates>_vocab<size>_beam<n>`

Exemple : `run_001_fr-en_seed42_freeze5k_vocab1k_beam5`

Le répertoire de sortie suit `runs/<langpair>/<run_id>/` (voir contrat d'artifacts §2.5).

### 8.2 Reproductibilité (RF-20 à RF-23)

Chaque run doit conserver :

- commit Git (`git rev-parse HEAD`) ;
- `requirements.lock.txt` (optionnel mais recommandé après bootstrap) ;
- copie figée `runs/.../config.yaml` ;
- seeds et flags CuDNN loggés dans `train.log` ;
- commandes exactes exécutées (train / evaluate / infer).

Les scripts d'entraînement appellent `set_seed()` dans [`scripts/st_common.py`](../scripts_communs/st_common.py) selon `experiment.seed` et `experiment.deterministic` du YAML.

### 8.3 Suivi des expériences

Fichier agrégé recommandé : `runs/experiments_tracking.csv`

```csv
run_id,lang_pair,pipeline,segment_mode,seed,freeze_updates,vocab_size,beam,bleu_dev,bleu_test,chrf_dev,chrf_test,ter_dev,ter_test,train_hours,gpu_hours,estimated_gpu_cost_usd,max_gpu_mem_gb,gemini_duration_min,gemini_cost_usd,gemini_input_usd_per_1m,gemini_output_usd_per_1m,git_commit,status,notes
```

Mettre à jour après chaque run pour comparer à la Table 8 Pantagruel.

**Coût API Gemini (baseline `gemini_st`)** — grilles **tier Standard** ([tarifs officiels](https://ai.google.dev/gemini-api/docs/pricing)) :

| Modèle | Entrée (ST audio) | Sortie texte (incl. thinking) | Config YAML |
|--------|-------------------|-------------------------------|-------------|
| **Gemini 2.5 Flash** | **1,00** USD / 1M tokens (audio ; texte/image/vidéo : 0,30) | **2,50** | `gemini_flash*.yaml` |
| **Gemini 3.5 Flash** | **1,50** USD / 1M tokens (multimodal, audio = texte) | **9,00** | `gemini_flash_35_*.yaml` |

L'étape `3_Gemini/evaluate_gemini.py` estime le coût dans `eval/metrics.json` (`gemini_cost_estimate_usd`) à partir des tokens API et des champs `pricing.*` de la config du run ; mise à jour CSV via `scripts_communs/update_experiments_tracking.py`.

```bash
# Synchronisation manuelle (tous les runs avec eval/metrics.json)
python scripts_communs/update_experiments_tracking.py --all

# Un seul run
python scripts_communs/update_experiments_tracking.py \
  --run-dir runs/fr-en/run_001_gemini_flash_sentence_like_v2
```

**Note :** les runs évalués avant l'introduction des champs `runtime` / `gemini_cost_estimate_usd` n'ont pas de coût rétroactif dans `metrics.json` ; relancer `evaluate` avec une config `pricing` renseignée pour remplir `gemini_cost_usd`.

### 8.4 Checklist de clôture d'un run

- [ ] `config.yaml` dans le dossier run
- [ ] commit Git enregistré (dans tracking ou notes)
- [ ] `train.log` et checkpoints `best.pt` / `last.pt`
- [ ] `eval/dev_predictions.txt`, `eval/test_predictions.txt`
- [ ] `eval/sacrebleu_dev.txt`, `eval/sacrebleu_test.txt` (signature SacreBLEU)
- [ ] `eval/protocol.json` (version `eval_protocol_version`)
- [ ] `eval/metrics.json`
- [ ] ligne ajoutée à `experiments_tracking.csv`

### 8.5 Évaluation SacreBLEU (référence)

L'étape `5_evaluate.py` produit les métriques avec protocole figé. Pour une invocation manuelle (debug) :

```bash
sacrebleu datasets/manifests/fr-en/valid.target.txt \
  -i runs/fr-en/<run_id>/eval/dev_predictions.txt \
  -m bleu chrf ter -w 2 \
  > runs/fr-en/<run_id>/eval/sacrebleu_dev_manual.txt
```

**Règle :** même commande et mêmes options entre tous les runs comparés ; conserver la signature dans les artifacts.

### 8.6 Smoke test local (FR→FR, hors ST)

Le checkpoint `PantagrueLLM/Speech_Text_Base_fr_1K_4GB` sert à valider l'environnement HF (encodeur seul) ou un proxy ASR Whisper — voir [README.md § Smoke test](../README.md#smoke-test-frfr-asr--encodeur-pantagruel).

---

## 9. Template de configuration run (YAML)

À placer sous `1_Transformer/configs/<langpair>/base.yaml` (chemins à adapter). Les champs non encore lus par `4_train.py` restent documentés pour alignement papier.

```yaml
experiment:
  name: "fr-en_base"
  lang_pair: "fr-en"
  output_dir: "runs/fr-en/run_001_fr-en_seed42_freeze5k_vocab1k_beam5"
  seed: 42
  deterministic: true

data:
  train_manifest: "datasets/manifests/fr-en/train.tsv"
  valid_manifest: "datasets/manifests/fr-en/valid.tsv"
  test_manifest: "datasets/manifests/fr-en/test.tsv"
  spm_model: "datasets/processed/spm/fr-en_1000.model"
  sample_rate: 16000
  min_duration_s: 1.0
  max_duration_s: 30.0

model:
  encoder_name: "PantagrueLLM/Pantagruel-Base"
  decoder_layers: 6
  decoder_heads: 8
  hidden_dim: 768
  dropout: 0.1

train:
  max_updates: 120000
  warmup_updates: 10000
  freeze_encoder_updates: 5000
  learning_rate_peak: 0.0002
  weight_decay: 0.01
  label_smoothing: 0.1
  batch_size: 8
  gradient_accumulation: 8
  gradient_clip_norm: 1.0
  amp_dtype: "bf16"
  eval_every_updates: 1000
  early_stopping_patience: 8
  best_checkpoint_metric: "bleu_dev"

decode:
  beam_size: 5
  max_len_a: 1.2
  max_len_b: 10
```

**Commandes associées (pipeline actuel) :**

```bash
python 1_Transformer/pipeline.py spm --langpair fr-en --vocab-size 1000
python 1_Transformer/pipeline.py train --config 1_Transformer/configs/fr-en/base.yaml --run-id run_001_fr-en
python 1_Transformer/pipeline.py evaluate --config 1_Transformer/configs/fr-en/base.yaml --run-id run_001_fr-en
```