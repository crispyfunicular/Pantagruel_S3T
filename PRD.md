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

| Étape | Fichier | Subcommand `pipeline.py` | Rôle |
| :--- | :--- | :--- | :--- |
| Bootstrap | `scripts/bootstrap.sh` | — | Installation venv + dépendances Phase 1 |
| 0 — Preflight | `scripts/0_preflight.py` | `preflight` | Validation machine distante (Linux + CUDA) |
| 1 — Download | `scripts/1_download.py` | `download` | Téléchargement m-TEDx (OpenSLR-100), défaut `fr-en` |
| 2 — Prepare | `scripts/2_prepare.py` | `prepare` | Audio 16 kHz, manifests, normalisation texte |
| 3 — SPM | `scripts/3_spm.py` | `spm` | Tokenizers SentencePiece (train uniquement) |
| 4 — Train | `scripts/4_train.py` | `train` | Entraînement ST (encodeur + décodeur) |
| 5 — Evaluate | `scripts/5_evaluate.py` | `evaluate` | Décodage + métriques SacreBLEU |
| 6 — Infer | `scripts/6_infer.py` | `infer` | Inférence sur nouveaux audios |
| Orchestrateur | `scripts/pipeline.py` | `run` (+ toutes les étapes) | Routeur CLI, enchaînement `--from-stage` / `--to-stage` |

**Règles d'architecture :**
* Chaque stage expose un point d'entrée (`main()` / `run_from_namespace(args)`) et peut être exécuté **directement** ou via `pipeline.py`.
* `pipeline.py` ne contient pas la logique métier des stages : il délègue aux modules numérotés.
* Les options CLI communes (`--verbose`, `--dry-run`, `--log-file`) sont homogènes entre stages.

### 2.4 Qualité logicielle et workflow de contribution

* **Langues :** code et commentaires en **anglais** ; documentation projet (`README.md`, `PRD.md`, `AGENTS.md`, etc.) en **français**.
* **Lint / format :** **Ruff** obligatoire avant chaque commit (`ruff check`, `ruff format --check`).
* **Tests :** **pytest** obligatoire avant chaque commit.
* **Hooks :** configuration `pre-commit` recommandée (voir [AGENTS.md](AGENTS.md)).
* **Documentation :** toute évolution fonctionnelle du pipeline doit mettre à jour **PRD.md** et **README.md** dans le même commit.
* **Référence agents :** conventions détaillées dans [AGENTS.md](AGENTS.md).

---

## 3. Besoins Fonctionnels & Pipeline de Données

### 3.1 Ingestion et Prétraitement des Données
* **RF-01 :** Téléchargement automatique ou scripté du corpus m-TEDx (OpenSLR-100).
* **RF-02 :** Normalisation audio : Conversion systématique de tous les segments audio en format WAV, 16 kHz, mono, 16-bit PCM.
* **RF-03 :** Filtrage des données : Élimination des segments vides ou dont la transcription textuelle est manquante.
* **RF-04 :** Filtrage de durée : Exclure les segments hors bornes (ex: `<1s` ou `>30s`) pour stabiliser l'entraînement.
* **RF-05 :** Règle anti-fuite de données : Respect strict des splits `train/valid/test` d'origine; aucune phrase de test ne doit apparaître dans les corpus de tokenisation ou d'entraînement.

### 3.2 Tokenisation (Texte Cible)
* **RF-06 :** Entraînement de trois tokeniseurs distincts via **SentencePiece** (algorithme Unigram ou BPE) sur les textes cibles (anglais, portugais, espagnol) du jeu d'entraînement.
* **RF-07 :** Taille du vocabulaire ciblée : Entre 1 000 et 5 000 sous-mots (Subwords) pour s'adapter à la faible quantité de données par couple de langues.
* **RF-08 :** Normalisation texte explicite et figée avant tokenisation : `NFKC`, suppression d'espaces redondants, normalisation de casse (stratégie documentée), traitement stable de la ponctuation et des chiffres.

### 3.3 Entraînement et Optimisation
* **RF-09 :** Calcul de la perte via une entropie croisée (Cross-Entropy Loss) standard avec lissage d'étiquette (*Label Smoothing* = 0.1).
* **RF-10 :** Implémentation d'un planificateur de taux d'apprentissage (Learning Rate Scheduler) avec une phase de montée linéaire (*Warmup*) et une décroissance inverse de la racine carrée (Inverse Square Root Decay).
* **RF-11 :** Stratégie de gel des poids (*Freezing*) : Possibilité de geler l'encodeur Pantagruel pendant les N premières étapes (ex: 5 000 à 10 000 updates) pour stabiliser le décodeur initialisé aléatoirement.
* **RF-12 :** Stabilité d'entraînement : clipping du gradient (ex: `max_norm=1.0`), `gradient_accumulation`, et entraînement mixte (`fp16`/`bf16`) pour atteindre un batch effectif reproductible sur GPU limité.

### 3.4 Inférence et Évaluation
* **RF-13 :** Implémentation d'une recherche par faisceau (**Beam Search Decoding**) avec une largeur de faisceau (*Beam Width*) de 5.
* **RF-14 :** Évaluation via la bibliothèque officielle **SacreBLEU** avec la signature standard pour garantir la reproductibilité et la comparaison équitable avec LeBenchmark 2.0.
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
* Scripts : `scripts/bootstrap.sh`, `scripts/0_preflight.py`, orchestration via `scripts/pipeline.py preflight`.
* Créer un environnement virtuel isolé (`conda` ou `venv` avec Python 3.10+).
* Installer les dépendances clés : `torch`, `transformers`, `speechbrain`, `sacrebleu`, `sentencepiece`, `tensorboard`.
* Installer les outils dev : `pip install -r requirements-dev.txt` puis `pre-commit install`.
* Valider la chaîne qualité : `ruff check .`, `ruff format --check .`, `pytest`.
* Configurer les accès au GPU (Vérification de `cuda.is_available()`).
* Télécharger les poids pré-entraînés du modèle Pantagruel sur Hugging Face (`PantagrueLLM/`).
* Fixer les seeds globales et préparer un template de configuration de run versionné.
* **Jalon go/no-go :** environnement reproductible validé (dépendances figées + script de seed unique + test GPU OK).

### Phase 2 : Ingestion et Tokenisation m-TEDx (Jours 3-5)
* Scripts : `scripts/1_download.py` (implémenté), `scripts/2_prepare.py`, `scripts/3_spm.py`.
* Télécharger les partitions m-TEDx via `python scripts/1_download.py` (défaut: `fr-en`; option multi-paires: `fr-en,fr-pt,fr-es`).
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
* Script : `scripts/4_train.py`.
* Implémenter la boucle d'entraînement principale conforme aux paramètres de *LeBenchmark 2.0* (Parcollet et al., 2024).
* Configurer l'optimiseur AdamW (`weight_decay=0.01`) et le scheduler de Learning Rate.
* Intégrer les mécanismes de régularisation : Dropout (0.1) et SpecAugment (masquage temporel et fréquentiel sur les features acoustiques).
* Configurer `gradient_accumulation`, AMP (`fp16`/`bf16`) et clipping du gradient.
* Lancer les scripts d'entraînement pour le premier couple (recommandation : `fr-es` car il fait 25h, idéal pour un prototypage rapide).
* Surveiller la convergence de la courbe de perte (Loss) et du `BLEU dev` sur TensorBoard ou Weights & Biases.
* **Jalon go/no-go :** perte en baisse + `BLEU dev` supérieur à la baseline minimale sur `fr-es`.

### Phase 5 : Inférence, Génération et Évaluation (Jours 13-15)
* Scripts : `scripts/5_evaluate.py`, `scripts/6_infer.py`.
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
   * *Atténuation :* Figer la commande canonique d'évaluation, stocker la signature SacreBLEU et appliquer un protocole texte identique sur toutes les expériences.