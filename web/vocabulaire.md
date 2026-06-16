# Vocabulaire S3T — termes techniques, abréviations et codes

Ce document recense le **langage du projet** S3T (Speech Translation replication) : acronymes, identifiants, modes, chemins et notions scientifiques. Chaque entrée vise une **définition en français clair**, puis le **nom exact** tel qu’il apparaît dans le code ou la doc.

**Convention des entrées :**

| Champ | Signification |
|-------|----------------|
| **En clair** | Formulation accessible, sans jargon préalable |
| **Identifiant** | Valeur exacte dans le dépôt (CLI, YAML, CSV, chemins) |
| **Où** | Fichiers ou étapes concernés |

---

## Sommaire

1. [Projet et organisation](#1-projet-et-organisation)
2. [Abréviations et acronymes](#2-abréviations-et-acronymes)
3. [Corpus et données](#3-corpus-et-données)
4. [Découpage audio (segmentation)](#4-découpage-audio-segmentation)
5. [Manifestes et colonnes TSV](#5-manifestes-et-colonnes-tsv)
6. [Modèles, encodeurs et paradigmes](#6-modèles-encodeurs-et-paradigmes)
7. [Entraînement et inférence](#7-entraînement-et-inférence)
8. [Évaluation et métriques](#8-évaluation-et-métriques)
9. [Variantes du pipeline (dossiers 1–5)](#9-variantes-du-pipeline-dossiers-15)
10. [Étapes du pipeline (0–6) et commandes CLI](#10-étapes-du-pipeline-06-et-commandes-cli)
11. [Configurations YAML](#11-configurations-yaml)
12. [Runs, artifacts et suivi](#12-runs-artifacts-et-suivi)
13. [Identifiants de runs (exemples)](#13-identifiants-de-runs-exemples)
14. [Chemins et répertoires](#14-chemins-et-répertoires)
15. [Outils, bibliothèques et services externes](#15-outils-bibliothèques-et-services-externes)
16. [Matériel et exécution](#16-matériel-et-exécution)
17. [Documents et références internes](#17-documents-et-références-internes)

---

## 1. Projet et organisation

### S3T

| | |
|---|---|
| **En clair** | Nom du dépôt : projet de réplication des expériences de traduction de la parole (ST) autour de l’article Pantagruel, avec plusieurs variantes comparables sur les mêmes données. |
| **Identifiant** | `S3T` (racine du repo GETALP) |
| **Où** | README, tous les pipelines |

### Variante (numérotée 1 à 5)

| | |
|---|---|
| **En clair** | Une piste expérimentale distincte après la préparation commune des données (étapes 0–2) : baseline Transformer, speechLLM, API Gemini, cascade ASR→MT, ou encodeur multimodal Speech_Text. |
| **Identifiant** | Dossiers `1_Transformer/`, `2_speechLLM/`, `3_Gemini/`, `4_cascade/`, `5_Pantagruel_multimodal/` |
| **Où** | README § tableau variantes |

### scripts_communs

| | |
|---|---|
| **En clair** | Code partagé par toutes les variantes : préparation des données (0–2), utilitaires ST (`st_common.py`), protocole d’éval, bootstrap. |
| **Identifiant** | Répertoire `scripts_communs/` |
| **Où** | Étapes 0–2, `pipeline.py` commun |

### pipeline (routeur)

| | |
|---|---|
| **En clair** | Programme en ligne de commande qui enchaîne les étapes (`preflight`, `download`, `prepare`, `train`, etc.) sans contenir la logique métier elle-même. |
| **Identifiant** | `scripts_communs/pipeline.py`, `1_Transformer/pipeline.py`, `2_speechLLM/pipeline.py`, … |
| **Où** | Chaque dossier de variante + commun |

### PRD

| | |
|---|---|
| **En clair** | Document d’exigences produit : objectifs scientifiques, architecture, hyperparamètres cibles, contrat des fichiers de run. |
| **Identifiant** | [docs/PRD.md](PRD.md) |
| **Où** | Référence obligatoire avant commit si le comportement change |

### Temps A / Temps B (présentation)

| | |
|---|---|
| **En clair** | Temps A = alignement strict sur le protocole article (utterance, SacreBLEU, objectif beam 5). Temps B = extensions et ablations (sentence_like, greedy intermédiaire, optimisations). |
| **Identifiant** | Termes narratifs dans `presentation_fr_en_pantagruel.md` |
| **Où** | Slides, rapport |

---

## 2. Abréviations et acronymes

| Abrév. | En clair | Identifiant / contexte S3T |
|--------|----------|----------------------------|
| **ST** | Traduction de la parole : l’audio source est traduit directement en texte dans une langue cible (ici fr→en), sans étape ASR séparée pour la baseline E2E. | Speech Translation ; tâche principale du PRD |
| **S2T** | Synonyme courant en anglais pour la même tâche (*speech-to-text translation*). | Parfois dans la littérature ; S3T utilise surtout **ST** |
| **E2E** | Bout en bout : un seul modèle (ou une seule chaîne entraînée) de l’audio à la traduction. | Opposé à **cascade** |
| **SSL** | Apprentissage auto-supervisé sur de grandes quantités de parole/texte sans transcriptions alignées pour le pré-entraînement. | Encodeurs Pantagruel, LeBenchmark |
| **JEPA** | Prédiction dans l’espace latent (pas de reconstruction du signal brut) ; cadre théorique (LeCun, 2022) derrière data2vec et Pantagruel. | [Entrée détaillée §6](#jepa-joint-embedding-predictive-architecture) |
| **ASR** | Reconnaissance vocale : audio → transcription dans la même langue (ici français). | Étape 1 de la cascade (`4_cascade`) |
| **MT** | Traduction automatique texte→texte (ici fr→en via Marian). | Étape 2 de la cascade |
| **SPM** | SentencePiece : découpe le texte anglais en sous-mots (tokens) pour le décodeur Transformer. | Étape `3_spm`, variantes 1 et 5 |
| **BLEU** | Score de similarité entre traductions produites et références humaines (plus haut = souvent mieux, avec réserves). | Métrique centrale |
| **SacreBLEU** | Implémentation standardisée de BLEU (signature reproductible). | Paquet `sacrebleu`, fichiers `sacrebleu_*.txt` |
| **chrF** | Métrique complémentaire (caractères n-grammes). | `metrics.json`, secondaire |
| **TER** | Taux d’erreur de traduction (éditions). | `metrics.json`, secondaire |
| **HF** | Hugging Face : plateforme de modèles (`PantagrueLLM/...`). | `transformers`, téléchargement poids |
| **API** | Appel réseau à un modèle hébergé (Gemini), sans entraînement local. | `3_Gemini`, `GEMINI_API_KEY` |
| **GPU / VRAM** | Carte graphique / mémoire vidéo pour l’entraînement et l’inférence. | Preflight, configs `amp_dtype` |
| **AMP** | Précision mixte (fp16 / bf16) pour accélérer l’entraînement. | `train.amp_dtype` |
| **CLI** | Interface ligne de commande (`python ... pipeline.py train`). | Tous les `pipeline.py` |
| **YAML** | Fichier de configuration lisible (hyperparamètres d’un run). | `configs/fr-en/*.yaml` |
| **TSV** | Tableau texte à tabulations (manifests, exports relecture). | `train.tsv`, `valid.tsv`, `test.tsv` |
| **WAV** | Fichier audio non compressé (ici 16 kHz, mono, PCM 16 bits). | `datasets/processed/` |
| **NFKC** | Normalisation Unicode des textes avant tokenisation SPM. | `2_prepare`, PRD RF-08 |
| **MFA** | Montreal Forced Aligner (alignement phonétique, corpus oralité externe). | `docs/corpus_oralite_externe.md` |
| **GETALP** | Laboratoire / contexte d’accueil du stage (non code). | Mentions rapport |
| **INA** | Partenaire des corpus français (pré-entraînement Pantagruel Large). | Doc 14k / 114k |
| **RF-xx** | Exigence fonctionnelle numérotée dans le PRD. | PRD §3 |
| **BL-xx / AB-xx** | Baseline / ablation numérotée dans le PRD. | PRD §6 |

---

## 3. Corpus et données

### m-TEDx (multilingual TEDx)

| | |
|---|---|
| **En clair** | Corpus d’exposés TED multilingues ; S3T utilise la branche française comme entrée audio et des traductions anglais / portugais / espagnol selon la paire. |
| **Identifiant** | Corpus **OpenSLR-100** ; paires `fr-en`, `fr-pt`, `fr-es` |
| **Où** | `1_download.py`, PRD §1.3 |

### Paire de langues (lang pair)

| | |
|---|---|
| **En clair** | Couple langue source → langue cible du benchmark (ex. parole française, texte anglais). |
| **Identifiant** | `fr-en`, `fr-pt`, `fr-es` ; CLI `--langpair` / `--langpairs` |
| **Où** | Manifests, `runs/<langpair>/` |

### OpenSLR-100

| | |
|---|---|
| **En clair** | Identifiant du jeu de données sur le portail OpenSLR utilisé pour télécharger m-TEDx. |
| **Identifiant** | Référencé dans le PRD et `1_download.py` |
| **Où** | Étape download |

### Split (train / valid / test)

| | |
|---|---|
| **En clair** | Trois parties disjointes du corpus : entraînement, validation (choix du meilleur modèle), test final (score de rapport). |
| **Identifiant** | Fichiers `train.tsv`, `valid.tsv`, `test.tsv` |
| **Où** | Après `2_prepare` ; **valid** est nommé **dev** dans les artifacts d’éval (`dev_predictions.txt`) |

### Anti-fuite (data leak)

| | |
|---|---|
| **En clair** | Vérification qu’aucun segment de test n’apparaît dans l’entraînement ou la construction du tokenizer. |
| **Identifiant** | `detect_leaks`, `--fail-on-leak` dans `2_prepare.py` |
| **Où** | PRD RF-05, protocole évaluation §2.3 |

### Corpus oralité pluriTAL (externe)

| | |
|---|---|
| **En clair** | Audios hors m-TEDx pour tests qualitatifs (oralité, style lu) ; pas de score SacreBLEU (pas de référence anglaise). |
| **Identifiant** | `datasets/external/oralite_fr/` |
| **Où** | [corpus_oralite_externe.md](corpus_oralite_externe.md) |

---

## 4. Découpage audio (segmentation)

> **Idée centrale :** « segmentation » = **comment on découpe** les enregistrements en exemples d’entraînement/évaluation. Ce n’est **pas** la taille de l’encodeur (1k / 14k).

### Segmentation (général)

| | |
|---|---|
| **En clair** | Règle qui définit un exemple = un fichier audio + une traduction de référence. |
| **Identifiant** | Champ `segment_mode` ; option CLI `--segment-mode` |
| **Où** | `2_prepare.py`, `experiments_tracking.csv`, protocole évaluation |

### utterance (mode d’origine)

| | |
|---|---|
| **En clair** | Chaque prise telle que fournie par m-TEDx : segments relativement courts, alignés sur l’annotation du corpus. C’est le découpage de l’article Pantagruel (Table 8). |
| **Identifiant** | `--segment-mode utterance` (défaut) |
| **Où** | `datasets/manifests/<pair>/`, `datasets/processed/<pair>/` |

### sentence_like (mode regroupé S3T)

| | |
|---|---|
| **En clair** | Fusion de prises consécutives du même exposé (même intervenant si possible) pour former des blocs plus longs (10 s cible, max 15 s), en coupant de préférence après `. ? !`. Objectif : segments plus proches d’une phrase complète, souvent plus stables à l’entraînement. |
| **Identifiant** | `--segment-mode sentence_like` |
| **Où** | `datasets/manifests_sentence/<pair>/`, `datasets/processed_sentence/<pair>/` |
| **Paramètres** | `--sentence-target-duration`, `--sentence-max-duration`, `--sentence-require-punctuation` |

### talk (exposé TED)

| | |
|---|---|
| **En clair** | Un discours TEDx ; la fusion `sentence_like` ne mélange pas deux talks différents. |
| **Identifiant** | `talk_id` en interne dans `2_prepare.py` |
| **Où** | Logique de fusion |

### Règle d’incohérence train/eval

| | |
|---|---|
| **En clair** | Un modèle entraîné sur un découpage ne doit pas être évalué sur l’autre (scores non comparables). |
| **Identifiant** | Documenté protocole utterance § règles, protocole évaluation §2.2 |
| **Où** | Tous les runs |

---

## 5. Manifestes et colonnes TSV

### Manifest

| | |
|---|---|
| **En clair** | Table listant chaque exemple : identifiant, chemin audio, texte cible, métadonnées. |
| **Identifiant** | `train.tsv`, `valid.tsv`, `test.tsv` |
| **Où** | `datasets/manifests*/<langpair>/` |

### Colonnes standard (`2_prepare`)

| Colonne | En clair |
|---------|----------|
| `id` | Identifiant unique du segment |
| `audio` | Chemin vers le WAV 16 kHz |
| `n_frames` | Nombre d’échantillons audio (mono) |
| `tgt_text` | Traduction de référence (langue cible, ex. anglais) |
| `speaker` | Identifiant locuteur |
| `tgt_lang` | Code langue cible (`en`, `pt`, `es`) |
| `src_text` | Transcription source française (si présente) |
| `src_lang` | Code langue source (`fr`) |

### dev vs valid

| | |
|---|---|
| **En clair** | Même split : le corpus s’appelle `valid.tsv`, mais les fichiers d’évaluation portent le préfixe `dev_` (`dev_predictions.txt`, `sacrebleu_dev.txt`). |
| **Identifiant** | `valid` (manifest) ↔ `dev` (eval) |
| **Où** | `eval_protocol.py`, tous les `evaluate` |

---

## 6. Modèles, encodeurs et paradigmes

### Pantagruel (article / famille de modèles)

| | |
|---|---|
| **En clair** | Famille d’encodeurs français auto-supervisés pour texte et parole ([JEPA](#jepa-joint-embedding-predictive-architecture) / data2vec 2.0), utilisée comme « oreille » du système ST. |
| **Identifiant** | Article 2026 ; checkpoints HF `PantagrueLLM/...` |
| **Où** | PRD, README, configs `model.encoder_name` |

### JEPA (Joint Embedding Predictive Architecture)

| | |
|---|---|
| **En clair** | Famille d’approches d’[apprentissage auto-supervisé](#2-abréviations-et-acronymes) où le modèle **ne reconstruit pas** l’entrée brute (audio, image, texte) : il **prédit des représentations internes** (vecteurs continus) dans un espace latent. L’idée, formalisée par LeCun (2022), est d’apprendre la **structure** du signal plutôt que sa surface. |
| **Identifiant** | *Joint Embedding Predictive Architecture* ; implémentation Pantagruel : **data2vec 2.0** (Baevski et al., 2023) |
| **Référence** | LeCun, Y. (2022). *A path towards autonomous machine intelligence* (version 0.9.2, 2022-06-27). *Open Review*, 62(1), 1–62. |
| **Où** | Article Pantagruel §3 ; encodeurs `PantagrueLLM/speech-*` utilisés en variantes 1, 2 et 5 |

**Mécanisme (article Pantagruel, figure 1) :**

1. L’audio (ou le texte) passe dans un **encodeur professeur** qui voit l’entrée **complète**.
2. Un **encodeur étudiant** ne voit qu’une partie de l’entrée (le reste est **masqué**).
3. Un petit **décodeur** doit prédire, pour les zones masquées, les représentations que le professeur a calculées.
4. La perte mesure l’écart (distance L2) entre prédiction et cible — pas entre sortie et signal d’origine.
5. Les poids du professeur ne s’entraînent pas directement : ils suivent une **moyenne mobile exponentielle** (EMA) de ceux de l’étudiant, ce qui stabilise l’apprentissage.

**Contraste avec d’autres pré-entraînements :**

| Approche | Ce que le modèle prédit | Exemple |
|----------|-------------------------|---------|
| **BERT / MLM** | Tokens textuels discrets masqués | CamemBERT, FlauBERT |
| **wav2vec 2.0 / HuBERT** | Unités acoustiques quantifiées (pseudo-phonèmes) | LeBenchmark 2.0 |
| **JEPA / data2vec** | Vecteurs continus contextualisés (espace latent) | **Pantagruel** (parole) |

Pour la **parole**, Pantagruel utilise une perte purement JEPA (prédiction dans l’espace latent). Pour le **texte**, l’article combine JEPA et MLM (prédiction de tokens masqués en complément), car le texte discret bénéficie encore d’une composante token-level.

**Lien avec S3T :** nous n’entraînons pas l’encodeur Pantagruel — nous **chargeons** un checkpoint déjà pré-entraîné (1k, 14k ou 114k h de parole française) et l’utilisons comme point de départ pour la traduction. La qualité de ces représentations JEPA conditionne en grande partie les scores ST.

**Cousins cités dans l’article :** I-JEPA et V-JEPA (image / vidéo), A-JEPA et WavJEPA (audio).

### Table 8 (Pantagruel)

| | |
|---|---|
| **En clair** | Tableau de résultats de l’article (ST fr→en sur m-TEDx, utterance) ; référence 17,5 BLEU pour Pantagruel-B-1k. |
| **Identifiant** | « Table 8 » (pas un fichier du repo) |
| **Où** | README, `protocole_utterance_pantagruel.md` |

### Encodeur (acoustique / SSL)

| | |
|---|---|
| **En clair** | Réseau qui transforme l’audio en représentations internes denses (vecteurs dans le temps). |
| **Identifiant** | `model.encoder_name`, branche `encoder.*` dans les checkpoints |
| **Où** | `st_common.S3TModel`, speechLLM |

### Décodeur (Transformer 6 couches)

| | |
|---|---|
| **En clair** | Réseau qui génère le texte anglais token par token en regardant les représentations audio (attention croisée). |
| **Identifiant** | `decoder_layers: 6`, `1_Transformer/4_train.py` |
| **Où** | Variantes 1 et 5 |

### B-1k / L-14k / L-114k (échelle de pré-entraînement)

| | |
|---|---|
| **En clair** | Volume d’heures de parole française utilisé pour le pré-entraînement Pantagruel (pas la durée d’un clip m-TEDx). 1k ≈ mille heures (Base) ; 14k / 114k = variantes Large. |
| **Identifiant** | HF : `speech-base-1K`, `speech-large-14K`, `speech-large-114K` |
| **Où** | Configs `*_large_14k.yaml`, protocole utterance |

### speech-base-1K (checkpoint HF)

| | |
|---|---|
| **En clair** | Encodeur parole seule Pantagruel Base (1k h pré-train), sortie 768 dim — référence principale S3T aujourd’hui. |
| **Identifiant** | `PantagrueLLM/speech-base-1K` |
| **Où** | `base.yaml`, speechLLM `b1.yaml` |

### Speech_Text / speech_text (multimodal)

| | |
|---|---|
| **En clair** | Checkpoint Pantagruel parole + texte (pré-entraînement multimodal), testé en variante 5 avec le même décodeur ST. |
| **Identifiant** | `PantagrueLLM/Speech_Text_Base_fr_1K_4GB`, `model.encoder_api: speech_text` |
| **Où** | `5_Pantagruel_multimodal/` |

### LeBenchmark

| | |
|---|---|
| **En clair** | Ligne de base concurrente de l’article (famille wav2vec2 / data2vec française), distincte des variantes S3T. |
| **Identifiant** | Cité Table 8 (14 BLEU wav2vec B-1k) |
| **Où** | README baselines article |

### FlauBERT / CamemBERT

| | |
|---|---|
| **En clair** | Baselines texte de l’article Pantagruel (pas utilisées comme pipelines ST dans S3T). |
| **Identifiant** | Noms de modèles dans le papier |
| **Où** | README § contexte article |

### Gelé (frozen) / dégelé (unfrozen)

| | |
|---|---|
| **En clair** | Gelé = les poids ne sont pas mis à jour à l’entraînement ; dégelé = fine-tuning autorisé. |
| **Identifiant** | `freeze_encoder_updates`, `freeze_encoder: false` (speechLLM), `trainable_state` |
| **Où** | PRD RF-11, runs `run_002` vs `run_005` speechLLM |

### Projecteur (speechLLM)

| | |
|---|---|
| **En clair** | Petite couche linéaire (souvent Linear → ReLU → Linear) qui adapte les représentations audio à l’espace du LLM ; seul module entraîné en configuration B1 stricte. |
| **Identifiant** | `SpeechLLMModel`, checkpoint `trainable_state` |
| **Où** | `2_speechLLM/` |

### LLM (grand modèle de langue)

| | |
|---|---|
| **En clair** | Modèle textuel génératif (ex. Phi-2) qui produit l’anglais à partir des embeddings injectés. |
| **Identifiant** | `model.llm_name` (ex. `microsoft/phi-2`) |
| **Où** | speechLLM configs |

### B1 / B2 (speechLLM)

| | |
|---|---|
| **En clair** | B1 = seul le projecteur s’entraîne (encodeur + LLM gelés) — implémenté. B2 = extension future (plus de modules entraînables). |
| **Identifiant** | `b1.yaml`, plan `plan_migration_speechllm.md` |
| **Où** | `2_speechLLM/` |

### USER / ASSISTANT (format prompt)

| | |
|---|---|
| **En clair** | Gabarit conversation : la partie utilisateur contient l’audio (embeddings) + consigne ; la partie assistant contient la traduction à apprendre ou générer. |
| **Identifiant** | Chaînes `USER:` / `ASSISTANT:` dans le collate speechLLM |
| **Où** | `2_speechLLM/README.md`, plan migration |

### Downsampling (k=5)

| | |
|---|---|
| **En clair** | Réduction du nombre de frames audio avant le projecteur (ex. concaténer 5 pas de temps), comme dans SLAM-ASR. |
| **Identifiant** | Paramètre modèle speechLLM (défaut k=5) |
| **Où** | `plan_migration_speechllm.md` |

### Cascade ASR→MT

| | |
|---|---|
| **En clair** | Deux modèles en série : Whisper transcrit le français, Marian traduit le texte en anglais. |
| **Identifiant** | `4_cascade`, configs `cascade.yaml` |
| **Où** | Whisper `openai/whisper-large-v3`, Marian `Helsinki-NLP/opus-mt-fr-en` |

### Gemini ST (baseline API)

| | |
|---|---|
| **En clair** | Google Gemini reçoit l’audio (ou le fichier) et renvoie la traduction anglaise ; pas d’entraînement local. |
| **Identifiant** | `gemini-2.5-flash`, `gemini-3.5-flash` dans `model.gemini_id` |
| **Où** | `3_Gemini/`, variable d’environnement `GEMINI_API_KEY` |

### trust_remote_code

| | |
|---|---|
| **En clair** | Autorisation Hugging Face d’exécuter du code custom fourni avec le checkpoint Pantagruel. |
| **Identifiant** | `trust_remote_code: true` dans YAML |
| **Où** | Chargement encodeurs |

---

## 7. Entraînement et inférence

### Update (pas d’epoch)

| | |
|---|---|
| **En clair** | Une itération d’optimisation (un pas de gradient, éventuellement après accumulation). S3T raisonne souvent en updates (ex. 80k) plutôt qu’en epochs. |
| **Identifiant** | `max_updates`, `eval_every_updates` |
| **Où** | YAML `train.*`, logs `train.log` |

### Warmup (montée en charge LR)

| | |
|---|---|
| **En clair** | Phase où le taux d’apprentissage augmente progressivement au début pour stabiliser l’entraînement. |
| **Identifiant** | `warmup_updates` |
| **Où** | Configs `base_utterance.yaml` (ex. 4000) |

### freeze_encoder_updates

| | |
|---|---|
| **En clair** | Nombre d’updates pendant lesquelles l’encodeur reste gelé avant éventuel fine-tuning joint (baseline ST). |
| **Identifiant** | `train.freeze_encoder_updates` (ex. 1000) |
| **Où** | `1_Transformer` configs utterance |

### Label smoothing

| | |
|---|---|
| **En clair** | Régularisation qui adoucit la cible one-hot pour éviter la sur-confiance du décodeur. |
| **Identifiant** | `label_smoothing: 0.1` |
| **Où** | PRD RF-09 |

### Gradient accumulation / clipping

| | |
|---|---|
| **En clair** | Accumulation = simuler un plus grand batch en plusieurs passes ; clipping = plafonner la norme du gradient pour éviter les explosions. |
| **Identifiant** | `gradient_accumulation`, `gradient_clip_norm` |
| **Où** | YAML train |

### Checkpoint

| | |
|---|---|
| **En clair** | Fichier sauvegardant les poids du modèle à un instant donné. |
| **Identifiant** | `checkpoints/best.pt`, `checkpoints/last.pt` |
| **Où** | `runs/<langpair>/<run_id>/` |

### best.pt vs last.pt

| | |
|---|---|
| **En clair** | best = meilleur score dev (valid) ; last = dernier état en fin d’entraînement. L’évaluation officielle utilise en général best. |
| **Identifiant** | Noms de fichiers fixes |
| **Où** | Toutes variantes entraînables |

### trainable_state (speechLLM)

| | |
|---|---|
| **En clair** | Checkpoint ne contenant que les poids entraînables (projecteur, éventuellement encodeur si dégelé). |
| **Identifiant** | `checkpoints/best.pt` format speechLLM |
| **Où** | `2_speechLLM/evaluate.py` |

### Inférence (infer)

| | |
|---|---|
| **En clair** | Traduire de nouveaux fichiers audio (hors bench ou corpus externe). |
| **Identifiant** | Subcommand `infer`, `6_infer.py` |
| **Où** | Chaque variante |

### dry-run

| | |
|---|---|
| **En clair** | Exécution à blanc : affiche le plan sans charger les modèles ni écrire les résultats finaux. |
| **Identifiant** | `--dry-run` |
| **Où** | CLI pipelines |

### smoke test / --limit

| | |
|---|---|
| **En clair** | Test court sur quelques segments (`--limit 5`) pour valider que le code tourne. |
| **Identifiant** | Scripts `smoke_*`, `--limit` |
| **Où** | Cascade, bench, encodeurs 14k/114k |

---

## 8. Évaluation et métriques

### SacreBLEU corpus BLEU

| | |
|---|---|
| **En clair** | Score BLEU calculé sur tout le jeu valid ou test d’un coup, avec paramètres standard enregistrés (signature). |
| **Identifiant** | Fichiers `eval/sacrebleu_dev.txt`, `sacrebleu_test.txt` |
| **Où** | Protocole `2026-06-02-v1` |

### Signature SacreBLEU

| | |
|---|---|
| **En clair** | Chaîne qui décrit exactement comment le BLEU a été calculé (version outil, tokenisation). |
| **Identifiant** | Dans `sacrebleu_*.txt` et `protocol.json` |
| **Où** | `eval_protocol.py` |

### BLEU dev / BLEU test

| | |
|---|---|
| **En clair** | Score sur le split validation (sélection modèle) vs test (rapport final). |
| **Identifiant** | Colonnes `bleu_dev`, `bleu_test` ; parfois noté SacreBLEU dev/test |
| **Où** | README tableaux résultats, CSV tracking |

### Protocole d’évaluation figé

| | |
|---|---|
| **En clair** | Ensemble de règles immuable (décodage, métrique, ordre des lignes) pour comparer les variantes ; toute modification = nouvelle version. |
| **Identifiant** | Version `2026-06-02-v1`, module `eval_protocol.py` |
| **Où** | [protocole_evaluation.md](protocole_evaluation.md), `eval/protocol.json` |

### protocol.json

| | |
|---|---|
| **En clair** | Fiche JSON par run listant version de protocole, décodage réel, version sacrebleu. |
| **Identifiant** | `runs/.../eval/protocol.json` |
| **Où** | Après chaque `evaluate` conforme |

### Greedy decoding

| | |
|---|---|
| **En clair** | À chaque pas, choisir le token le plus probable (pas d’exploration de plusieurs hypothèses). |
| **Identifiant** | Implémentation ST v1 (beam non appliqué) |
| **Où** | Protocole §4.1 ; écart documenté vs papier beam 5 |

### Beam search (beam 5)

| | |
|---|---|
| **En clair** | Garder les 5 meilleures suites de tokens en parallèle à la génération ; souvent meilleur BLEU, plus lent. |
| **Identifiant** | `decode.beam_size: 5` (YAML ST) ; objectif papier |
| **Où** | PRD ; **non** appliqué en code ST v1 |

### max_len_b / max_new_tokens

| | |
|---|---|
| **En clair** | Longueur maximale de la traduction générée (tokens). |
| **Identifiant** | ST : `max_len_b` (128) ; speechLLM v1 figé : `max_new_tokens=48` |
| **Où** | Protocole §4 |

### Hypothèse / référence

| | |
|---|---|
| **En clair** | Hypothèse = sortie du modèle ; référence = traduction humaine du manifest (`tgt_text`). |
| **Identifiant** | `*_predictions.txt` vs colonne `tgt_text` |
| **Où** | Évaluation |

### Détokenisation SPM

| | |
|---|---|
| **En clair** | Reconstituer une phrase à partir des tokens SentencePiece (baseline ST uniquement à l’éval). |
| **Identifiant** | Pipeline `5_evaluate.py` |
| **Où** | Variante 1 ; speechLLM/Gemini/cascade = texte brut |

### bench_evaluate_variants.sh

| | |
|---|---|
| **En clair** | Script qui lance les évaluations de plusieurs variantes (sans ré-entraîner), sur `sentence_like` ou `utterance`. |
| **Identifiant** | `scripts/bench_evaluate_variants.sh [utterance]` |
| **Où** | Protocole §8 |

---

## 9. Variantes du pipeline (dossiers 1–5)

| # | Dossier | En clair | Identifiant pipeline (tracking) |
|---|---------|----------|----------------------------------|
| — | `scripts_communs/` | Données + utilitaires communs | — |
| 1 | `1_Transformer/` | ST end-to-end : Pantagruel + décodeur Transformer + SPM | `transformer` / baseline ST |
| 2 | `2_speechLLM/` | Projecteur + LLM gelé (SLAM-ASR style) | `speechllm_b1` |
| 3 | `3_Gemini/` | API Gemini audio→anglais | `gemini_st` |
| 4 | `4_cascade/` | Whisper FR puis Marian fr→en | `cascade` |
| 5 | `5_Pantagruel_multimodal/` | Encodeur Speech_Text + même décodeur que 1 | `pantagruel_multimodal` (selon run) |

### SLAM-ASR / « embarrassingly simple »

| | |
|---|---|
| **En clair** | Article de référence pour speechLLM : entraîner uniquement un adaptateur entre encodeur gelé et LLM gelé. |
| **Identifiant** | `2_speechLLM/embarrassingly_simple_approach.pdf` |
| **Où** | Plan migration, README speechLLM |

---

## 10. Étapes du pipeline (0–6) et commandes CLI

| Étape | Module | En clair | Subcommand |
|-------|--------|----------|------------|
| Bootstrap | `bootstrap.sh` | Crée le venv, installe les dépendances | — |
| 0 | `0_preflight.py` | Vérifie Linux, CUDA, versions | `preflight` |
| 1 | `1_download.py` | Télécharge m-TEDx | `download` |
| 2 | `2_prepare.py` | WAV 16 kHz + manifests + normalisation texte | `prepare` |
| 3 | `3_spm.py` | Entraîne SentencePiece sur textes cibles | `spm` |
| 4 | `4_train.py` | Entraîne ST Transformer | `train` |
| 5 | `5_evaluate.py` | Décode + SacreBLEU | `evaluate` |
| 6 | `6_infer.py` | Inférence WAV arbitraire | `infer` |
| — | `pipeline.py` | Enchaîne des étapes | `run` |

**Options CLI fréquentes :**

| Option | En clair |
|--------|----------|
| `--langpair` / `--langpairs` | Paire(s) de langues |
| `--run-id` | Nom du dossier d’expérience sous `runs/` |
| `--config` | Fichier YAML d’hyperparamètres |
| `--from-stage` / `--to-stage` | Borne d’enchaînement dans `run` |
| `--verbose` / `-v` | Logs détaillés |
| `--resume` | Reprendre une étape interrompue |
| `--verify-only` | Contrôler les manifests sans régénérer |

---

## 11. Configurations YAML

Sections communes (`1_Transformer/configs/fr-en/base*.yaml`, analogues speechLLM) :

| Clé | En clair |
|-----|----------|
| `experiment.name` | Libellé humain de l’expérience |
| `experiment.lang_pair` | Paire `fr-en` |
| `experiment.output_dir` | Dossier de sortie (souvent `runs/fr-en/<run_id>`) |
| `experiment.seed` | Graine aléatoire pour reproductibilité |
| `experiment.deterministic` | Mode déterministe CUDA (si possible) |
| `data.train_manifest` / `valid_manifest` / `test_manifest` | Chemins TSV |
| `data.spm_model` | Modèle SentencePiece (baseline ST) |
| `data.sample_rate` | Fréquence audio (16000 Hz) |
| `model.encoder_name` | ID Hugging Face de l’encodeur |
| `model.decoder_layers` / `decoder_heads` / `hidden_dim` | Architecture décodeur |
| `model.llm_name` | ID LLM (speechLLM) |
| `model.gemini_id` | ID modèle Gemini |
| `model.encoder_api` | `speech_text` pour variante multimodale |
| `train.max_updates` | Nombre max d’updates |
| `train.batch_size` / `gradient_accumulation` | Batch effectif |
| `train.amp_dtype` | `fp16` ou `bf16` |
| `decode.beam_size` | Largeur de faisceau (cible papier ; voir protocole pour implémentation réelle) |
| `decode.max_len_b` | Longueur max génération ST |
| `prompt.template` | Consigne speechLLM / Gemini |
| `pricing.*` | Tarifs API pour estimation coût Gemini |

---

## 12. Runs, artifacts et suivi

### run_id

| | |
|---|---|
| **En clair** | Nom unique d’une expérience (dossier sous `runs/<langpair>/`). |
| **Identifiant** | Ex. `run_002_speechllm_b1_sentence_long` |
| **Où** | CLI `--run-id`, CSV tracking |

### Convention de nommage (PRD)

| | |
|---|---|
| **En clair** | Schéma recommandé pour encoder seed, gel, vocab, beam dans le nom. |
| **Identifiant** | `run_<id>_<langpair>_seed<seed>_freeze<updates>_vocab<size>_beam<n>` |
| **Où** | PRD §8.1 |

### Contrat d’artifacts (par run)

| Fichier / dossier | En clair |
|-------------------|----------|
| `config.yaml` | Copie figée de la config |
| `train.log` | Journal JSONL d’entraînement |
| `metrics.json` | Scores et métadonnées |
| `checkpoints/` | Poids sauvegardés |
| `eval/dev_predictions.txt` | Traductions sur valid |
| `eval/test_predictions.txt` | Traductions sur test |
| `eval/sacrebleu_*.txt` | BLEU + signature |
| `eval/protocol.json` | Protocole appliqué |
| `eval/metrics.json` | Détails decode, coût Gemini, échecs cascade |

### experiments_tracking.csv

| | |
|---|---|
| **En clair** | Tableau de bord central : une ligne par run, BLEU, segmentation, coûts API, statut. |
| **Identifiant** | `runs/experiments_tracking.csv` |
| **Où** | Mis à jour par `update_experiments_tracking.py` |

### Colonnes CSV (suivi)

| Colonne | En clair |
|---------|----------|
| `run_id` | Identifiant d’expérience |
| `lang_pair` | ex. `fr-en` |
| `pipeline` | Variante (`speechllm_b1`, `gemini_st`, …) |
| `segment_mode` | `utterance` ou `sentence_like` |
| `seed` | Graine |
| `freeze_updates` | Updates de gel encodeur (ST) |
| `vocab_size` | Taille vocab SPM |
| `beam` | Paramètre décodage enregistré |
| `bleu_dev` / `bleu_test` | Scores principaux |
| `chrf_*` / `ter_*` | Métriques secondaires |
| `train_hours` / `gpu_hours` | Temps machine |
| `gemini_cost_usd` | Coût estimé API |
| `git_commit` | Commit du dépôt au moment du run |
| `status` | État (succès, échec, en cours) |
| `notes` | Commentaires libres (découpage, signature, etc.) |

### prepare_&lt;pair&gt;.json

| | |
|---|---|
| **En clair** | Rapport de l’étape prepare : mode de segmentation, statistiques de fusion. |
| **Identifiant** | `artifacts/prepare_<langpair>.json` |
| **Où** | Après `2_prepare` |

### download_manifest.json

| | |
|---|---|
| **En clair** | Trace des fichiers téléchargés (étape 1). |
| **Identifiant** | `artifacts/download_manifest.json` |
| **Où** | `1_download.py` |

### EVAL_PROTOCOL_VERSION

| | |
|---|---|
| **En clair** | Constante code de la version du protocole (ex. `2026-06-02-v1`). |
| **Identifiant** | `scripts_communs/eval_protocol.py` |
| **Où** | Tout changement de règle d’éval → incrément + ré-éval |

---

## 13. Identifiants de runs (exemples)

> Les `run_id` sont **historiques** ; la lecture portfolio doit toujours préciser **segment_mode** et **variante**.

| run_id (extrait) | En clair |
|------------------|----------|
| `run_001_transformer_baseline_sentence_like` | Baseline ST Transformer, découpage regroupé |
| `run_002_transformer_baseline_utterance` | Baseline ST, découpage article (Table 8) |
| `run_002_speechllm_b1_sentence_long` | speechLLM B1, encodeur gelé, sentence_like |
| `run_005_speechllm_b1_sentence_long_unfreeze_encoder` | speechLLM B1, encodeur fine-tuné |
| `run_001_gemini_flash_sentence_like_v2` | Gemini 2.5 Flash, sentence_like |
| `run_003_gemini_35_flash_sentence_like` | Gemini 3.5 Flash (prévu) |
| `run_001_cascade_sentence_like` | Cascade complète sentence_like |
| `run_001_cascade_utterance` | Cascade utterance |
| `run_003_speechllm_b1_utterance_long` | speechLLM utterance |
| `run_010_transformer_baseline_utterance_large_14k` | ST L-14k — **échec** (collapse, tour, juin 2026) |
| `run_014_transformer_baseline_utterance_large_14k_v2` | ST L-14k retry (gel 5k, early stop) |
| `run_011_..._114k` | ST, encodeur Large 114k h |
| `run_012_...` / `run_013_...` | speechLLM Large 14k / 114k |
| `run_001_pantagruel_multimodal` | Variante Speech_Text multimodale |

Suffixes de fichiers config : `_utterance`, `_sentence_like`, `_large_14k`, `_long` (durée entraînement / updates), `_unfreeze_encoder`.

---

## 14. Chemins et répertoires

| Chemin | En clair |
|--------|----------|
| `datasets/raw/` | Archives / sources après download |
| `datasets/manifests/<pair>/` | Manifests utterance |
| `datasets/manifests_sentence/<pair>/` | Manifests sentence_like |
| `datasets/processed/<pair>/` | WAV utterance |
| `datasets/processed_sentence/<pair>/` | WAV sentence_like |
| `datasets/processed/spm/` | Modèles SentencePiece entraînés |
| `runs/<langpair>/<run_id>/` | Tous les artifacts d’une expérience |
| `logs/` | Journaux nohup / wrappers scripts |
| `.venv/` | Environnement Python virtuel |
| `requirements.txt` / `requirements.lock.txt` | Dépendances / verrouillage reproductible |

---

## 15. Outils, bibliothèques et services externes

| Nom | En clair | Rôle S3T |
|-----|----------|----------|
| **PyTorch** (`torch`) | Framework deep learning | Entraînement, inférence |
| **torchaudio** | Audio pour PyTorch | Chargement WAV |
| **transformers** | Modèles Hugging Face | Encodeurs, Whisper, Marian, LLM |
| **sentencepiece** | Tokenisation sous-mots | SPM étape 3 |
| **sacrebleu** | Métriques BLEU/chrF/TER | Évaluation |
| **soundfile** | Lecture/écriture WAV | prepare, collate |
| **google-genai** | Client API Google | Gemini |
| **ruff** | Linter + formateur Python | Qualité code |
| **pytest** | Tests unitaires | CI locale |
| **pre-commit** | Hooks git | Qualité avant commit |
| **TensorBoard** | Visualisation courbes | Optionnel |
| **fairseq** | Toolkit FAIR (historique papier) | **Non** dans le code S3T actuel ; référence protocole article |
| **SpeechBrain** | Toolkit speech (mention PRD) | Divergences documentées vs implémentation PyTorch custom |

---

## 16. Matériel et exécution

| Terme | En clair |
|-------|----------|
| **CUDA** | Couche NVIDIA pour GPU ; vérifiée en preflight |
| **Preflight** | Contrôle machine avant jobs longs |
| **nohup** | Lancer un job qui continue après déconnexion SSH |
| **VRAM** | Mémoire GPU ; critique pour LLM 7B, Whisper large |
| **cu124** (ex.) | Version CUDA ciblée par l’index PyTorch dans le README |

Voir [estimation_ressources_fr_en.md](estimation_ressources_fr_en.md) pour ordres de grandeur disque/GPU.

---

## 17. Documents et références internes

| Fichier | En clair |
|---------|----------|
| [PRD.md](PRD.md) | Exigences et architecture cible |
| [protocole_evaluation.md](protocole_evaluation.md) | Règles SacreBLEU et décodage figés |
| [protocole_utterance_pantagruel.md](protocole_utterance_pantagruel.md) | Bench comparables Table 8 |
| [plan_migration_speechllm.md](plan_migration_speechllm.md) | Feuille de route speechLLM B1/B2 |
| [presentation_fr_en_pantagruel.md](presentation_fr_en_pantagruel.md) | Support oral 20–25 min |
| [presentation_fr_en_pantagruel_10slides.md](presentation_fr_en_pantagruel_10slides.md) | Version courte |
| [corpus_oralite_externe.md](corpus_oralite_externe.md) | Tests hors m-TEDx |
| [estimation_ressources_fr_en.md](estimation_ressources_fr_en.md) | Budget disque/GPU |
| [rapport.md](../rapport.md) | Rapport de stage (synthèse résultats) |
| `Pantagruel_2026.pdf` | Article source (si présent localement) |

---

## Annexe — Trois axes à ne pas confondre (lecture des résultats)

```text
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Axe             │ Question                                                 │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ Paradigme       │ Comment traduit-on ? (E2E, speechLLM, API, cascade, …)   │
│ Encodeur        │ Quel pré-entraînement ? (1k vs 14k vs 114k heures)       │
│ Segmentation    │ Comment est découpé l’audio ? (utterance vs sentence_like)│
└─────────────────┴──────────────────────────────────────────────────────────┘
```

Un score BLEU n’a de sens **que** si ces trois axes (et la **version de protocole** SacreBLEU/décodage) sont identiques ou explicités.

---

*Dernière mise à jour : juin 2026 — à compléter lors de l’ajout de nouvelles variantes, protocoles ou colonnes CSV.*
