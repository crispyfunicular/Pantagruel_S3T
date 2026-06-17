# Cinq façons de traduire la parole

Ce projet compare cinq approches pour passer de l’audio français à du texte anglais (la [traduction de la parole](vocabulaire.md#2-abréviations-et-acronymes) — abréviation ST). Toutes partent des mêmes enregistrements : des extraits de conférences TED en français ([m-TEDx](vocabulaire.md#m-tedx-multilingual-tedx)).

L’objectif n’est pas de dire qu’une méthode est « la meilleure » en absolu, mais de mesurer ce que chaque idée apporte, avec les mêmes règles d’évaluation pour toutes ([SacreBLEU](vocabulaire.md#sacrebleu-corpus-bleu) sur les mêmes jeux de test).

> **Vocabulaire** — Les mots techniques (ST, [E2E](vocabulaire.md#2-abréviations-et-acronymes) — *bout en bout*, encodeur, BLEU, etc.) sont définis simplement dans le [glossaire du projet](vocabulaire.md).

---

## Avant les variantes : la préparation commune

Quelle que soit la méthode choisie, les étapes 0 à 2 sont identiques :

1. Télécharger le corpus m-TEDx : librement disponible en ligne sur [OpenSLR SLR100](https://www.openslr.org/100).
2. Préparer les enregistrements (audio + textes) :
   - **2a. Découper** : Le corpus brut fournit de longs fichiers audio au format **FLAC** (fichier audio compressé, comme un MP3 mais sans perte de qualité). On en extrait de courts segments : chaque extrait correspond à une prise de parole repérée dans les métadonnées du corpus.
   - **2b. Convertir** : Chaque segment est réenregistré en **WAV**, le format audio standard utilisé par la suite du pipeline. On impose une cadence de **16 kHz** (16 000 mesures du signal par seconde — ce qu’attendent les modèles Pantagruel), une piste **mono** (un seul canal, pas de stéréo), et un encodage **PCM 16 bits** (chaque point du signal est stocké sur 16 bits : la manière classique de représenter le son dans un fichier WAV).
   - **2c. Nettoyer** : Normalisation des textes français et anglais, filtrage des segments trop courts ou trop longs, et contrôle des jeux d’entraînement, validation et test (pour éviter qu’un même locuteur apparaisse à la fois à l’entraînement et au test).
3. Construire des listes d’exemples : pour chaque extrait, on associe le fichier audio et la traduction anglaise de référence ([manifest](vocabulaire.md#manifest)).

Ensuite seulement, chaque variante applique sa propre recette pour produire l’anglais à partir de l’audio.

---

## Les réglages que l’on peut faire varier

Chaque expérience combine plusieurs choix. Pour comparer deux résultats, il faut vérifier qu’on parle des mêmes réglages — sinon le score peut refléter autre chose que la variante elle-même.

```text
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Réglage         │ Question                                                 │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ Variante        │ Quelle recette ? (Transformer, speechLLM, Gemini, …)     │
│ Découpage audio │ Petits ou longs morceaux ? (utterance / sentence_like)   │
│ Encodeur        │ Combien d’heures de pré-entraînement ? (1k / 14k / 114k) │
│ Gel / dégel     │ Quels blocs apprennent pendant l’entraînement ?          │
│ Décodage        │ Comment le modèle choisit les mots à la génération ?     │
└─────────────────┴──────────────────────────────────────────────────────────┘
```

### Variante (la recette globale)

C’est le « comment » traduire : un seul modèle bout en bout, un LLM avec adaptateur, une API cloud, deux modèles en chaîne, etc. Les cinq pistes sont détaillées dans les sections suivantes.

### Découpage audio : *utterance* ou *sentence_like*

On ne change pas le corpus source, seulement la façon de le [découper](vocabulaire.md#segmentation-général) en exemples :

| Mode | En clair | Usage typique |
|------|----------|---------------|
| *utterance* | Petits morceaux tels que fournis par m-TEDx | Comparaison avec l’article Pantagruel (Table 8) |
| *sentence_like* | Fusion de morceaux voisins pour approcher une phrase complète (10–15 s) | Segments souvent plus stables à l’entraînement |

Un modèle entraîné sur l’un ne doit pas être évalué sur l’autre : les scores ne seraient pas comparables.

### Taille de l’encodeur : 1k, 14k, 114k

Il s’agit du volume de parole française que l’[encodeur](vocabulaire.md#encodeur-acoustique--ssl) Pantagruel a « entendue » pendant son pré-entraînement, avant notre fine-tuning sur m-TEDx. Ce n’est pas la durée d’un extrait audio.

| Libellé | Ordre de grandeur | Rôle |
|---------|-------------------|------|
| [B-1k](vocabulaire.md#b-1k--l-14k--l-114k-échelle-de-pré-entraînement) | 1 000 h | Référence principale du stage aujourd’hui |
| L-14k | 14 000 h | Variante Large de l’article (24 BLEU en utterance) |
| L-114k | 114 000 h | Encore plus de données (25 BLEU dans l’article) |


### Gelé ou dégelé

Lors de l’entraînement, chaque bloc du modèle peut être [gelé ou dégelé](vocabulaire.md#gelé-frozen--dégelé-unfrozen) :

- **Gelé** : les poids restent fixes (on garde ce qu’apporte le pré-entraînement).
- **Dégelé** : les poids peuvent être mis à jour (le bloc s’adapte à m-TEDx).

Exemples dans ce projet :

| Bloc | Variante 1 (Transformer) | Variante 2 (speechLLM B1) |
|------|---------------------------|---------------------------|
| **Encodeur Pantagruel** | Gelé au début, puis parfois dégelé après N pas d’entraînement | Gelé par défaut ; expérience « encodeur dégelé » testée |
| **Décodeur ou projecteur** | Entraîné | Seul le projecteur est entraîné (mode B1) |
| **LLM** | — | Gelé |

Geler limite le coût GPU et évite d’« oublier » le pré-entraînement ; dégeler peut gagner quelques points de BLEU au prix d’un entraînement plus lourd et plus risqué.

### Décodage : greedy et beam

Une fois le modèle entraîné, il faut encore choisir comment produire le texte anglais, mot par mot :

| Mode | En clair | Effet habituel |
|------|----------|----------------|
| [Greedy](vocabulaire.md#greedy-decoding) | À chaque pas, prendre le mot le plus probable | Rapide ; parfois sous-optimal |
| [Beam search](vocabulaire.md#beam-search-beam-5) (ex. beam 5) | Garder plusieurs suites de mots en parallèle (ici 5) | Plus lent ; souvent meilleur BLEU |

L’article Pantagruel utilise beam 5 ; la baseline ST de ce dépôt utilise encore le greedy à l’évaluation (écart documenté). speechLLM et Gemini ont leurs propres réglages (nombre de tokens max, température pour l’API).

### Autres réglages (selon la variante)

- Durée d’entraînement (nombre de [pas d’optimisation](vocabulaire.md#update-pas-depoch)).
- Taux d’apprentissage, taille des lots, [gel de l’encodeur](vocabulaire.md#freeze_encoder_updates) pendant les N premiers pas.
- Consigne envoyée au modèle ([prompt](vocabulaire.md#user--assistant-format-prompt)) pour speechLLM et Gemini.
- Choix du LLM gelé (Phi-2, Mistral, Qwen, etc.) en speechLLM.

---

## 1. Traduction directe avec Transformer (baseline Pantagruel)

**En une phrase :** un seul modèle apprend à lire l’audio et à écrire l’anglais, de bout en bout (*end-to-end* ou E2E).

C’est la piste de référence de l’article [Pantagruel](vocabulaire.md#pantagruel-article--famille-de-modèles) (2026). Le système se compose de trois blocs :

| Étape | Bloc | Rôle | Entraîné ? |
|:-----:|------|------|------------|
| 1 | **[Encodeur](vocabulaire.md#encodeur-acoustique--ssl) Pantagruel** | Transforme l’audio en représentations internes (une sorte de « compréhension acoustique ») | D’abord [gelé](vocabulaire.md#gelé-frozen--dégelé-unfrozen), puis parfois affiné |
| 2 | **[Décodeur](vocabulaire.md#décodeur-transformer-6-couches) Transformer (6 couches)** | Génère le texte anglais, mot par mot (en unités SPM) | Oui |
| 3 | **[SentencePiece](vocabulaire.md#2-abréviations-et-acronymes) (SPM)** | Découpe et recompose le texte anglais en petites unités (sous-mots) | Entraîné une fois sur les textes anglais du corpus |

**Comment l’encodeur Pantagruel a-t-il appris ?** Avant notre fine-tuning sur m-TEDx, l’encodeur a été pré-entraîné sur de grandes quantités de parole française (1k, 14k ou 114k h) avec une approche [**JEPA**](vocabulaire.md#jepa-joint-embedding-predictive-architecture) [^18] (*Joint Embedding Predictive Architecture*) : au lieu de reconstruire l’audio brut ou de prédire des unités acoustiques discrètes (comme wav2vec ou HuBERT), le modèle apprend à **prédire des représentations internes** dans un espace latent. Concrètement, un encodeur « étudiant » voit l’audio partiellement masqué et doit retrouver les vecteurs qu’un encodeur « professeur » (qui voit tout l’audio) aurait produits sur les parties cachées — implémentation **data2vec 2.0** dans l’article Pantagruel. Ce pré-entraînement donne à l’encodeur une « compréhension acoustique » riche ; notre travail consiste ensuite à brancher un décodeur de traduction dessus. Détail : [glossaire JEPA](vocabulaire.md#jepa-joint-embedding-predictive-architecture).

**Pourquoi SentencePiece ?** Le décodeur ne manipule pas des mots entiers : il prédit une suite de petites unités issues d’un vocabulaire fixe (ici 1000, comme dans l’article Pantagruel). SentencePiece apprend ce vocabulaire sur les traductions anglaises du corpus. Intérêt : un mot rare ou absent à l’entraînement peut quand même être produit en le recomposant morceau par morceau ; le modèle reste plus compact qu’avec un dictionnaire « un mot = une entrée ». À l’entraînement, les phrases de référence sont découpées en unités SPM ; à la génération, le décodeur en émet une à une, puis SPM les rassemble en phrase lisible.

**Pourquoi cette variante ?** C’est le cœur scientifique du stage : reproduire les résultats de l’article (environ 17,5 BLEU test sur le découpage *utterance*, encodeur pré-entraîné sur 1000 h de parole française — voir [B-1k](vocabulaire.md#b-1k--l-14k--l-114k-échelle-de-pré-entraînement)).

**Résultats indicatifs :**

| Caractéristiques | BLEU test | Durée |
|------------------|----------:|-------|
| *utterance* B-1k [^1] | **16,7** | 1 h 15 GPU (tour) |
| *utterance* L-14k v2 [^16] | 17,2 | early stop 21k upd. |
| *utterance* L-14k **v3** [^21] | 21,2 | 9–10 h GPU |
| *utterance* L-14k **v5** SpecAugment [^27] | **26,1** | 7,6 h GPU — **dépasse papier L-14k (24)** ; **meilleur ST local** |
| *utterance* L-14k v6 long [^31] | 25,1 | early stop (14 juin) — sous run_026 |
| *utterance* L-14k v7 SPM 5k [^32] | 24,0 | early stop (14 juin) — sous run_026 |
| *utterance* L-14k v8 SPM 8k [^34] | 22,2 | early stop (14 juin) — sous run_031 |
| *utterance* L-114k v5 SpecAugment [^33] | **23,5** | early stop (14 juin) — meilleur L-114k local |
| *utterance* B-1k v5 SpecAugment [^35] | **19,75** | 15 juin |
| *utterance* L-14k v9 warmup 10k [^36] | 0,60 | interrompu — éval best.pt non représentative |
| *utterance* L-14k v10 finetune freq [^41] | *en cours* | finetune run_026 + SpecAugment freq (3,5 h) |
| *utterance* L-114k v9 SpecAugment freq [^38] | *en file* | (waiter post run_033, @ 23,5k/80k) |
| *utterance* L-114k v2 [^15] | **19,6** | 9 h GPU (early stop 21k) |
| *utterance* L-114k **v3** [^26] | **20,2** | 9–10 h GPU |
| *sentence_like* [^2] | 15 | 8 h GPU |

**Avantages et inconvénients :**

| | |
|---|---|
| **Forces** | Seule piste alignée sur la Table 8 Pantagruel ; contrôle total du pipeline ; checkpoints réutilisables ; pas de coût API ; B-1k proche du papier (16,7 vs 17,5) |
| **Faiblesses** | Entraînement long (jusqu’à 10 h sur Large) ; sensible aux hyperparamètres (collapses documentés) ; décodage **greedy** (pas beam 5 papier) ; écart 3–5 pts sous le papier en Large ; forte VRAM |
| **Coût** | GPU uniquement (1–10 h selon taille encodeur) |
| **Quand la choisir** | Réplication article, analyse scientifique, déploiement sans cloud |

**Pistes d’amélioration :**
- Encodeur **L-14k** : v5 SpecAugment [^27] (**26,1**, **meilleur local**, vocab 1k) ; **`run_041`** finetune freq **en cours** ; `run_036` interrompu (éval **0,60**) ; `run_037` non lancé. **L-114k** v5 [^33] (23,5) ; `run_033` SPM 5k **en cours** ; `run_038` freq **en file**.
- Implémenter le [décodage par faisceau](vocabulaire.md#beam-search-beam-5) (beam 5) — **fait** à l’évaluation (`5_evaluate.py`).
- Affiner l’entraînement : durée, [gel de l’encodeur](vocabulaire.md#freeze_encoder_updates), taux d’apprentissage, taille des lots.
- Poursuivre l’alignement sur le protocole *utterance* de l’article (Table 8) tout en documentant clairement les runs *sentence_like*.

#### Réplication stricte de l’article Pantagruel (Table 8)

Les runs utterance visent bien la Table 8 (même corpus, même découpage, mêmes encodeurs HF), mais la réplication reste **partielle** : plusieurs paramètres du protocole LeBenchmark / fairseq sont présents dans les fichiers YAML sans être exécutés par le code, et les runs « réussis » (v2, v3) ont surtout corrigé des **collapses** plutôt que copié fidèlement fairseq.

**Ce qui correspond déjà au papier**

| Élément | Cible Pantagruel / LeBenchmark | État S3T |
|---------|--------------------------------|----------|
| Corpus | m-TEDx fr→en | ok |
| Segmentation | *utterance* native | ok (`datasets/manifests/fr-en/`) |
| Encodeur | B-1k / L-14k / L-114k | checkpoints HF homologues |
| Décodeur | Transformer 6 couches, 8 têtes | ok |
| Tokenisation | SentencePiece 1k | ok |
| Optimiseur | AdamW (β₁=0,9, β₂=0,98) | ok dans `4_train.py` |
| Label smoothing | 0,1 | ok |
| Gel encodeur | 5 000 premiers pas (PRD) | ok à partir des configs v2 |
| Métrique | SacreBLEU test | ok (signature figée) |

**Écarts documentés (code ou protocole)**

| Élément | Cible papier / PRD | État actuel |
|---------|-------------------|-------------|
| **Beam search** | beam = 5 à l’évaluation | **beam 5** (`5_evaluate.py`, `decode.beam_size` YAML) |
| **Warmup LR** | 10 000 pas (montée progressive) | **implémenté** (`warmup_updates` lu par `4_train.py`) |
| **SpecAugment** | masquage temps/fréquence sur l’audio | **implémenté** (temporel + fréquentiel ST ; temporel speechLLM) |
| **Speed perturbation** | présente côté fairseq historique | **absente** (extension PRD « Temps B ») |
| **Batch effectif** | 64–256 séquences (PRD) | 8 séquences (contrainte VRAM) |
| **Durée max** | jusqu’à 120k pas (template PRD §9) | configs utterance : 80k ; early stop souvent @20k |
| **Early stopping** | patience 5–10 (PRD) | ajout S3T (patience 2 en v2) — mécanisme propre au dépôt |
| **Éval dev en cours de train** | — | `max_eval_batches: 20` en v2 → dev partiel (corrigé en v3) |

**Checklist — run « aussi fidèle que possible » sans fairseq**

Ordre suggéré (impact estimé sur la comparabilité Table 8) :

1. **Coder le beam search** en évaluation — **fait** (`5_evaluate.py`, `6_infer.py`).
2. **Implémenter le warmup LR** (10 000 pas) dans `4_train.py` — **fait** ; ablation `run_036` interrompue (éval **0,60**).
3. **Ajouter SpecAugment** — **fait** (v5+) ; `run_041` finetune freq **en cours** ; `run_038` (freq L-114k) **en file**.
4. **Rédiger une config dédiée** `base_utterance_replication_strict.yaml` calquée sur le template PRD §9 :
   - `segment_mode: utterance`
   - `freeze_encoder_updates: 5000`
   - `warmup_updates: 10000`
   - `learning_rate_peak: 0.0002` (ou 1e-4 selon ablation)
   - `max_updates: 120000`
   - `batch_size` × `gradient_accumulation` le plus proche possible de 64 séquences (selon GPU)
   - `eval_every_updates: 1000`
   - `max_eval_batches: null` (dev complet)
   - pas d’early stop agressif au premier essai (pour comparer au protocole papier)
   - `decode.beam_size: 5`
5. **Lancer sur les trois tailles d’encodeur** (B-1k, L-14k, L-114k) avec **au moins 2 seeds** (le papier rapporte ±0,4 BLEU).
6. **Documenter les écarts résiduels** inévitables : stack PyTorch/HF vs fairseq, pas de speed perturbation, batch plus petit.

**Critères de succès (réplication « réussie »)**

| Encodeur | BLEU test cible (papier) | Meilleur S3T actuel (greedy) |
|----------|--------------------------|------------------------------|
| B-1k | 17,5 | 16,7 [^1] — quasi atteint |
| L-14k | 24,0 | **26,1** [^27] — au-dessus du papier |
| L-114k | 25,2 | **23,5** [^33] (run_028) ; 20,2 [^26] (run_019) |

Tant que les points 1–3 ne sont pas codés, un run ne peut pas prétendre à une réplication contrôlée paramètre par paramètre — seulement à une **approximation documentée**, comme les runs v2/v3 actuels.

Références techniques : [PRD §5 et §9](PRD.md), [protocole utterance](protocole_utterance_pantagruel.md), [protocole d’évaluation](protocole_evaluation.md).

**Dossier :** `1_Transformer/`

---

## 2. Parole vers un grand modèle de langue (speechLLM)

**En une phrase :** on ne réentraîne presque rien, seulement un petit adaptateur entre l’oreille (Pantagruel) et un modèle de texte déjà très capable (un [LLM](vocabulaire.md#llm-grand-modèle-de-langue)).

L’idée vient de l’article [*An Embarrassingly Simple Approach for LLM with Strong ASR Capacity*](https://arxiv.org/abs/2402.08846) (méthode SLAM-ASR, Ma et al., 2024) : au lieu de construire un décodeur sur mesure, on branche l’audio sur un LLM existant (ici Phi-2) via un [projecteur](vocabulaire.md#projecteur-speechllm) — quelques couches linéaires.

| Étape | Composant | Rôle | Entraîné ? |
|:-----:|-----------|------|------------|
| 1 | **Encodeur Pantagruel** | Lit l’audio | Non ([gelé](vocabulaire.md#gelé-frozen--dégelé-unfrozen)) — sauf en expérience « encodeur dégelé » |
| 2 | **Projecteur** | Adapte les signaux audio au format attendu par le LLM | Oui (c’est tout l’entraînement en mode [B1](vocabulaire.md#b1--b2-speechllm)) |
| 3 | **LLM (Phi-2)** | Produit l’anglais comme dans une conversation | Non (gelé) |

Le modèle apprend avec un format de type dialogue : une consigne du côté [USER](vocabulaire.md#user--assistant-format-prompt), la traduction attendue du côté ASSISTANT.

**Pourquoi cette variante ?** Tester si un **LLM généraliste**, avec très peu de paramètres entraînés, peut rivaliser avec un système ST classique, et comparer le coût matériel (beaucoup de mémoire GPU pour charger le LLM).

**Résultats indicatifs :**

| Caractéristiques | BLEU test | Durée |
|------------------|----------:|-------|
| *utterance* B-1k [^3] | 7,5 | 2 h |
| *utterance* L-14k gelé [^12] | **15,0** | 1,4 h GPU — **48 tok** |
| *utterance* L-114k gelé [^14] | **15,2** | 4–6 h GPU — **48 tok** |
| *utterance* L-14k + Qwen2.5-3B [^24] | 13,0 | 0,4 h GPU éval — sous Phi-2 |
| *utterance* L-114k v2 (128 tok) [^22] | 5,6 | 15 h GPU — **échec** vs run_013 |
| *utterance* L-14k v3 (128 tok) [^23] | 5,5 | 4 h GPU — **échec** ; hyp. 3× trop longues |
| *utterance* L-14k replicate (48 tok) [^25] | **14,2** | 2,5 h GPU — proche run_012 (15,03) |
| *utterance* L-14k dégelé [^13] | 3,7 | 2–3 h GPU — sous run_012 gelé |
| *sentence_like*, encodeur gelé [^4] | 16 | 2 h |
| *sentence_like*, encodeur dégelé [^5] | **19** | 2 h |

**Avantages et inconvénients :**

| | |
|---|---|
| **Forces** | Très peu de paramètres entraînés (projecteur seul en B1) ; convergence rapide (20k updates, 1–6 h) ; bon en *sentence_like* dégelé (19 BLEU) ; warmup LR déjà implémenté ; **48 tok** → 15 BLEU utterance (`run_012`/`run_013`) |
| **Faiblesses** | Très faible en *utterance* B-1k (7,5 vs 16,7 ST) ; configs **128 tok** (v2/v3) → 5–6 BLEU (hypothèses trop longues) ; dégel encodeur risqué (3,7 test) ; sensible au décodage (beam ≥ 4 → boucles) ; forte VRAM (Phi-2 / Mistral) |
| **Coût** | GPU 1–6 h + chargement LLM en mémoire |
| **Quand la choisir** | Explorer l’approche LLM + adaptateur minimal ; pas comme baseline Table 8 aujourd’hui |

**Pistes d’amélioration :**
- **Décodage** : garder `max_new_tokens: 48` (comme `run_012`) — les configs v2/v3 à **128 tok** ont produit des hypothèses 3× trop longues et un BLEU 5–6 (`run_017`, `run_021`).
- Lire les exemples produits (`eval/dev_predictions.txt`) : répétitions, longueur des hypothèses, erreurs récurrentes.
- Tester un encodeur plus grand (14k / 114k h), comme pour la variante 1.
- Essayer d’autres [LLM](vocabulaire.md#llm-grand-modèle-de-langue) gelés (Llama, Mistral, Qwen) — `run_018` Qwen 13 test, sous Phi-2.
- Ajuster le décodage ([beam](vocabulaire.md#beam-search-beam-5), nombre max de tokens) et la durée d’entraînement du projecteur.
- Pousser au-delà du mode [B1](vocabulaire.md#b1--b2-speechllm) (dégel partiel du LLM ou de l’encodeur sur *utterance*).

**Dossier :** `2_speechLLM/`

---

## 3. Modèle cloud Gemini (API)

**En une phrase :** on envoie l’audio à Google Gemini et on récupère la traduction anglaise, sans entraînement local.

| Étape | Élément | Détail |
|:-----:|---------|--------|
| 1 | **Entrée** | Fichier audio ou flux audio (français) |
| 2 | **Modèle Gemini** ([API](vocabulaire.md#2-abréviations-et-acronymes)) | Reçoit l’audio et produit la traduction anglaise — sans entraînement local |
| 3 | **Sortie** | Texte anglais proposé par le modèle |

Coût : facturation à l’appel ; suivie dans les logs de run.

**Pourquoi cette variante ?** C’est une ligne de référence externe : que vaut un grand modèle multimodal commercial, comparé à nos systèmes entraînés sur m-TEDx ? Utile pour situer le travail de stage par rapport à l’état de l’art « prêt à l’emploi ».

**Résultats indicatifs :**

| Modèle | Découpage | Réf. | BLEU test | Durée | Coût API |
|--------|-----------|:----:|----------:|------:|---------:|
| **2.5 Flash** | *utterance* | [^6] | **34** | 1–2 h | voir `eval/metrics.json` (runs historiques) |
| **2.5 Flash** | *sentence_like* | [^7] | 23 | 1–2 h | idem |
| **3.5 Flash** | *utterance* | [^10] | **13** | 99 min | **0,60 $** |
| **3.5 Flash** | *sentence_like* | [^11] | **1,5** | 42 min | **0,52 $** |
| **3.5 Flash v2** | *utterance* | [^19] | **41** | 66 min | **0,94 $** |
| **3.5 Flash v2** | *sentence_like* | [^20] | **36,8** | 38 min | **1,27 $** |

**Comparaison 2.5 vs 3.5 (*utterance*) :** sous `max_output_tokens=256` (run [^10]), le 3.5 est **20 points sous le 2.5** (13,4 vs 33,7) — hypothèses **tronquées**. Relance v2 [^19] (`8192` tokens, `thinking_level: minimal`, garde-fous anti-boucles) : BLEU test **41,1** — **devant** le 2.5 (33,7) et la cascade (37,4). Run [^17] (sans garde-fous) avait un test biaisé (20,3, 2 outliers).

**Avantages et inconvénients :**

| | |
|---|---|
| **Forces** | Meilleurs BLEU du projet (41 utterance) ; zéro entraînement GPU ; rapide (1 h) ; coût faible (1 $ par run complet) |
| **Faiblesses** | Dépendance fournisseur Google ; reproductibilité limitée ; runs v1 non conclusifs (troncature) ; comparaison prudente avec modèles entraînés sur m-TEDx seul |
| **Coût** | 0,50–1,30 $ API par run (durée 38–66 min selon découpage) |
| **Quand la choisir** | Plafond de performance « prêt à l’emploi » ; bench externe sans GPU d’entraînement |

**Point de vigilance :**  
Les extraits m-TEDx sont librement accessibles sur Internet (vidéos, transcriptions, sous-titres) et le corpus complet est librement téléchargeable en ligne. On ne peut pas exclure que Gemini ait rencontré des contenus proches lors de son pré-entraînement. Les scores de cette baseline se comparent donc avec prudence aux systèmes entraînés uniquement sur nos jeux train/dev/test : une partie de la performance peut refléter une familiarité avec le corpus plutôt qu’une vraie généralisation.

**Pistes d’amélioration :**
- Gemini 3.5 v2 *sentence_like* [^20] terminé (36,8 test) — comparer au 2.5 (23) et à l’utterance v2 (41).
- Affiner le prompt (traduction complète, anglais seul).
- Affiner la consigne (prompt) : traduction **complète**, anglais uniquement, sans commentaire ni markdown.
- Documenter le coût par run ([API](vocabulaire.md#2-abréviations-et-acronymes) facturée à l’usage) — champs `gemini_cost_estimate_usd` et `runtime` dans `eval/metrics.json`.

**Dossier :** `3_Gemini/`

---

## 4. Deux étapes en chaîne : reconnaissance puis traduction (cascade)

**En une phrase :** d’abord transcrire le français à l’écrit, puis traduire ce texte en anglais, comme le ferait un humain avec deux outils séparés.

### ASR et MT, les deux briques de la cascade
- **Reconnaissance automatique de la parole** (*Automatic Speech Recognition* ou [ASR](vocabulaire.md#2-abréviations-et-acronymes)) : reconnaissance automatique de la parole : l’audio devient du texte dans la même langue (ici, français *oral* → français *écrit*).
- **Traduction automatique** (TA) (*Machine Translation* ou MT) : le texte passe d’une langue à une autre (ici, français → anglais).

| Étape | Élément | Outil | Tâche |
|:-----:|---------|-------|-------|
| 1 | **[ASR](vocabulaire.md#2-abréviations-et-acronymes)** | Whisper (large) | Audio français → texte français |
| 2 | **[MT](vocabulaire.md#2-abréviations-et-acronymes)** | Marian (opus-mt-fr-en) | Texte français → texte anglais |

Ce n’est pas de la traduction bout en bout (E2E) : l’anglais ne dépend que de la transcription intermédiaire. Si l’ASR se trompe, l’erreur se propage.

**Pourquoi cette variante ?** Les cascades restent très utilisées en production. Les comparer aux modèles E2E montre le compromis entre simplicité de déploiement, interprétabilité (on peut lire la transcription française) et score global.

**Résultats indicatifs :** (inférence seule, pas d’entraînement)

| Caractéristiques | BLEU test | Durée |
|------------------|----------:|-------|
| *utterance* [^8] | 37 | 3–5 h GPU (inférence) |
| *sentence_like* | — | non rapporté |

**Avantages et inconvénients :**

| | |
|---|---|
| **Forces** | Très bon BLEU utterance (37,4) ; transcription française lisible (interprétabilité) ; pas d’entraînement ; modèles ASR/MT SOTA |
| **Faiblesses** | Erreurs ASR propagées en cascade ; 2 modèles = latence et VRAM ; hors Table 8 papier ; bench *sentence_like* incomplet |
| **Coût** | 3–5 h GPU inférence (tour) |
| **Quand la choisir** | Baseline production classique ; comparer interprétabilité vs E2E |

**Pistes d’amélioration :**
- Tester un modèle [ASR](vocabulaire.md#2-abréviations-et-acronymes) plus léger ou plus lourd (Whisper medium vs large) et mesurer le compromis vitesse / qualité.
- Essayer un autre traducteur texte ([MT](vocabulaire.md#2-abréviations-et-acronymes)), par exemple NLLB à la place de Marian.
- Compléter le bench en *sentence_like* pour avoir le tableau complet sur les deux découpages.
- Analyser les erreurs de transcription française qui se répercutent sur l’anglais.

**Dossier :** `4_cascade/`

---

## 5. Encodeur multimodal Speech_Text (expérimental)

**En une phrase :** même architecture que la variante 1, mais avec un encodeur Pantagruel entraîné sur parole et texte ensemble, pas sur la parole seule.

L’encodeur [`Speech_Text`](vocabulaire.md#speech_text--speech_text-multimodal) repose sur un **backbone Transformer partagé** pré-entraîné sur trois types de données : parole seule, texte seul, et paires parole-texte **non-parallèles**. L’objectif de pré-entraînement est de type **JEPA / data2vec 2.0 multimodal** : le modèle prédit des représentations latentes communes pour la parole et le texte sans alignement explicite. L’hypothèse : ces représentations pourraient mieux servir la traduction ST que des représentations acoustiques seules. Le décodeur Transformer et SentencePiece restent les mêmes qu’en variante 1 :

| Étape | Bloc | Rôle | Entraîné ? |
|:-----:|------|------|------------|
| 1 | **Encodeur Speech_Text** | Transforme l’audio (pré-entraînement parole + texte) | D’abord gelé, puis parfois affiné |
| 2 | **Décodeur Transformer (6 couches)** | Génère le texte anglais, mot par mot (en unités SPM) | Oui |
| 3 | **SentencePiece (SPM)** | Fournit le vocabulaire et reconstitue la phrase à partir des unités générées | Entraîné une fois sur les textes anglais du corpus |

**Pourquoi cette variante ?** Explorer si la multimodalité au niveau de l’encodeur aide la ST — piste annoncée dans le titre du rapport de stage.

**Résultats indicatifs :**

| Caractéristiques | BLEU test | Durée |
|------------------|----------:|-------|
| *utterance* | — | — |
| *sentence_like* [^9] | 8 | 8 h GPU |

**Avantages et inconvénients :**

| | |
|---|---|
| **Forces** | Teste si un pré-entraînement parole+texte aide la ST ; même architecture décodeur que la variante 1 |
| **Faiblesses** | Pire score local (7,9 test) ; pas de run *utterance* ; même coût d’entraînement que ST sans gain mesuré |
| **Coût** | 8 h GPU (comme ST *sentence_like*) |
| **Quand la choisir** | Ablation encodeur multimodal uniquement — non prioritaire pour le bench principal |

**Pistes d’amélioration :**
- Reprendre les réglages d’entraînement de la variante 1 (gel, durée, décodage) avant de conclure sur l’encodeur multimodal.
- Lancer un run en *utterance* pour comparer au protocole article.
- Relancer `run_040` dès qu’un nouveau checkpoint `Speech_Text` sera publié (le checkpoint intermédiaire `Base_fr_1K_4GB` a été retiré de Hugging Face en juin 2026 ; l’entraînement du modèle multimodal final à grande échelle est en cours).
- Trancher : creuser l’hypothèse multimodale ou recentrer l’effort sur les variantes 1 et 2.

**Dossier :** `5_Pantagruel_multimodal/`

---

## Vue d’ensemble

```text
Audio français (m-TEDx)
        │
        ├──► [1] Pantagruel + décodeur     ──► anglais  (E2E, référence article)
        ├──► [2] Pantagruel → projecteur → LLM ──► anglais  (peu de paramètres entraînés)
        ├──► [3] Gemini (cloud)          ──► anglais  (pas d’entraînement local)
        ├──► [4] Whisper → Marian        ──► anglais  (deux modèles en série)
        └──► [5] Speech_Text + décodeur  ──► anglais  (encodeur multimodal)
```

| # | Nom court | Paradigme | Entraînement | Durée typique | Intérêt principal |
|---|-----------|-----------|--------------|---------------|-------------------|
| 1 | Transformer ST | E2E | Oui (GPU) | 1–10 h | Réplication Pantagruel |
| 2 | speechLLM B1 | E2E via LLM | Oui (projecteur) | 1–6 h | LLM + adaptateur minimal |
| 3 | Gemini | API | Non | 1 h | Référence commerciale |
| 4 | Cascade | ASR puis MT | Non (inférence) | 3–5 h | Baseline classique en deux temps |
| 5 | Speech_Text | E2E | Oui (GPU) | 8 h | Test encodeur parole+texte |

---

## Synthèse des meilleurs scores

Meilleur BLEU test SacreBLEU observé par variante. Les paramètres listés sont ceux du run correspondant — voir les sections détaillées ci-dessus.

| # | Variante | Réf. | BLEU test | Découpage | Durée | Paramètres du run |
|---|----------|:----:|----------:|-----------|-------|-------------------|
| 3 | Gemini 3.5 v2 | [^19] | **41** | *utterance* | 66 min | max 8192 tok + garde-fous ; 0,94 $ |
| 4 | Cascade | [^8] | 37 | *utterance* | 3–5 h GPU | Whisper large-v3 → Marian |
| 3 | Gemini 2.5 | [^6] | 34 | *utterance* | 1–2 h | temp 0 ; max 256 tok |
| 1 | Transformer ST | [^21] | **21,2** | *utterance* | 9–10 h GPU | L-14k v3 ; gel 5k ; greedy (L-114k v3 [^26] : 20,2) |
| 2 | speechLLM B1 | [^5] | 19 | *sentence_like* | 2 h | B-1k dégelé ; Phi-2 gelé ; beam 1 / 48 tok |
| 3 | Gemini 3.5 v2 | [^20] | 36,8 | *sentence_like* | 38 min | max 8192 ; 1,27 $ |
| 2 | speechLLM B1 | [^14] | 15,2 | *utterance* | 4–6 h GPU | L-114k gelé ; beam 1 / **48 tok** (128 tok → 5–6, voir [^22][^23]) |
| 1 | Transformer ST | [^2] | 15 | *sentence_like* | 8 h | B-1k ; greedy |
| 5 | Speech_Text + ST | [^9] | 8 | *sentence_like* | 8 h | Speech_Text B-1k ; greedy |
| 3 | Gemini 3.5 v1 | [^10] | 13 | *utterance* | 99 min | troncature — **non conclusif** |

Sur *utterance*, **Gemini 3.5 v2** (41 [^19]) devance la cascade (37) et Gemini 2.5 (34). Meilleur modèle **entraîné localement** : ST L-14k v3 (**21,2** [^21]) ; speechLLM L-114k (15,2 [^14]). Les runs 3.5 v1 [^10] restent non conclusifs. Les scores *utterance* et *sentence_like* ne sont pas directement comparables.

## Comment lire les chiffres

Avant de comparer deux scores, reprendre la [liste des réglages](#les-réglages-quon-peut-faire-varier) : variante, découpage, taille d’encodeur, gel/dégel, décodage. Un BLEU n’a de sens que si ces axes (et la version du protocole SacreBLEU) sont identiques ou clairement indiqués.

Un BLEU plus élevé suggère une traduction plus proche des références humaines, mais ce n’est qu’un indicateur — la qualité réelle se juge aussi en lisant des exemples (`eval/dev_predictions.txt` dans chaque run).

Pour le détail des scores, protocoles et identifiants de runs : voir le [rapport de stage](../rapport.md), le [README](../README.md) (tableaux avec durées) et le [glossaire](vocabulaire.md). Durées mesurées : `runs/fr-en/<run_id>/metrics.json` (`gpu_hours`, `duration_s`) ; agrégat [`runs/experiments_tracking.csv`](../runs/experiments_tracking.csv) après rsync depuis les machines GPU.

---

## Références des runs

Syntaxe : notes de bas de page Markdown (`[^n]`), supportées par Pandoc, GitHub et la plupart des visualiseurs récents.

[^1]: `run_004_transformer_baseline_utterance_v2`
[^2]: `run_001_transformer_baseline_sentence_like`
[^3]: `run_003_speechllm_b1_utterance_long`
[^4]: `run_002_speechllm_b1_sentence_long`
[^5]: `run_005_speechllm_b1_sentence_long_unfreeze_encoder`
[^6]: `run_001_gemini_flash_utterance_full`
[^7]: `run_001_gemini_flash_sentence_like_v2`
[^8]: `run_001_cascade_utterance`
[^9]: `run_001_pantagruel_multimodal`
[^10]: `run_003_gemini_35_flash_utterance`
[^11]: `run_003_gemini_35_flash_sentence_like` — 1,62 / **1,45** test (troncature)
[^12]: `run_012_speechllm_b1_utterance_large_14k` — 15,49 / **15,03** test
[^13]: `run_015_speechllm_b1_utterance_large_14k_unfreeze` — 3,90 / **3,65** test
[^14]: `run_013_speechllm_b1_utterance_large_114k` — 15,92 / **15,24** test
[^15]: `run_016_transformer_baseline_utterance_large_114k_v2` — 20,30 / **19,63** test (early stop 21k upd.)
[^16]: `run_014_transformer_baseline_utterance_large_14k_v2` — 17,12 / **17,21** test
[^21]: `run_020_transformer_baseline_utterance_large_14k_v3` — 22,05 / **21,22** test (eval dev complet, patience 4)
[^22]: `run_017_speechllm_b1_utterance_large_114k_v2` — 6,56 / **5,60** test (max 128 tok — échec vs run_013)
[^23]: `run_021_speechllm_b1_utterance_large_14k_v3` — 5,84 / **5,48** test (max 128 tok — hyp. 3× trop longues)
[^24]: `run_018_speechllm_b2bis_utterance_large_14k_qwen25_3b` — 13,96 / **12,95** test (Qwen2.5-3B)
[^25]: `run_023_speechllm_b1_utterance_large_14k_replicate` — 15,26 / **14,23** test (48 tok — proche run_012)
[^26]: `run_019_transformer_baseline_utterance_large_114k_v3` — 21,09 / **20,19** test (eval dev complet)
[^27]: `run_026_transformer_baseline_utterance_large_14k_v5` — 26,57 / **26,12** test (SpecAugment, 7,6 h GPU, early stop @55k)
[^28]: `run_024_transformer_baseline_utterance_large_14k_v4` — batch 64 ; **0,20 / 0,35** test (collapse)
[^29]: `run_022_speechllm_b1_utterance_large_114k_v3` — 5,28 / **4,78** test (128 tok — échec)
[^30]: `run_025_transformer_baseline_utterance_large_114k_v4` — batch 64 ; **0,24 / 0,31** test (collapse)
[^31]: `run_027_transformer_baseline_utterance_large_14k_v6_long` — 26,37 / **25,12** test (120k updates, early stop — sous run_026)
[^32]: `run_031_transformer_baseline_utterance_large_14k_v7_spm5k` — 24,24 / **24,02** test (SPM 5k — sous run_026)
[^33]: `run_028_transformer_baseline_utterance_large_114k_v5` — 24,08 / **23,51** test (SpecAugment — meilleur L-114k local)
[^34]: `run_034_transformer_baseline_utterance_large_14k_v8_spm8k` — 23,36 / **22,24** test (SPM 8k — sous run_031)
[^35]: `run_035_transformer_baseline_utterance_b1k_v5` — **19,75** test (B-1k SpecAugment, 15 juin)
[^36]: `run_036_transformer_baseline_utterance_large_14k_v9_warmup10k` — **interrompu** — éval best.pt **0,60 / 0,42** test/dev
[^37]: `run_037_transformer_baseline_utterance_large_14k_v9_specaug_strong` — **non lancé**
[^38]: `run_038_transformer_baseline_utterance_large_114k_v9_specaug_freq` — **en file** (après run_033)
[^39]: `run_039_speechllm_b1_utterance_large_14k_v5_specaug` — **13,84 / 14,59** test/dev (16 juin — sous run_023)
[^40]: `run_040_pantagruel_multimodal_utterance_v2` — **échec** (HF `Speech_Text_Base_fr_1K_4GB` 404 — checkpoint intermédiaire retiré ; modèle multimodal final en cours d’entraînement)
[^41]: `run_041_transformer_finetune_utterance_large_14k_v10_specaug_freq_from_run026` — **en cours** (finetune run_026)
[^032]: `run_032_speechllm_b1_utterance_large_114k_replicate` — **14,15 / 15,14** test/dev (48 tok)
[^033]: `run_033_transformer_baseline_utterance_large_114k_v7_spm5k` — **en cours** (@ 23,5k/80k, best dev 21,9)
[^17]: `run_004_gemini_35_flash_utterance_v2` — 41,42 / **20,32** test (2 outliers, sans garde-fous)
[^19]: `run_005_gemini_35_flash_utterance_v2` — 41,42 / **41,09** test (garde-fous anti-boucles)
[^20]: `run_004_gemini_35_flash_sentence_like_v2` — 38,69 / **36,76** test (garde-fous anti-boucles)
[^18]: LeCun, Y. (2022). *A path towards autonomous machine intelligence* (version 0.9.2, 2022-06-27). *Open Review*, 62(1), 1–62.
