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
| *sentence_like* | Fusion de morceaux voisins pour approcher une phrase complète (~10–15 s) | Segments souvent plus stables à l’entraînement |

Un modèle entraîné sur l’un ne doit pas être évalué sur l’autre : les scores ne seraient pas comparables.

### Taille de l’encodeur : 1k, 14k, 114k

Il s’agit du volume de parole française que l’[encodeur](vocabulaire.md#encodeur-acoustique--ssl) Pantagruel a « entendue » pendant son pré-entraînement, avant notre fine-tuning sur m-TEDx. Ce n’est pas la durée d’un extrait audio.

| Libellé | Ordre de grandeur | Rôle |
|---------|-------------------|------|
| [B-1k](vocabulaire.md#b-1k--l-14k--l-114k-échelle-de-pré-entraînement) | ~1 000 h | Référence principale du stage aujourd’hui |
| L-14k | ~14 000 h | Variante Large de l’article (~24 BLEU en utterance) |
| L-114k | ~114 000 h | Encore plus de données (~25 BLEU dans l’article) |


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

**Pourquoi SentencePiece ?** Le décodeur ne manipule pas des mots entiers : il prédit une suite de petites unités issues d’un vocabulaire fixe (ici ~1000, comme dans l’article Pantagruel). SentencePiece apprend ce vocabulaire sur les traductions anglaises du corpus. Intérêt : un mot rare ou absent à l’entraînement peut quand même être produit en le recomposant morceau par morceau ; le modèle reste plus compact qu’avec un dictionnaire « un mot = une entrée ». À l’entraînement, les phrases de référence sont découpées en unités SPM ; à la génération, le décodeur en émet une à une, puis SPM les rassemble en phrase lisible.

**Pourquoi cette variante ?** C’est le cœur scientifique du stage : reproduire les résultats de l’article (environ 17,5 BLEU test sur le découpage *utterance*, encodeur pré-entraîné sur ~1000 h de parole française — voir [B-1k](vocabulaire.md#b-1k--l-14k--l-114k-échelle-de-pré-entraînement)).

**Résultats indicatifs :**

| Caractéristiques | BLEU test | Durée |
|------------------|----------:|-------|
| *utterance* B-1k [^1] | **~16,7** | ~1 h 15 |
| *utterance* L-14k [^16] | **~17,2** | early stop ~21k upd. |
| *utterance* L-114k [^15] | **~19,6** | ~9 h GPU (OVH, early stop ~21k) |
| *sentence_like* [^2] | ~15 | ~8 h |

**Pistes d’amélioration :**
- Encodeur **L-14k** mesuré [^16] : gain marginal vs B-1k (~+0,5 BLEU), loin du papier (~24) ; **L-114k** [^15] mesuré (~19,6) — gain vs L-14k mais encore ~5,6 pts sous le papier (~25,2).
- Implémenter le [décodage par faisceau](vocabulaire.md#beam-search-beam-5) (beam 5, comme dans l’article) à la place du greedy utilisé aujourd’hui.
- Affiner l’entraînement : durée, [gel de l’encodeur](vocabulaire.md#freeze_encoder_updates), taux d’apprentissage, taille des lots.
- Poursuivre l’alignement sur le protocole *utterance* de l’article (Table 8) tout en documentant clairement les runs *sentence_like*.

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
| *utterance* B-1k [^3] | ~7,5 | ~2 h |
| *utterance* L-14k gelé [^12] | **~15,0** | ~1,4 h GPU (OVH) |
| *utterance* L-114k gelé [^14] | **~15,2** | ~4–6 h GPU (OVH) |
| *utterance* L-14k dégelé [^13] | ~3,7 | ~2–3 h GPU (Modyco) — sous run_012 gelé |
| *sentence_like*, encodeur gelé [^4] | ~16 | ~2 h |
| *sentence_like*, encodeur dégelé [^5] | **~19** | ~2 h |

**Pistes d’amélioration :**
- Lire les exemples produits (`eval/dev_predictions.txt`) : répétitions, traductions trop courtes ou trop longues, erreurs récurrentes.
- Tester un encodeur plus grand (14k / 114k h), comme pour la variante 1.
- Essayer d’autres [LLM](vocabulaire.md#llm-grand-modèle-de-langue) gelés (Llama, Mistral, Qwen) -> chaque modèle demande un projecteur réentraîné.
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
| **2.5 Flash** | *utterance* | [^6] | **~34** | ~1–2 h | voir `eval/metrics.json` (runs historiques) |
| **2.5 Flash** | *sentence_like* | [^7] | ~23 | ~1–2 h | idem |
| **3.5 Flash** | *utterance* | [^10] | **~13** | ~99 min | **~0,60 $** |
| **3.5 Flash** | *sentence_like* | [^11] | **~1,5** | ~42 min | **~0,52 $** |
| **3.5 Flash v2** | *utterance* | [^19] | **~41** | ~66 min | **~0,94 $** |
| **3.5 Flash v2** | *sentence_like* | [^20] | **~36,8** | ~38 min | **~1,27 $** |

**Comparaison 2.5 vs 3.5 (*utterance*) :** sous `max_output_tokens=256` (run [^10]), le 3.5 est **~20 points sous le 2.5** (13,4 vs 33,7) — hypothèses **tronquées**. Relance v2 [^19] (`8192` tokens, `thinking_level: minimal`, garde-fous anti-boucles) : BLEU test **41,1** — **devant** le 2.5 (33,7) et la cascade (37,4). Run [^17] (sans garde-fous) avait un test biaisé (20,3, 2 outliers).

**Point de vigilance :**  
Les extraits m-TEDx sont librement accessibles sur Internet (vidéos, transcriptions, sous-titres) et le corpus complet est librement téléchargeable en ligne. On ne peut pas exclure que Gemini ait rencontré des contenus proches lors de son pré-entraînement. Les scores de cette baseline se comparent donc avec prudence aux systèmes entraînés uniquement sur nos jeux train/dev/test : une partie de la performance peut refléter une familiarité avec le corpus plutôt qu’une vraie généralisation.

**Pistes d’amélioration :**
- Gemini 3.5 v2 *sentence_like* [^20] terminé (~36,8 test) — comparer au 2.5 (~23) et à l’utterance v2 (~41).
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
| *utterance* [^8] | ~37 | ~4 h |
| *sentence_like* | — | — |

**Pistes d’amélioration :**
- Tester un modèle [ASR](vocabulaire.md#2-abréviations-et-acronymes) plus léger ou plus lourd (Whisper medium vs large) et mesurer le compromis vitesse / qualité.
- Essayer un autre traducteur texte ([MT](vocabulaire.md#2-abréviations-et-acronymes)), par exemple NLLB à la place de Marian.
- Compléter le bench en *sentence_like* pour avoir le tableau complet sur les deux découpages.
- Analyser les erreurs de transcription française qui se répercutent sur l’anglais.

**Dossier :** `4_cascade/`

---

## 5. Encodeur multimodal Speech_Text (expérimental)

**En une phrase :** même architecture que la variante 1, mais avec un encodeur Pantagruel entraîné sur parole et texte ensemble, pas sur la parole seule.

L’encodeur [`Speech_Text`](vocabulaire.md#speech_text--speech_text-multimodal) a vu du français oral et écrit pendant son pré-entraînement. L’hypothèse : ces représentations pourraient mieux servir la traduction. Le décodeur Transformer et SentencePiece restent les mêmes qu’en variante 1 :

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
| *sentence_like* [^9] | ~8 | ~8 h |

**Pistes d’amélioration :**
- Reprendre les réglages d’entraînement de la variante 1 (gel, durée, décodage) avant de conclure sur l’encodeur multimodal.
- Lancer un run en *utterance* pour comparer au protocole article.
- Tester un checkpoint [`Speech_Text`](vocabulaire.md#speech_text--speech_text-multimodal) plus récent ou plus grand, si disponible.
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

| # | Nom court | Paradigme | Entraînement local | Intérêt principal |
|---|-----------|-----------|--------------------|-------------------|
| 1 | Transformer ST | E2E | Oui (GPU, long) | Réplication Pantagruel |
| 2 | speechLLM B1 | E2E via LLM | Oui (projecteur surtout) | LLM + adaptateur minimal |
| 3 | Gemini | API | Non | Référence commerciale |
| 4 | Cascade | ASR puis MT | Non (inférence seule) | Baseline classique en deux temps |
| 5 | Speech_Text | E2E | Oui | Test encodeur parole+texte |

---

## Synthèse des meilleurs scores

Meilleur BLEU test SacreBLEU observé par variante. Les paramètres listés sont ceux du run correspondant — voir les sections détaillées ci-dessus.

| # | Variante | Réf. | BLEU test | Découpage | Paramètres du run |
|---|----------|:----:|----------:|-----------|-------------------|
| 1 | Transformer ST | [^15] | ~19,6 | *utterance* | Encodeur **L-114k** v2 ; gel 5k ; early stop ~21k ; beam 5 (L-14k [^16] : ~17,2) |
| 2 | speechLLM B1 | [^5] | ~19 | *sentence_like* | Encodeur B-1k dégelé ; Phi-2 gelé ; projecteur seul ; 20k upd. ; beam 1 / 48 tok. |
| 2 | speechLLM B1 | [^14] | ~15,2 | *utterance* | Encodeur **L-114k** gelé ; Phi-2 gelé ; beam 1 / 48 tok. (L-14k [^12] : ~15,0) |
| 3 | Gemini 2.5 Flash | [^6] | ~34 | *utterance* | API `gemini-2.5-flash` ; *température* 0 ; max 256 tokens |
| 3 | Gemini 3.5 Flash | [^10] | ~13 | *utterance* | `gemini-3.5-flash` ; max 256 ; troncatures (voir §3) |
| 3 | Gemini 3.5 Flash v2 | [^19] | ~41 | *utterance* | max 8192 + thinking minimal + garde-fous (`run_005`) |
| 3 | Gemini 3.5 Flash v2 | [^20] | ~36,8 | *sentence_like* | max 8192 + thinking minimal + garde-fous (`run_004`) |
| 4 | Cascade ASR→MT | [^8] | ~37 | *utterance* | Whisper large-v3 → Marian opus-mt-fr-en ; inférence seule |
| 5 | Speech_Text + ST | [^9] | ~8 | *sentence_like* | Encodeur Speech_Text B-1k ; décodeur 6 couches + SPM ; 80k updates ; greedy |

Sur *utterance*, **Gemini 3.5 v2** (~41 [^19]) devance la cascade (~37) et Gemini 2.5 (~34), loin devant les modèles entraînés localement (meilleur ST : **~19,6** L-114k [^15] ; meilleur speechLLM : **~15,2** L-114k [^14]). Les runs 3.5 v1 [^10] restent non conclusifs (troncature). Sur *sentence_like*, Gemini 3.5 v2 (~36,8 [^20]) devance Gemini 2.5 (~23). Les scores *utterance* et *sentence_like* ne sont pas directement comparables entre eux (voir ci-dessous).

## Comment lire les chiffres

Avant de comparer deux scores, reprendre la [liste des réglages](#les-réglages-quon-peut-faire-varier) : variante, découpage, taille d’encodeur, gel/dégel, décodage. Un BLEU n’a de sens que si ces axes (et la version du protocole SacreBLEU) sont identiques ou clairement indiqués.

Un BLEU plus élevé suggère une traduction plus proche des références humaines, mais ce n’est qu’un indicateur — la qualité réelle se juge aussi en lisant des exemples (`eval/dev_predictions.txt` dans chaque run).

Pour le détail des scores, protocoles et identifiants de runs : voir le [rapport de stage](../rapport.md) et le [glossaire](vocabulaire.md).

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
[^12]: `run_012_speechllm_b1_utterance_large_14k` — 15,49 / **15,03** test (OVH)
[^13]: `run_015_speechllm_b1_utterance_large_14k_unfreeze` — 3,90 / **3,65** test (Modyco)
[^14]: `run_013_speechllm_b1_utterance_large_114k` — 15,92 / **15,24** test (OVH)
[^15]: `run_016_transformer_baseline_utterance_large_114k_v2` — 20,30 / **19,63** test (OVH, early stop ~21k upd.)
[^16]: `run_014_transformer_baseline_utterance_large_14k_v2` — 17,12 / **17,21** test (Modyco)
[^17]: `run_004_gemini_35_flash_utterance_v2` — 41,42 / **20,32** test (2 outliers, sans garde-fous)
[^19]: `run_005_gemini_35_flash_utterance_v2` — 41,42 / **41,09** test (garde-fous anti-boucles)
[^20]: `run_004_gemini_35_flash_sentence_like_v2` — 38,69 / **36,76** test (garde-fous anti-boucles)
[^18]: LeCun, Y. (2022). *A path towards autonomous machine intelligence* (version 0.9.2, 2022-06-27). *Open Review*, 62(1), 1–62.
