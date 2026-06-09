# Cinq façons de traduire la parole

Ce projet compare cinq approches pour passer de l’audio français à du texte anglais (la [traduction de la parole](vocabulaire.md#2-abréviations-et-acronymes) — abréviation ST). Toutes partent des mêmes enregistrements : des extraits de conférences TED en français ([m-TEDx](vocabulaire.md#m-tedx-multilingual-tedx)).

L’objectif n’est pas de dire qu’une méthode est « la meilleure » en absolu, mais de mesurer ce que chaque idée apporte, avec les mêmes règles d’évaluation pour toutes ([SacreBLEU](vocabulaire.md#sacrebleu-corpus-bleu) sur les mêmes jeux de test).

> **Vocabulaire** — Les mots techniques (ST, E2E, encodeur, BLEU, etc.) sont définis simplement dans le [glossaire du projet](vocabulaire.md).

---

## Avant les variantes : la préparation commune

Quelle que soit la méthode choisie, les étapes 0 à 2 sont identiques :

1. Télécharger le corpus m-TEDx.
2. Préparer les enregistrements (audio + textes) :
   - **2a. Découper** : Le corpus brut fournit de longs fichiers audio au format **FLAC** (fichier audio compressé, comme un MP3 mais sans perte de qualité). On en extrait de courts segments : chaque extrait correspond à une prise de parole repérée dans les métadonnées du corpus.
   - **2b. Convertir** : Chaque segment est réenregistré en **WAV**, le format audio standard utilisé par la suite du pipeline. On impose une cadence de **16 kHz** (16 000 mesures du signal par seconde — ce qu’attendent les modèles Pantagruel), une piste **mono** (un seul canal, pas de stéréo), et un encodage **PCM 16 bits** (chaque point du signal est stocké sur 16 bits : la manière classique de représenter le son dans un fichier WAV).
   - **2c. Nettoyer** : Normalisation des textes français et anglais, filtrage des segments trop courts ou trop longs, et contrôle des jeux d’entraînement, validation et test (pour éviter qu’un même locuteur apparaisse à la fois à l’entraînement et au test).
3. Construire des listes d’exemples : pour chaque extrait, on associe le fichier audio et la traduction anglaise de référence ([manifest](vocabulaire.md#manifest)).

Ensuite seulement, chaque variante applique sa propre recette pour produire l’anglais à partir de l’audio.

**Point important :** on peut découper l’audio de deux manières ([segmentation](vocabulaire.md#segmentation-général)) :
- *utterance* : petits morceaux, comme dans l’article Pantagruel, ou
- *sentence_like* : morceaux plus longs regroupés.  

Un score n’est comparable qu’entre systèmes entraînés et testés sur le même découpage.

---

## 1. Traduction directe avec Transformer (baseline Pantagruel)

**En une phrase :** un seul modèle apprend à lire l’audio et à écrire l’anglais, de bout en bout.

C’est la piste de référence de l’article [Pantagruel](vocabulaire.md#pantagruel-article--famille-de-modèles) (2026). Le système se compose de trois blocs :

| Bloc | Rôle | Entraîné ? |
|------|------|------------|
| [Encodeur](vocabulaire.md#encodeur-acoustique--ssl) Pantagruel | Transforme l’audio en représentations internes (une sorte de « compréhension acoustique ») | D’abord [gelé](vocabulaire.md#gelé-frozen--dégelé-unfrozen), puis parfois affiné |
| [Décodeur](vocabulaire.md#décodeur-transformer-6-couches) Transformer (6 couches) | Génère le texte anglais, mot par mot | Oui |
| [SentencePiece](vocabulaire.md#2-abréviations-et-acronymes) (SPM) | Découpe l’anglais en petites unités que le décodeur manipule | Entraîné une fois sur les textes anglais du corpus |

**Pourquoi cette variante ?** C’est le cœur scientifique du stage : reproduire les résultats de l’article (environ 17,5 BLEU test sur le découpage *utterance*, encodeur pré-entraîné sur ~1000 h de parole française — voir [B-1k](vocabulaire.md#b-1k--l-14k--l-114k-échelle-de-pré-entraînement)).

**Résultats indicatifs :**
- *utterance* (comparabilité article Pantagruel) : **~16,7 BLEU** test après correction du protocole d’entraînement
- *sentence_like* : ~15 BLEU.

**Pistes d'amélioration :**
- Passer à un encodeur pré-entraîné sur plus de données ([14k ou 114k h](vocabulaire.md#b-1k--l-14k--l-114k-échelle-de-pré-entraînement) de parole française) — priorité actuelle après un premier essai 14k non concluant.
- Implémenter le [décodage par faisceau](vocabulaire.md#beam-search-beam-5) (beam 5, comme dans l’article) à la place du greedy utilisé aujourd’hui.
- Affiner l’entraînement : durée, [gel de l’encodeur](vocabulaire.md#freeze_encoder_updates), taux d’apprentissage, taille des lots.
- Poursuivre l’alignement sur le protocole *utterance* de l’article (Table 8) tout en documentant clairement les runs *sentence_like*.

**Dossier :** `1_Transformer/`

---

## 2. Parole vers un grand modèle de langue (speechLLM)

**En une phrase :** on ne réentraîne presque rien, seulement un petit adaptateur entre l’oreille (Pantagruel) et un modèle de texte déjà très capable (un [LLM](vocabulaire.md#llm-grand-modèle-de-langue)).

L’idée vient de l’article *SLAM-ASR* (« embarrassingly simple ») : au lieu de construire un décodeur sur mesure, on branche l’audio sur un LLM existant (ici Phi-2) via un [projecteur](vocabulaire.md#projecteur-speechllm) — quelques couches linéaires.

| Composant | Rôle | Entraîné ? |
|-----------|------|------------|
| Encodeur Pantagruel | Lit l’audio | Non ([gelé](vocabulaire.md#gelé-frozen--dégelé-unfrozen)) — sauf en expérience « encodeur dégelé » |
| Projecteur | Adapte les signaux audio au format attendu par le LLM | Oui (c’est tout l’entraînement en mode [B1](vocabulaire.md#b1--b2-speechllm)) |
| LLM (Phi-2) | Produit l’anglais comme dans une conversation | Non (gelé) |

Le modèle apprend avec un format de type dialogue : une consigne du côté [USER](vocabulaire.md#user--assistant-format-prompt), la traduction attendue du côté ASSISTANT.

**Pourquoi cette variante ?** Tester si un LLM généraliste, avec très peu de paramètres entraînés, peut rivaliser avec un système ST classique — et comparer le coût matériel (beaucoup de mémoire GPU pour charger le LLM).

**Résultat indicatif :**
- *sentence_like* : ~16 BLEU test (encodeur gelé) ; ~19 en dégelant l’encodeur.
- *utterance* : ~7,5 BLEU test — écart net avec la baseline Transformer ; relecture qualitative des traductions prioritaire.

**Pistes d'amélioration :**
- Lire les exemples produits (`eval/dev_predictions.txt`) : répétitions, traductions trop courtes ou trop longues, erreurs récurrentes.
- Tester un encodeur plus grand (14k / 114k h), comme pour la variante 1.
- Essayer d’autres [LLM](vocabulaire.md#llm-grand-modèle-de-langue) gelés (Llama, Mistral, Qwen) — chaque modèle demande un projecteur réentraîné.
- Ajuster le décodage ([beam](vocabulaire.md#beam-search-beam-5), nombre max de tokens) et la durée d’entraînement du projecteur.
- Pousser au-delà du mode [B1](vocabulaire.md#b1--b2-speechllm) (dégel partiel du LLM ou de l’encodeur sur *utterance*).

**Dossier :** `2_speechLLM/`

---

## 3. Modèle cloud Gemini (API)

**En une phrase :** on envoie l’audio à Google Gemini et on récupère la traduction anglaise — sans entraînement local.

| Aspect | Détail |
|--------|--------|
| Entrée | Fichier audio ou flux audio |
| Sortie | Texte anglais proposé par le modèle |
| Entraînement | Aucun dans ce dépôt |
| Coût | Facturation à l’appel ([API](vocabulaire.md#2-abréviations-et-acronymes)) ; suivie dans les logs de run |

**Pourquoi cette variante ?** C’est une ligne de référence externe : que vaut un grand modèle multimodal commercial, comparé à nos systèmes entraînés sur m-TEDx ? Utile pour situer le travail de stage par rapport à l’état de l’art « prêt à l’emploi ».

**Résultat indicatif :**
- *utterance* : ~34 BLEU test.
- *sentence_like* : ~23 BLEU test (Gemini 2.5 Flash).

**Pistes d'amélioration :**
- Comparer Gemini 3.5 Flash et 2.5 Flash sur les deux découpages, à prompt et température identiques.
- Affiner la consigne (prompt) envoyée au modèle.
- Documenter le coût par run ([API](vocabulaire.md#2-abréviations-et-acronymes) facturée à l’usage) pour situer la référence commerciale face aux systèmes locaux.

**Dossier :** `3_Gemini/`

---

## 4. Deux étapes en chaîne : reconnaissance puis traduction (cascade)

**En une phrase :** d’abord transcrire le français à l’écrit, puis traduire ce texte en anglais — comme le ferait un humain avec deux outils séparés.

| Étape | Outil | Tâche |
|-------|-------|-------|
| 1 — [ASR](vocabulaire.md#2-abréviations-et-acronymes) | Whisper (large) | Audio français → texte français |
| 2 — [MT](vocabulaire.md#2-abréviations-et-acronymes) | Marian (opus-mt-fr-en) | Texte français → texte anglais |

Ce n’est pas de la traduction [bout en bout](vocabulaire.md#2-abréviations-et-acronymes) (E2E) : l’anglais ne dépend que de la transcription intermédiaire. Si l’ASR se trompe, l’erreur se propage.

**Pourquoi cette variante ?** Les cascades restent très utilisées en production. Les comparer aux modèles E2E montre le compromis entre simplicité de déploiement, interprétabilité (on peut lire la transcription française) et score global.

**Résultat indicatif :**
- *utterance* : ~37 BLEU test — meilleur score du bench à ce jour sur ce découpage (comparaison valable seulement à protocole et segmentation identiques).

**Pistes d'amélioration :**
- Tester un modèle [ASR](vocabulaire.md#2-abréviations-et-acronymes) plus léger ou plus lourd (Whisper medium vs large) et mesurer le compromis vitesse / qualité.
- Essayer un autre traducteur texte ([MT](vocabulaire.md#2-abréviations-et-acronymes)), par exemple NLLB à la place de Marian.
- Compléter le bench en *sentence_like* pour avoir le tableau complet sur les deux découpages.
- Analyser les erreurs de transcription française qui se répercutent sur l’anglais.

**Dossier :** `4_cascade/`

---

## 5. Encodeur multimodal Speech_Text (expérimental)

**En une phrase :** même architecture que la variante 1, mais avec un encodeur Pantagruel entraîné sur parole et texte ensemble, pas sur la parole seule.

L’encodeur [`Speech_Text`](vocabulaire.md#speech_text--speech_text-multimodal) a vu du français oral et écrit pendant son pré-entraînement. L’hypothèse : ces représentations pourraient mieux servir la traduction. Le décodeur Transformer et SentencePiece restent les mêmes que en variante 1.

**Pourquoi cette variante ?** Explorer si la multimodalité au niveau de l’encodeur aide la ST — piste annoncée dans le titre du rapport de stage.

**Résultat indicatif :**
- *sentence_like* : ~8 BLEU test — nettement sous la variante 1 sur le même découpage.

**Pistes d'amélioration :**
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
        ├──► [3] Gemini (cloud)          ──► anglais  (pas d'entraînement local)
        ├──► [4] Whisper → Marian        ──► anglais  (deux modèles en série)
        └──► [5] Speech_Text + décodeur  ──► anglais  (encodeur multimodal)
```

| # | Nom court | Paradigme | Entraînement local | Intérêt principal |
|---|-----------|-----------|--------------------|-------------------|
| 1 | Transformer ST | [E2E](vocabulaire.md#2-abréviations-et-acronymes) | Oui (GPU, long) | Réplication Pantagruel |
| 2 | speechLLM B1 | E2E via LLM | Oui (projecteur surtout) | LLM + adaptateur minimal |
| 3 | Gemini | API | Non | Référence commerciale |
| 4 | Cascade | ASR puis MT | Non (inférence seule) | Baseline classique en deux temps |
| 5 | Speech_Text | E2E | Oui | Test encodeur parole+texte |

---

## Comment lire les chiffres

Trois réglages ne doivent pas être mélangés quand on compare deux lignes du tableau :

1. **La variante** (l’une des cinq ci-dessus).
2. **Le découpage audio** (*utterance* vs *sentence_like*).
3. **La taille de l’encodeur** (aujourd’hui surtout [1k h](vocabulaire.md#speech-base-1k-checkpoint-hf) ; l’article montre aussi 14k et 114k).

Un BLEU plus élevé suggère une traduction plus proche des références humaines, mais ce n’est qu’un indicateur — la qualité réelle se juge aussi en lisant des exemples (`eval/dev_predictions.txt` dans chaque run).

Pour le détail des scores, protocoles et identifiants de runs : voir le [rapport de stage](../rapport.md) et le [glossaire](vocabulaire.md).
