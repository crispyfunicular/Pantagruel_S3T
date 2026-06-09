# Cinq exemples de traductions — lecture qualitative

Ce document complète [variantes.md](variantes.md) : au-delà du score BLEU, on peut **lire** ce que chaque système produit sur les mêmes extraits audio.

> **Vocabulaire** — Les termes techniques sont définis dans le [glossaire](vocabulaire.md).

---

## Protocole de ces exemples

| Paramètre | Valeur |
|-----------|--------|
| **Corpus** | m-TEDx fr→en ([OpenSLR SLR100](https://www.openslr.org/100)) |
| **Découpage** | *utterance* (segments natifs du corpus) |
| **Split** | `valid` (souvent appelé *dev* dans les fichiers d’évaluation) |
| **Exposé** | Même intervention TED (`9fxo9YJhnG8` — témoignage sur l’amour) |
| **Français affiché** | Texte source aligné sur l’audio dans le [manifest](vocabulaire.md#manifest) (`src_text`) — ce n’est **pas** une transcription ASR, sauf pour la cascade (Whisper produit d’abord du français, puis Marian traduit ; seule la sortie anglaise finale est reportée ici). |
| **Référence anglaise** | Traduction humaine m-TEDx (`tgt_text`) |

Les traductions ci-dessous sont extraites telles quelles des fichiers `eval/` de chaque run (juin 2026). Elles peuvent contenir des coquilles, répétitions ou sorties incomplètes — c’est justement l’intérêt d’une relecture qualitative.

### Runs utilisés (variantes 1 à 4 + Gemini)

| # | Variante | Run | Encodeur / outil |
|---|----------|-----|------------------|
| — | **Référence m-TEDx** | — | Traduction humaine du corpus |
| 1 | Transformer ST | `run_004_transformer_baseline_utterance_v2` | Pantagruel B-1k + décodeur 6 couches |
| 2 | speechLLM B1 | `run_012_speechllm_b1_utterance_large_14k` | Pantagruel L-14k + Phi-2 gelé |
| 3 | Gemini 2.5 Flash | `run_001_gemini_flash_utterance_full` | API `gemini-2.5-flash` |
| 4 | Cascade ASR→MT | `run_001_cascade_utterance` | Whisper large-v3 → Marian fr-en |

La **variante 5** (Speech_Text) a été évaluée en *sentence_like* (segments fusionnés) : voir [en bas de page](#variante-5--segments-fusionnés-sentence_like).

---

## Phrase 1 — Le coup de foudre

**Identifiant :** `9fxo9YJhnG8_5`

**Français (m-TEDx)**  
Il m’a regardée avec ses grands yeux bleus et on s’est aimés au premier regard.

| Source | Traduction anglaise |
|--------|---------------------|
| **Référence m-TEDx** | His huge blue eyes watched me back, and we fell in love at first sight. |
| **1 — Transformer** | He looked at me with his great blue eyes and we were loved at the first look. |
| **2 — speechLLM** | He looked at me with his big blue eyes, and we fell in love at first sight. |
| **3 — Gemini 2.5** | He looked at me with his big blue eyes, and we fell in love at first sight. |
| **4 — Cascade** | Looked at me with his big blue eyes and we loved each other at first glance. |

**Lecture rapide :** speechLLM et Gemini retrouvent une formulation proche de la référence ; le Transformer glisse sur la tournure (*were loved at the first look*) ; la cascade omet le sujet en tête de phrase.

---

## Phrase 2 — L’émerveillement au réveil

**Identifiant :** `9fxo9YJhnG8_6`

**Français (m-TEDx)**  
Aujourd’hui encore, quand je me lève le matin, je regarde mon homme à côté de moi et je suis toujours émerveillée.

| Source | Traduction anglaise |
|--------|---------------------|
| **Référence m-TEDx** | Still today, when I wake up in the morning, I watch my man lying next to me and I am still amazed. |
| **1 — Transformer** | Today, when I get riding up in the morning, I look at my man, and I am still awaken. |
| **2 — speechLLM** | Even today, when I wake up in the morning, I look at my man and I am always amazed. |
| **3 — Gemini 2.5** | Even today, when I get up in the morning, I look at my man next to me and I am always amazed. |
| **4 — Cascade** | Even today, when I get up in the morning, I look at my man next to me and I am always amazed. |

**Lecture rapide :** trois pistes (speechLLM, Gemini, cascade) sont très proches du sens ; le Transformer invente *riding up* et *awaken*.

---

## Phrase 3 — Papillons dans le ventre

**Identifiant :** `9fxo9YJhnG8_7`

**Français (m-TEDx)**  
J’ai mon cœur qui palpite et encore les papillons dans le ventre, presque 24 heures sur 24.

| Source | Traduction anglaise |
|--------|---------------------|
| **Référence m-TEDx** | My heart rate rises and I am still head over heals, almost 24 hours a day. |
| **1 — Transformer** | I have my heart that papers, and there are still 24-mands in the stomach. |
| **2 — speechLLM** | My heart is fluttering, I still have butterflies in my stomach, almost 24 hours a day. |
| **3 — Gemini 2.5** | My heart is pounding, I still have butterflies in my stomach, almost 24/7. |
| **4 — Cascade** | I’ve got my heart palpitating, I still have butterflies in my belly, almost 24 hours a day. |

**Lecture rapide :** l’image des *butterflies* est bien rendue par speechLLM, Gemini et la cascade ; le Transformer déraille (*papers*, *24-mands*).

---

## Phrase 4 — Rien n’a changé

**Identifiant :** `9fxo9YJhnG8_8`

**Français (m-TEDx)**  
Rien n’a changé.

| Source | Traduction anglaise |
|--------|---------------------|
| **Référence m-TEDx** | Nothing has changed. |
| **1 — Transformer** | Nothing changed. |
| **2 — speechLLM** | *(répétitions)* « Nothing has changed. » × plusieurs fois |
| **3 — Gemini 2.5** | Nothing has changed! |
| **4 — Cascade** | Nothing has changed. |

**Lecture rapide :** segment court : la cascade et Gemini sont corrects ; speechLLM boucle sur la même phrase — symptôme de décodage instable sur les segments brefs.

---

## Phrase 5 — Comme au premier jour

**Identifiant :** `9fxo9YJhnG8_9`

**Français (m-TEDx)**  
Rien n’a changé : je l’aime comme au premier jour.

| Source | Traduction anglaise |
|--------|---------------------|
| **Référence m-TEDx** | Nothing has changed: I love him as much I did the first day. |
| **1 — Transformer** | Nothing to change, I love it as in the first day. |
| **2 — speechLLM** | the first day. *(sortie tronquée)* |
| **3 — Gemini 2.5** | Nothing has changed, I love him/her like on the first day. |
| **4 — Cascade** | Nothing has changed, I love her like on the first day. |

**Lecture rapide :** Gemini et la cascade restituent l’essentiel ; le Transformer confond *it* / *him* ; speechLLM ne renvoie qu’un fragment.

---

## Variante 5 — segments fusionnés (*sentence_like*)

L’encodeur **Speech_Text** n’a pas été évalué sur les mêmes utterances que les quatre autres variantes : le run `run_001_pantagruel_multimodal` utilise des **segments plus longs** (fusion de prises voisines, ~10–15 s). On ne peut donc pas aligner ligne à ligne les tableaux ci-dessus.

Exemple comparable (même exposé TED, segment fusionné) :

**Identifiant :** `9fxo9YJhnG8_m1` — **Run :** `run_001_pantagruel_multimodal`

**Français (m-TEDx, fusionné)**  
Aujourd’hui encore, quand je me lève le matin, je regarde mon homme à côté de moi et je suis toujours émerveillée. J’ai mon cœur qui palpite et encore les papillons dans le ventre, presque 24 heures sur 24.

| Source | Traduction anglaise |
|--------|---------------------|
| **Référence m-TEDx** | Still today, when I wake up in the morning, I watch my man lying next to me and I am still amazed. My heart rate rises and I am still head over heals, almost 24 hours a day. |
| **5 — Speech_Text + ST** | Today, when I look at the morning, I look at my man, and I always looked my heart. I’m still pair. I’m still paper. I’m still path in the body. I’m still a few hours in the body. |

Ce run reste expérimental (BLEU test ~8 en *sentence_like*) : la sortie mélange anglais approximatif et répétitions (*still paper*, *still pair*).

---

## Comment utiliser cette page

- Comparer **deux variantes** sur une même ligne : même audio, même référence, seule la recette change.
- Ne pas sur-interpréter **cinq phrases** : le BLEU corpus sur tout le `valid` / `test` reste la métrique principale (voir [variantes.md § Synthèse](variantes.md#synthèse-des-meilleurs-scores)).
- Pour aller plus loin : fichiers `eval/dev_predictions.txt` ou `eval/dev_review.tsv` dans chaque run sous `runs/fr-en/<run_id>/`.

---

[← Retour aux cinq variantes](variantes.md) · [Glossaire](vocabulaire.md) · [Rapport de stage](../rapport.md)
