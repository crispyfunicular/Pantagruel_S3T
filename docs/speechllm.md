# speechLLM — explications pour la réunion

## 1. En une phrase : c’est quoi speechLLM ?

On traduit directement de l’audio français vers l’anglais en branchant trois blocs :

```text
Audio FR (16 kHz)
    → Encodeur Pantagruel (comprend la parole)
    → Projecteur (petit réseau qu’on entraîne)
    → LLM (ex. Phi-2, modèle de langue anglais)
    → Texte anglais généré
```

**Idée clé (article SLAM-ASR / « embarrassingly simple ») :** on n’entraîne **presque rien** au début — surtout le **projecteur** qui sert de « traducteur d’interface » entre l’espace parole et l’espace texte du LLM.

**Comparaison rapide avec la baseline ST classique :**

| | Baseline ST (`1_Transformer`) | speechLLM B1 (`2_speechLLM`) |
|--|-------------------------------|------------------------------|
| Tête de génération | Décodeur Transformer entraîné from scratch | LLM pré-entraîné (Phi-2) |
| Ce qu’on entraîne | Décodeur (+ parfois encodeur) | Surtout le **projecteur** |
| Tokenisation | SentencePiece (vocab 1k) | Tokenizer du LLM |

---

## 2. Gel / dégel de l’encodeur — qu’est-ce que ça veut dire ?

### L’encodeur Pantagruel, c’est quoi ?

C’est un **modèle pré-entraîné** sur beaucoup d’heures de parole française. Il transforme l’audio en une suite de vecteurs (représentations numériques). Dans S3T, on utilise par défaut **`PantagrueLLM/speech-base-1K`** (famille « Base », ~1 000 h de pré-entraînement).

### « Gelé » (`freeze_encoder: true`)

**Gelé = les poids de l’encodeur ne bougent pas pendant l’entraînement.**

- Seul le **projecteur** (et éventuellement d’autres petits modules) apprend.
- L’encodeur reste exactement comme Hugging Face l’a fourni.
- **Avantage :** moins de VRAM, entraînement plus stable, moins de risque de « casser » un bon modèle SSL.
- **Inconvénient :** l’encodeur ne s’adapte pas à m-TEDx ni à la tâche ST.

**Run S3T correspondant :** `run_002_speechllm_b1_sentence_long` (encodeur **gelé**).

### « Dégelé » (`freeze_encoder: false`)

**Dégelé = on autorise l’optimiseur à modifier aussi les poids de l’encodeur.**

- Le projecteur **et** l’encodeur apprennent ensemble.
- **Avantage :** l’encodeur peut mieux coller au corpus TEDx et à la traduction fr→en.
- **Inconvénient :** plus lourd en GPU, plus de risque d’instabilité ; le checkpoint doit bien sauvegarder les poids encodeur (voir incident run_004).

**Run S3T correspondant :** `run_005_speechllm_b1_sentence_long_unfreeze_encoder` (encodeur **dégelé**).

### Analogie simple

| Métaphore | Gelé | Dégelé |
|-----------|------|--------|
| Voiture | On ne touche qu’au GPS (projecteur) ; moteur (encodeur) d’origine | On retouche aussi le moteur pour la route TEDx |
| Équipe | Seul l’interprète (projecteur) s’adapte ; l’experte audio (encodeur) reste figée | L’experte audio affine aussi sa lecture |

### Ce que ce n’est **pas**

- **Ce n’est pas** « speechLLM 1 vs speechLLM 2 ».
- **Ce n’est pas** la comparaison **1k vs 14k vs 114k** (taille de pré-entraînement — voir section 4).
- C’est la **même** encodeur `speech-base-1K` ; seul change : est-ce qu’on la **fine-tune** ou non ?

---

## 3. Ablation « dégel encodeur » — qu’est-ce que ça veut dire ?

### Définition

Une **ablation** = une expérience où on change **une seule chose** pour mesurer son effet.

Ici : **même modèle, même données, même LLM, même budget d’entraînement** — seul le booléen **gel / dégel encodeur** change.

| Run | Encodeur | BLEU dev | BLEU test | Segmentation |
|-----|----------|----------|-----------|--------------|
| `run_002` | **gelé** | 19,99 | 15,89 | `sentence_like` |
| `run_005` | **dégelé** | 19,25 | **18,83** | `sentence_like` |

### Lecture des chiffres

- **Test :** dégel aide (+2,9 BLEU vs run_002) → généralisation un peu meilleure sur le split test.
- **Dev :** léger recul (−0,7) → possible sur-apprentissage du valid ou bruit de mesure ; à confirmer avec **2 seeds**.

### Incident run_004 (à mentionner brièvement)

`run_004` était une première tentative « dégel » : l’entraînement tournait, mais l’**évaluation était fausse** (BLEU ~0) parce que le fichier `best.pt` ne contenait **pas** les poids `encoder.*`. Corrigé avant `run_005`. **Seul run_005 fait foi** pour l’ablation dégel.

---

## 4. Encodeurs 14k / 114k — qu’est-ce que c’est ?

### Ce que « 14k » et « 114k » signifient

Ce ne sont **pas** des noms de corpus m-TEDx ni de splits train/valid/test.

Ce sont des **durées de pré-entraînement** de l’encodeur Pantagruel **avant** qu’on l’utilise dans S3T :

| Nom Hugging Face | Nom papier | Heures de parole FR (pré-train) | Taille |
|------------------|------------|----------------------------------|--------|
| `speech-base-1K` | Pantagruel-B-1k | ~1 000 h | Base |
| `speech-large-14K` | Pantagruel-L-14k | ~14 000 h (+ LeBenchmark) | Large |
| `speech-large-114K` | Pantagruel-L-114k | ~114 000 h (+ INA-100k) | Large |

**Pour ST fr→en utterance**, le papier (Table 8) reporte par exemple :

- B-1k ≈ **17,5** BLEU test  
- L-14k ≈ **24,0**  
- L-114k ≈ **25,2**

### Ce qu’on a fait aujourd’hui en speechLLM

Tous les runs speechLLM reportés utilisent encore **`speech-base-1K`** (gelé ou dégelé).

Les configs **Large** existent déjà pour la suite :

- `run_012_speechllm_b1_utterance_large_14k`
- `run_013_speechllm_b1_utterance_large_114k`

### Ce que ça implique pour speechLLM

Changer 1k → 14k, ce n’est **pas** juste cocher une case à l’éval : il faut **ré-entraîner** le projecteur (dimensions internes peuvent changer si on change de famille d’encodeur). Chaque couple `(encodeur, LLM)` = un run d’entraînement dédié.

---

## 5. Contrat d’artifacts — qu’est-ce que c’est ?

### Définition simple

Le **contrat d’artifacts** = la **liste des fichiers** qu’un run doit produire pour être **reproductible, comparable et auditable**.

Sans ça, on ne peut pas :

- retrouver la config exacte,
- recharger le bon checkpoint,
- comparer deux runs au même SacreBLEU,
- expliquer un chiffre six mois plus tard.

### Où ça vit

Pour chaque run : `runs/fr-en/<run_id>/`

### Fichiers attendus (speechLLM)

| Fichier | Rôle |
|---------|------|
| `config.yaml` | Copie figée de la config YAML du run |
| `checkpoints/best.pt` | Meilleur modèle (selon BLEU **dev**) |
| `checkpoints/last.pt` | Dernier état à la fin du train |
| `train.log` | Une ligne JSON par update (loss, BLEU dev partiel…) |
| `metrics.json` | Résumé (durée GPU, best BLEU dev, etc.) |
| `eval/dev_predictions.txt` | Une ligne = une hypothèse anglaise (valid) |
| `eval/test_predictions.txt` | Idem pour test |
| `eval/sacrebleu_dev.txt` | Score BLEU + **signature** SacreBLEU |
| `eval/sacrebleu_test.txt` | Idem test |
| `eval/protocol.json` | Protocole d’éval figé (décodage, version…) |

### Agrégat projet

`runs/experiments_tracking.csv` = tableau de bord (une ligne par run : pipeline, segment_mode, BLEU, statut, notes).

### Exemple concret

Run `run_005` **ok** = tous ces fichiers présents + BLEU signé → on peut le citer en réunion.

Run `run_004` **invalid_eval** = train OK mais éval non fiable → **ne pas** mettre ses BLEU dans un tableau comparatif.

---

## 6. Segmentation : `utterance` vs `sentence_like`

Deux façons de découper le **même** corpus m-TEDx :

| Mode | Idée | Usage S3T |
|------|------|-----------|
| **`utterance`** | Segments natifs du corpus (comme le papier Pantagruel) | Bench Table 8 |
| **`sentence_like`** | Fusion de segments pour approcher des « phrases » (~10 s) | Runs historiques juin 2026 |

**Règle d’or :** ne pas comparer un modèle entraîné en `sentence_like` avec une éval en `utterance` (ou l’inverse).

Runs speechLLM **sentence_like** : faits (`run_002`, `run_005`).  
Run speechLLM **utterance** : **à lancer** (`run_003`).

---

## 7. Décodage speechLLM (paramètres qui comptent)

Protocole figé pour les runs de référence :

| Paramètre | Valeur retenue | Effet |
|-----------|----------------|-------|
| `beam_size` | **1** (greedy) | Une seule hypothèse ; stable pour nous |
| `max_new_tokens` | **48** | Limite la longueur générée |
| Prompt | `Translate the French speech to English.` | Identique Gemini / ST pour comparer |

**Attention :** augmenter beam ou max tokens peut provoquer des **boucles** (`me me me…`, `iveive…`) et faire chuter le BLEU. Ce n’est pas un bug SacreBLEU : c’est le générateur qui dérape.

---

## 8. Tableau de bord — où on en est (juin 2026)

### Fait (speechLLM, `sentence_like`)

| Run | Question testée | BLEU dev | BLEU test | Statut |
|-----|-----------------|----------|-----------|--------|
| `run_002` | B1 minimal (encodeur gelé) | 19,99 | 15,89 | ok |
| `run_005` | Ablation dégel encodeur | 19,25 | **18,83** | ok |
| `run_004` | Première tentative dégel | 0,30 | 0,46 | **invalid_eval** |

### Pas encore fait (speechLLM)

| Priorité | Run | Objectif |
|----------|-----|----------|
| **P0** | `run_003_speechllm_b1_utterance_long` | Même B1 gelé, segmentation **utterance** (comparable bench Pantagruel) |
| **P1** | `run_012` / `run_013` | Encodeur **Large** 14k / 114k + utterance |
| **P2** | 2e seed sur meilleur réglage | Robustesse statistique |
| **P3** | Autre LLM (Llama-3.2-3B, etc.) | B2bis — un entraînement projecteur **par** LLM |

### Contexte autres variantes (utterance, pour situer speechLLM)

| Variante | BLEU test utterance |
|----------|---------------------|
| Cascade Whisper→Marian | **37,41** |
| Gemini 2.5 Flash | **33,72** |
| ST Transformer B-1k (`run_002`, échec) | 3,79 |
| ST Transformer B-1k v2 (`run_004`, terminé) | **16,68** |
| **speechLLM** | *pas encore mesuré en utterance* |

speechLLM n’a pas encore de score utterance : **priorité réunion** = lancer `run_003` (GPU libre après `run_004` ST v2).

---

## 9. Prochaines commandes (tour GPU)

### Run utterance B1 (gelé) — priorité immédiate

```bash
cd ~/S3T && source .venv/bin/activate
python 2_speechLLM/pipeline.py run \
  --config 2_speechLLM/configs/fr-en/b1_utterance_long.yaml \
  --run-id run_003_speechllm_b1_utterance_long -v
```

Durée indicative : **~3–4 h GPU** (20k updates).

### Après run_003 — ablation dégel en utterance (à créer si besoin)

Dupliquer la config en `freeze_encoder: false` + nouveau `run_id` (ex. `run_006_…`), **ne pas** réutiliser un checkpoint sentence_like.

### Encodeurs Large (plus tard)

```bash
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-14k
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-114k
```

---

## 10. Messages clés pour la réunion (3 bullets)

1. **speechLLM B1 fonctionne** sur m-TEDx fr→en (`sentence_like`) : ~16–20 BLEU test selon gel/dégel ; le dégel encodeur améliore le test (`run_005` vs `run_002`).
2. **Ce qu’on n’a pas encore prouvé** : performance en **utterance** (bench papier) ni avec encodeurs **14k/114k**.
3. **Prochaine étape concrète** : `run_003` utterance + mise à jour `experiments_tracking.csv` + 3 exemples qualitatifs pour éviter les boucles de génération.

---

## 11. Liens doc / code

| Document | Contenu |
|----------|---------|
| [2_speechLLM/README.md](../2_speechLLM/README.md) | Usage CLI |
| [docs/plan_migration_speechllm.md](plan_migration_speechllm.md) | Roadmap B1/B2 |
| [docs/protocole_evaluation.md](protocole_evaluation.md) | SacreBLEU, décodage figé |
| [docs/protocole_utterance_pantagruel.md](protocole_utterance_pantagruel.md) | Bench utterance |
| [rapport.md](../rapport.md) §5 | Tableaux comparatifs tous pipelines |
| [runs/experiments_tracking.csv](../runs/experiments_tracking.csv) | Suivi chiffré |
