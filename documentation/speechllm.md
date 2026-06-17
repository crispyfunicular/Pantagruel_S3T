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

## 2. Journal des expériences — ce qui a été fait, en cours, et à faire

Cette section est la **référence opérationnelle** : chaque ligne correspond à un run réel (ou planifié) avec son objectif, ce qui change par rapport aux autres, les paramètres retenus et le statut.

### 2.1 Vue d’ensemble (tous les runs speechLLM)

#### Tableau A — paramètres et scores

| Run ID | Segm. | Encodeur | Gel enc. | Updates | Décodage | BLEU dev | BLEU test | Statut |
|--------|-------|----------|----------|---------|----------|----------|-----------|--------|
| `run_001_speechllm_b1_sentence` | sent. | B-1k | gelé | 500 | beam 4 / 128 | ~0,42* | — | pilote |
| `run_002_speechllm_b1_sentence_long` | sent. | B-1k | gelé | 20k | beam 1 / 48 | **19,99** | **15,89** | ok |
| `run_004_speechllm_…_unfreeze_encoder_v2` | sent. | B-1k | dégelé | 20k | beam 1 / 48 | 0,30 | 0,46 | invalid_eval |
| `run_005_speechllm_…_unfreeze_encoder` | sent. | B-1k | dégelé | 20k | beam 1 / 48 | **19,25** | **18,83** | ok |
| `run_003_speechllm_b1_utterance_long` | utt. | B-1k | gelé | 20k | beam 1 / 48 | **10,00** | **7,47** | ok (tour) |
| `run_012_speechllm_b1_utterance_large_14k` | utt. | L-14k | gelé | 20k | beam 1 / 48 | — | — | à lancer |
| `run_013_speechllm_b1_utterance_large_114k` | utt. | L-114k | gelé | 20k | beam 1 / 48 | — | — | à lancer |
| `run_006_*` (prévu) | utt. | B-1k | dégelé | 20k | beam 1 / 48 | — | — | à créer |

Légende : **sent.** = `sentence_like` ; **utt.** = `utterance` ; LLM = Phi-2 gelé partout ; noms complets des `run_id` en §2.3.

\* `run_001` : BLEU partiel (20 batches valid), pas de SacreBLEU corpus.

#### Tableau B — ce qu’on cherche à savoir

| Run | Question | Par rapport à quoi ? | Explication |
|-----|-------------------------------|----------------------|------------------------|
| `run_001` | Est-ce que la chaîne complète **fonctionne** ? | — (premier essai) | *« On a vérifié que audio → Pantagruel → projecteur → Phi-2 tourne. Seulement 500 pas : pas un vrai score. »* |
| `run_002` | Un **petit projecteur seul** suffit-il pour traduire ? | Référence B1 | *« On n’entraîne que le projecteur ; Pantagruel et Phi-2 restent figés. Résultat : ~16 BLEU test (sentence_like). »* |
| `run_004` | Faire apprendre **l’encodeur aussi** améliore-t-il le BLEU ? | vs `run_002` (+ dégel encodeur) | *« Même idée que run_005, mais la sauvegarde était cassée : chiffres invalides. »* |
| `run_005` | Le **dégel encodeur** vaut-il le coup ? | vs `run_002` (seul le gel change) | *« +3 BLEU test vs run_002 ; léger recul sur le valid. Le dégel aide à généraliser. »* |
| `run_003` | Quel BLEU en **utterance** (comme le papier) ? | vs `run_002` (+ segmentation) | *« Premier score utterance : 7,5 BLEU test — bien sous notre ST (16,7) et le papier (~17,5). Hypothèses trop longues à inspecter. »* |
| `run_012` | Un encodeur **14k h** améliore-t-il speechLLM ? | vs `run_003` (+ taille encodeur) | *« Le papier montre ~24 BLEU en ST avec L-14k ; on teste si speechLLM en profite aussi. »* |
| `run_013` | Idem avec encodeur **114k h** ? | vs `run_012` (+ taille encodeur) | *« Même protocole, encodeur le plus gros du papier (~25 BLEU ST). »* |
| `run_006` | Le dégel aide-t-il **en utterance** ? | vs `run_003` (+ dégel, comme run_005) | *« run_003 faible : tester le dégel encodeur (comme run_005) avant d’autres leviers. »* |

**Règles de lecture :**

- Ne comparer deux runs que si la **segmentation** est identique (`utterance` ≠ `sentence_like`).
- Ne pas citer les BLEU de `run_004` speechLLM (bug checkpoint).
- Attention aux homonymes : `run_004_transformer_*` = baseline **ST**, pas speechLLM.

### 2.2 Paramètres communs à presque tous les runs B1

Ces valeurs sont **identiques** sauf mention contraire dans la fiche run :

| Paramètre | Valeur B1 | Rôle |
|-----------|-----------|------|
| Paire linguistique | fr→en | m-TEDx |
| Audio | 16 kHz | standard S3T |
| Encodeur (défaut) | `PantagrueLLM/speech-base-1K` | ~1 000 h pré-train FR |
| LLM | `microsoft/phi-2` | gelé (`freeze_llm: true`) |
| Downsampling | `downsample_k: 5` | regroupe 5 frames encodeur |
| Projecteur | hidden 2048, Linear→ReLU→Linear | **seul module entraîné** si encodeur gelé |
| Prompt | `Translate the French speech to English.` | identique Gemini / ST |
| Seed | 42 | reproductibilité |
| Batch effectif | 2 × accum 4 = **8** (1k) ; 1 × 8 = **8** (Large) | même batch effectif visé |
| Warmup | 1 000 updates | sauf run_001 (100) |
| Éval intermédiaire | tous les 1 000 updates, 20 batches valid | sélection `best.pt` sur BLEU **dev** |
| AMP | fp16 | tour GPU |
| Décodage officiel | `beam_size: 1`, `max_new_tokens: 48` | figé juin 2026 (évite boucles) |

### 2.3 Runs terminés — détail

#### `run_001_speechllm_b1_sentence` — smoke test (mai 2026)

| | |
|--|--|
| **Objectif** | Vérifier que le pipeline speechLLM tourne bout en bout (chargement Pantagruel + Phi-2, train, checkpoint). |
| **Config** | Figée dans `runs/fr-en/run_001_speechllm_b1_sentence/config.yaml` (smoke test ~500 updates ; proche du template `b1.yaml` mais manifests `sentence_like`) |
| **Ce qui diffère des runs sérieux** | Budget **500 updates** seulement ; décodage **beam 4 / 128 tokens** (non retenu ensuite). |
| **Paramètres train** | LR `1e-4`, WD `0`, encodeur + LLM gelés, manifests `sentence_like`. |
| **Résultat** | `best_bleu_dev ≈ 0,42` sur échantillon valid réduit — **pas** de SacreBLEU corpus signé. |
| **Conclusion** | Run pilote ; les comparaisons officielles partent de `run_002`. |

#### `run_002_speechllm_b1_sentence_long` — référence B1 gelé (juin 2026)

| | |
|--|--|
| **Objectif** | Première mesure **fiable** du paradigme B1 (projecteur seul) sur m-TEDx fr→en. |
| **Config** | `2_speechLLM/configs/fr-en/b1_sentence_long.yaml` |
| **Hypothèse** | Un petit projecteur peut aligner l’audio sur l’espace Phi-2 sans toucher à l’encodeur ni au LLM. |
| **Paramètres clés** | `freeze_encoder: true`, `freeze_llm: true`, **20 000 updates**, LR `1e-4`, WD `0`, décodage **beam 1 / 48 tok**. |
| **Données** | `datasets/manifests_sentence/fr-en/` (`sentence_like` — segments fusionnés ~10 s). |
| **Résultat** | BLEU dev **19,99** / test **15,89** — statut **ok**. |
| **Lecture** | Bon dev, écart dev→test (−4,1) : possible effet `max_new_tokens=48`, sur-adaptation valid, ou segmentation. |
| **Référence** | Run de référence pour toute ablation « une seule chose change ». |

#### `run_004_speechllm_b1_sentence_long_unfreeze_encoder_v2` — échec technique (juin 2026)

| | |
|--|--|
| **Objectif** | Même expérience que `run_005` : tester le **dégel encodeur** sur `sentence_like`. |
| **Config** | `b1_sentence_long_unfreeze_encoder.yaml` (même YAML que run_005). |
| **Paramètres** | Identiques à `run_005` : `freeze_encoder: false`, LR `5e-5`, WD `0,01`, 20k updates. |
| **Problème** | L’entraînement converge (BLEU dev partiel ~26 en cours de train), mais `best.pt` ne sauvegardait **pas** les poids `encoder.*`. |
| **Résultat éval** | BLEU dev **0,30** / test **0,46** — statut **invalid_eval**. |
| **Correctif** | Persistance checkpoint corrigée (`encoder.*` inclus si dégelé) + tests `tests/test_speechllm_checkpoint.py`. |
| **À retenir** | **Ne pas** utiliser ces chiffres ; seul `run_005` fait foi pour l’ablation dégel. |

#### `run_005_speechllm_b1_sentence_long_unfreeze_encoder` — ablation dégel encodeur (juin 2026)

| | |
|--|--|
| **Objectif** | Mesurer l’effet du **fine-tuning encodeur** à budget et données identiques à `run_002`. |
| **Config** | `2_speechLLM/configs/fr-en/b1_sentence_long_unfreeze_encoder.yaml` |
| **Seule différence vs run_002** | `freeze_encoder: false` → projecteur **+** encodeur entraînés. |
| **Ajustements liés au dégel** | LR **`5e-5`** (÷2 vs run_002), WD **`0,01`** (régularisation). |
| **Résultat** | BLEU dev **19,25** (−0,7) / test **18,83** (**+2,9**) — statut **ok**. |
| **Lecture** | Le dégel améliore la **généralisation test** au prix d’un léger recul dev ; à confirmer avec une 2e seed. |

#### `run_003_speechllm_b1_utterance_long` — bench utterance (juin 2026)

| | |
|--|--|
| **Objectif** | Obtenir le **premier SacreBLEU utterance** speechLLM, comparable au papier Pantagruel (Table 8) et aux autres variantes S3T. |
| **Config** | `2_speechLLM/configs/fr-en/b1_utterance_long.yaml` |
| **Différence principale vs run_002** | Segmentation **`utterance`** : manifests `datasets/manifests/fr-en/` (segments natifs m-TEDx). |
| **Paramètres** | Identiques à `run_002` (B-1k gelé, Phi-2 gelé, 20k updates, LR `1e-4`, beam 1 / 48 tok). |
| **Machine** | Tour GPU Modyco (`mpellissier@10.8.0.2`), terminé **2026-06-05** (~1 h 51 train + ~11 min éval ; ~31 Go VRAM en train). |
| **Résultat** | BLEU dev **10,00** / test **7,47** — statut **ok**. |
| **Lecture** | Fort écart vs `run_002` sentence_like (**15,89** test) : la segmentation compte ; speechLLM **sous** la ST utterance v2 (**16,68**) et loin du papier (~17,5). |
| **Signal qualitatif** | Hypothèses **~2,3× plus longues** que les références (`hyp_len/ref_len` SacreBLEU) ; TER > 100 % — relecture `eval/dev_predictions.txt` prioritaire. |
| **Artifacts** | Sur la tour : `runs/fr-en/run_003_…/{checkpoints,train.log,eval/}` — rsync `eval/` vers poste local + `update_experiments_tracking.py`. |

### 2.5 Manipulations annexes (sans run dédié)

| Manipulation | Ce qu’on a testé | Résultat |
|--------------|------------------|----------|
| **Décodage large** | `beam_size ≥ 4`, `max_new_tokens ≥ 128` (configs type `b1.yaml`) | Boucles (`me me me…`, `iveive…`), BLEU effondré — **anti-pattern** |
| **Décodage retenu** | `beam_size: 1`, `max_new_tokens: 48` | Stable pour `run_002` et `run_005` — **protocole figé** |
| **Bug checkpoint run_004** | Sauvegarde partielle en mode dégel | Éval invalide — corrigé avant `run_005` |
| **1re tentative run_003** | Lancement sans config sur le tour | Échec immédiat — corrigé par rsync `b1_utterance_long.yaml` |

### 2.6 Ce qu’il reste à faire — par priorité

| Priorité | Run / action | Objectif | Différence vs l’existant | Config / commande |
|----------|--------------|----------|--------------------------|-------------------|
| **P0** | Rsync + relecture `run_003` | Comprendre le BLEU faible (7,47) | Hypothèses longues, erreurs systématiques | `eval/dev_predictions.txt` |
| **P1** | `run_006_*` (à nommer) | Ablation **dégel** en utterance | Comme run_005 mais manifests utterance ; **nouveau run_id**, pas de reprise checkpoint sentence_like | Dupliquer `b1_utterance_long.yaml` → `freeze_encoder: false`, LR `5e-5`, WD `0,01` |
| **P1** | `run_012` / `run_013` | Encodeurs **Large** 14k / 114k | Change `encoder_name` ; batch **1×8** (VRAM) ; **ré-entraîne** le projecteur | `b1_utterance_large_14k.yaml`, `b1_utterance_large_114k.yaml` |
| **P2** | 2e seed | Robustesse statistique | Même config, `seed ≠ 42` | Dupliquer meilleure config |
| **P3** | B2 / autre LLM | Llama-3.2-3B, Mistral-7B… | **Un entraînement projecteur par couple (encodeur, LLM)** | Roadmap `plan_migration_speechllm.md` |
| **P3** | Relecture qualitative | Boucles, longueur, erreurs systématiques | — | `eval/dev_predictions.txt` vs Gemini |

**Ordre recommandé :** relecture qualitative `run_003` → ablation dégel utterance (`run_006`) → encodeurs Large (`run_012`/`013`, priorité encadrant) → seeds / autres LLM.

### 2.7 Arbre de décision — qu’est-ce qui change entre deux runs ?

```text
run_002 (référence sentence_like, gelé)
    │
    ├─ change segmentation ──────────────► run_003 (utterance, gelé)     [ok — 7,47 test]
    │
    ├─ change freeze_encoder ────────────► run_005 (sentence_like, dégelé)
    │                                        └─ run_004 = même chose, bug checkpoint
    │
    ├─ change encoder_name (1k → 14k/114k) ► run_012 / run_013 (utterance, gelé)
    │
    ├─ change llm_name ──────────────────► B2 (à planifier)
    │
    └─ change seed ──────────────────────► run_*_seed2 (à planifier)
```

**Ce qui ne doit pas changer** pour une comparaison valide : prompt, protocole SacreBLEU, `segment_mode`, décodage (beam 1 / 48), sauf si l’ablation porte précisément sur le décodage.

---

## 3. Gel / dégel de l’encodeur — qu’est-ce que ça veut dire ?

Voir le journal §2.3 (`run_002` vs `run_005`) pour les chiffres. Résumé conceptuel :

### L’encodeur Pantagruel

Modèle pré-entraîné sur ~1 000 h de parole française (`PantagrueLLM/speech-base-1K`). Il transforme l’audio en vecteurs.

| Mode | YAML | Effet | Run S3T |
|------|------|-------|---------|
| **Gelé** | `freeze_encoder: true` | Seul le projecteur apprend | `run_002`, `run_003` |
| **Dégelé** | `freeze_encoder: false` | Projecteur + encodeur apprennent | `run_005` (valide), `run_004` (bug) |

**Ce que ce n’est pas :** la comparaison 1k / 14k / 114k (taille de pré-entraînement — §4), ni « speechLLM 1 vs 2 » du dépôt (`1_Transformer` vs `2_speechLLM`).

---

## 4. Encodeurs 14k / 114k — qu’est-ce que c’est ?

| Nom Hugging Face | Heures pré-train FR | BLEU test ST utterance (papier Table 8) |
|------------------|---------------------|-------------------------------------------|
| `speech-base-1K` | ~1 000 h | ~17,5 |
| `speech-large-14K` | ~14 000 h | ~24,0 |
| `speech-large-114K` | ~114 000 h | ~25,2 |

**État speechLLM :** utterance B-1k mesuré (`run_003` : **7,47** test) ; sentence_like meilleur (`run_005` : **18,83**). Encodeurs **Large** (`run_012`, `run_013`) pas encore lancés — chaque taille impose un **nouvel entraînement** du projecteur.

---

## 5. Contrat d’artifacts

Chaque run valide produit sous `runs/fr-en/<run_id>/` :

| Fichier | Rôle |
|---------|------|
| `config.yaml` | Config figée |
| `checkpoints/best.pt` | Meilleur modèle (BLEU dev) |
| `train.log` | JSON par update |
| `eval/sacrebleu_*.txt` | BLEU + signature SacreBLEU |
| `eval/protocol.json` | Protocole d’éval figé |

Agrégat : `runs/experiments_tracking.csv`.

---

## 6. Segmentation : `utterance` vs `sentence_like`

| Mode | Manifests | Usage |
|------|-----------|-------|
| **`utterance`** | `datasets/manifests/fr-en/` | Bench Pantagruel / Table 8 |
| **`sentence_like`** | `datasets/manifests_sentence/fr-en/` | Runs historiques juin 2026 |

**Règle d’or :** ne jamais évaluer un modèle entraîné en `sentence_like` sur des manifests `utterance` (ou l’inverse).

---

## 7. Décodage speechLLM

| Paramètre | Valeur retenue | Pourquoi |
|-----------|----------------|----------|
| `beam_size` | **1** (greedy) | Stable ; beam large → boucles |
| `max_new_tokens` | **48** | Limite longueur ; 128+ → répétitions |
| Prompt | `Translate the French speech to English.` | Comparabilité inter-variantes |

Protocole complet : [protocole_evaluation.md](protocole_evaluation.md).

---

## 8. Contexte bench utterance (autres pipelines)

Bench **utterance** (même segmentation) — ordre décroissant BLEU test :

| Variante | Run | BLEU dev | BLEU test |
|----------|-----|----------|-----------|
| Cascade Whisper→Marian | `run_001_cascade_utterance` | 38,17 | **37,41** |
| Gemini 2.5 Flash | `run_001_gemini_flash_utterance_full` | 33,76 | **33,72** |
| ST Transformer B-1k v2 | `run_004_transformer_baseline_utterance_v2` | 16,84 | **16,68** |
| *Pantagruel papier Table 8* | — | — | *~17,5* |
| **speechLLM B1** | `run_003_speechllm_b1_utterance_long` | **10,00** | **7,47** |
| ST Transformer B-1k (échec) | `run_002_transformer_baseline_utterance` | 3,90 | 3,79 |

**Lecture :** speechLLM utterance est **sous** la ST S3T et le papier ; la cascade/Gemini ne sont pas comparables au paradigme B1 (projecteur seul) mais fixent le plafond pratique sur m-TEDx.

---

## 9. Commandes utiles

### Récupérer `run_003` (tour → poste local)

```bash
mkdir -p runs/fr-en/run_003_speechllm_b1_utterance_long/eval
rsync -avz mpellissier@10.8.0.2:~/S3T/runs/fr-en/run_003_speechllm_b1_utterance_long/eval/ \
  runs/fr-en/run_003_speechllm_b1_utterance_long/eval/
# optionnel : checkpoint pour infer local
rsync -avz mpellissier@10.8.0.2:~/S3T/runs/fr-en/run_003_speechllm_b1_utterance_long/checkpoints/best.pt \
  runs/fr-en/run_003_speechllm_b1_utterance_long/checkpoints/
python scripts_communs/update_experiments_tracking.py \
  --run-dir runs/fr-en/run_003_speechllm_b1_utterance_long
```

### Inférence locale (1 WAV) — checkpoint + GPU ≥ ~11 Go ou `--prefer-cpu`

```bash
source .venv/bin/activate
python 2_speechLLM/pipeline.py infer \
  --checkpoint runs/fr-en/run_003_speechllm_b1_utterance_long/checkpoints/best.pt \
  --config 2_speechLLM/configs/fr-en/b1_utterance_long.yaml \
  --input-audio datasets/processed/fr-en/test/<fichier>.wav -v
```

Déploiement code sur serveur IMAG : `./scripts/aker.sh rsync-code` (voir [README.md](../README.md) § Serveur IMAG aker).

### Prochain run GPU (tour ou cluster) — encodeurs Large

```bash
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-14k
bash scripts/run_pantagruel_encoder_scale_utterance.sh speechllm-114k
```

---

## 10. Résumé

1. **Fait et valide** : B1 gelé `sentence_like` (**15,89** test, `run_002`) ; ablation dégel (**18,83** test, `run_005`, +2,9 vs gelé) ; utterance B1 (**7,47** test, `run_003`).
2. **Fait mais invalide** : `run_004` speechLLM (bug checkpoint) — exclu des tableaux.
3. **Constat utterance** : speechLLM (**7,47**) << ST S3T (**16,68**) << papier (~17,5) ; hypothèses probablement trop longues — relecture qualitative avant nouveau run GPU.
4. **À faire** : rsync `run_003` → ablation dégel utterance (`run_006`) → encodeurs 14k/114k → seeds / autres LLM.

---

## 11. Liens doc / code

| Document | Contenu |
|----------|---------|
| [2_speechLLM/README.md](../2_speechLLM/README.md) | Usage CLI |
| [plan_migration_speechllm.md](plan_migration_speechllm.md) | Roadmap B1/B2 |
| [protocole_evaluation.md](protocole_evaluation.md) | SacreBLEU, décodage figé |
| [protocole_utterance_pantagruel.md](protocole_utterance_pantagruel.md) | Bench utterance |
| [rapport.md](../rapport.md) §5 | Tableaux comparatifs tous pipelines |
| [runs/experiments_tracking.csv](../runs/experiments_tracking.csv) | Suivi chiffré |
