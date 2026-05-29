# Plan — pipeline `speechLLM` (priorité actuelle)

**Décision (2026-05) :** la piste **speechLLM** est la **ligne principale** du projet dans un premier temps. La baseline ST classique (`encodeur Pantagruel + décodeur Transformer` dans `scripts/`) reste disponible comme **référence de comparaison**, mais n’est plus le prérequis bloquant avant toute implémentation.

**Référence article :** [2_speechLLM/embarrassingly_simple_approach.pdf](../2_speechLLM/embarrassingly_simple_approach.pdf) (SLAM-ASR : encodeur gelé + projecteur seul + LLM gelé).

---

## En quoi consiste la version `speechLLM` (ligne principale)

Version **fr→en** sur m-TEDx : on garde les stages données S3T (`0`–`3`), et on implémente le cœur modèle dans **`2_speechLLM/`** :

```text
preflight → download → prepare → (spm optionnel pour scoring)
    → 2_speechLLM/train → 2_speechLLM/evaluate → 2_speechLLM/infer
```

### Schéma cible (B1, prototype minimal)

```text
Audio FR (16 kHz)
    → Encodeur parole Pantagruel (HF, GELÉ)
    → Downsampling des frames (ex. concat k=5, comme SLAM-ASR)
    → Projecteur linéaire (SEUL module entraîné au départ)
    → Embeddings injectés dans le LLM
    → LLM chat (GELÉ, ex. modèle 7B instruction)
    → Génération tokens anglais (référence = traduction m-TEDx)
```

**Idée :** aligner l’espace « parole » sur l’espace « texte » du LLM, sans réentraîner tout le stack — proche de l’article *Embarrassingly Simple*.

### Entrées / sorties

| Élément | Choix |
|---------|--------|
| Corpus | m-TEDx **fr→en** (manifests `datasets/manifests/fr-en/`) |
| Cible | texte anglais de référence (normalisation PRD) |
| Métrique principale | **SacreBLEU** dev/test (protocole figé + signature) |
| Critère de sélection | meilleur **BLEU dev** → checkpoint `best.pt` |

### Format d’entraînement (template conversation)

```text
USER: <embedding_speech> <prompt court> ASSISTANT: <traduction anglais référence>
```

- **Prompt court figé** (ex. `Translate the French speech to English.`)
- **Loss** : cross-entropy **uniquement** sur les tokens `ASSISTANT` (traduction).
- **Inférence** : `USER: … ASSISTANT:` puis génération autoregressive (beam 4–5 pour l’éval principale).

### Briques techniques (B1)

| Composant | Entraînable en B1 ? |
|-----------|---------------------|
| Pantagruel-Base (HF) | non (gelé) |
| Downsampler (k=5 par défaut) | non |
| Projecteur `Linear → ReLU → Linear` | **oui (seul)** |
| LLM chat (7B ou plus petit en debug) | non (gelé) |
| Tokenizer LLM | — |

### Runs et artifacts

- Dossier dédié : `2_speechLLM/` (code train/eval/infer).
- Runs : `runs/fr-en/<run_id>_speechllm_b1/` avec le même contrat d’artifacts que le PRD (config, logs, checkpoints, eval/, SacreBLEU signé).

---

## Objectif (priorité speechLLM)

Mettre en œuvre une pipeline ST fr→en **speechLLM** reproductible sur m-TEDx, avec Pantagruel comme encodeur parole et un LLM pour la génération anglaise, puis valider la faisabilité (convergence, BLEU, coût GPU) avant toute optimisation secondaire.

## Périmètre

- **Priorité immédiate :**
  - squelette `2_speechLLM/` (pipeline, common, train, evaluate, infer) ;
  - config B1 `fr-en` ;
  - premier run GPU reproductible ;
  - protocole SacreBLEU aligné.
- **Reporté / secondaire :**
  - campagne complète baseline ST (`1_Transformer/4_train.py`) ;
  - comparaison systématique Table 8 Pantagruel via décodeur classique ;
  - ablations B2/B3 tant que B1 n’est pas stable.

## Positionnement par rapport à la baseline ST

| | Baseline ST (`scripts/`) | **speechLLM (priorité)** |
|---|--------------------------|---------------------------|
| Statut | référence / comparaison ultérieure | **ligne de travail actuelle** |
| Tête de génération | décodeur Transformer 6 couches | **LLM pré-entraîné** |
| Paramètres entraînés | décodeur (+ encodeur après dégel) | **projecteur** (B1) |
| Tokenisation | SentencePiece | **tokenizer LLM** |
| Où coder | `scripts/st_common.py`, `4_train.py`… | `2_speechLLM/speechllm_common.py`, … |

La baseline ST n’est **pas abandonnée** : elle sert de point de comparaison quand un run speechLLM B1 est disponible.

## Principes de gouvernance (révisés)

1. **Priorité speechLLM** : l’effort d’implémentation et les runs GPU ciblent d’abord `2_speechLLM/`.
2. **Données partagées** : stages `0`–`3` S3T inchangés (source de vérité m-TEDx).
3. **Protocole d’éval constant** : SacreBLEU, signatures, manifests — pour pouvoir comparer plus tard à la baseline ST ou au papier.
4. **Simplicité d’abord** (KISS, article SLAM-ASR) : pas de Q-Former, pas de LoRA, pas de finetune encodeur en B1.
5. **Traçabilité** : configs figées, seeds, `experiments_tracking.csv`, écarts documentés.

## Plan en 2 phases (ordre révisé)

### Phase 1 — Implémentation et validation speechLLM B1 (priorité)

**But :** pipeline speechLLM bout en bout sur fr→en.

**Livrables :**

| # | Livrable |
|---|----------|
| 1 | `2_speechLLM/speechllm_common.py` (encodeur, downsampler, projecteur, LLM, collate masqué) |
| 2 | `2_speechLLM/4_train.py`, `5_evaluate.py`, `6_infer.py`, `pipeline.py` |
| 3 | `2_speechLLM/configs/fr-en/b1.yaml` |
| 4 | `2_speechLLM/README.md` (usage CLI) |
| 5 | Au moins 1 run fr→en reproductible (idéalement 2 seeds) |
| 6 | `eval/sacrebleu_*.txt` + signature + `metrics.json` |

**Critère de sortie Phase 1 :**

- entraînement converge (loss / accuracy token réponse) ;
- BLEU dev mesurable et reproductible ;
- inférence fonctionnelle sur dev/test et WAV arbitraires.

**Ordre d’implémentation recommandé :**

1. Squelette + config B1  
2. Cœur modèle + collate masqué  
3. Train (projecteur seul)  
4. Evaluate + SacreBLEU  
5. Infer  
6. Run pilote GPU (petit LLM si besoin, puis 7B)

### Phase 2 — Consolidation et comparaison (après B1)

**But :** décider si on investit dans B2/B3 ou si on revient / complète la baseline ST.

**Actions :**

- comparer speechLLM B1 à la baseline ST **si** un run ST existe (sinon : comparaison reportée, pas bloquante) ;
- positionner les scores par rapport à la Table 8 Pantagruel (avec écarts de protocole documentés) ;
- décision : **B2** (ablations) vs pause vs reprise baseline ST.

**Sous-étapes B2/B3 (si Phase 1 concluante) :**

- **B2 :** projecteur linéaire vs MLP ; prompt ; beam 4 vs 5.
- **B3 :** dégel encodeur ou LoRA LLM seulement si plafond BLEU.

### Phase reportée — Baseline ST classique

**But :** réplication Pantagruel « papier » (encodeur + décodeur 6 couches).

- **Non bloquante** pour démarrer speechLLM.
- À lancer quand utile pour comparaison directe ou livrable « réplication Table 8 » explicite.
- Scripts existants : `1_Transformer/4_train.py`, `5_evaluate.py`, `6_infer.py`.

## Protocole d’évaluation (inchangé)

- même paire `fr-en` ;
- mêmes manifests et splits ;
- même commande SacreBLEU (signature dans les artifacts) ;
- tracking : `runs/experiments_tracking.csv` avec colonne `pipeline: speechllm_b1` ;
- au moins 2 seeds avant conclusion forte sur B1.

## Hypothèses à valider (Phase 1)

1. Le projecteur seul suffit à aligner Pantagruel → LLM pour la **traduction** fr→en (tâche plus dure que l’ASR du papier).
2. Un LLM chat anglais peut générer une traduction exploitable à partir d’embeddings speech français.
3. ~50 h m-TEDx suffisent pour voir une « émergence » d’alignement (ordre de grandeur : 1 epoch / ~12k steps, cf. article).
4. Le coût GPU (7B + encodeur gelé) reste acceptable pour le labo.

## Risques et mitigations

| Risque | Mitigation |
|--------|------------|
| ST cross-lingue plus difficile que ASR LibriSpeech | prompt fixe, LLM chat anglais, early stop sur BLEU dev |
| Corpus petit (~50 h) | projecteur seul, pas de finetune encodeur en B1 |
| VRAM insuffisante (LLM 7B) | démarrer Phi-2 / TinyLLaMA-Chat pour debug pipeline |
| Pas de comparaison ST immédiate | documenter écarts ; baseline ST en Phase 2 |
| Scope creep (multimodal généraliste) | verrouiller fr→en + dossier `2_speechLLM/` isolé |

## Stack technique

- **PyTorch** + **transformers** (comme le reste de S3T) ;
- pas d’obligation SpeechBrain / fairseq ;
- dépendances LLM : prévoir `accelerate` / quantisation si VRAM limitée (à documenter dans `2_speechLLM/README.md`).

## Checklist « prêt à implémenter » (priorité speechLLM)

- [x] structure `2_speechLLM/` créée (pipeline, common, configs)
- [x] design B1 validé (template USER/ASSISTANT, loss masquée)
- [x] protocole SacreBLEU figé pour les runs speechLLM
- [ ] budget GPU estimé (pilote + run B1)
- [x] conventions de nommage `*_speechllm_b1`
- [ ] premier run pilote GPU
- [ ] (Phase 2) comparaison baseline ST si disponible

## Naming des runs

- **Priorité :** `run_XXX_fr-en_speechllm_b1_*`
- **Comparaison ultérieure :** `run_XXX_fr-en_baseline_*`

## Résumé exécutif

Le projet se concentre d’abord sur une **pipeline speechLLM** (Pantagruel + projecteur + LLM) pour la traduction parole fr→en sur m-TEDx, inspirée de SLAM-ASR. La baseline ST classique reste dans le dépôt comme référence, mais **n’est plus la première étape obligatoire**. Le succès immédiat se juge sur un B1 reproductible (train + eval + SacreBLEU), puis sur d’éventuelles ablations B2/B3 ou une comparaison avec la baseline ST / Table 8.
