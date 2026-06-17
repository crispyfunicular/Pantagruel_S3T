# Protocole d’évaluation S3T (figé)

| Champ | Valeur |
|-------|--------|
| **Version** | `2026-06-02-v1` |
| **Module code** | [`scripts_communs/eval_protocol.py`](../scripts_communs/eval_protocol.py) |
| **Artefact par run** | `runs/<langpair>/<run_id>/eval/protocol.json` |
| **Statut** | **Figé** — toute modification impose une nouvelle version et une re-évaluation des runs comparables |

Ce document décrit **comment** les scores SacreBLEU du projet sont produits. Il complète [protocole_utterance_pantagruel.md](protocole_utterance_pantagruel.md) (segmentation papier) et le [PRD](PRD.md) (exigences globales).

---

## 1. Objectif

- Permettre une **comparaison équitable** entre variantes (ST, speechLLM, Gemini, cascade) sur **les mêmes** splits et la **même** métrique.
- Éviter de mélanger des changements de protocole (beam, normalisation texte, sous-ensemble dev) avec des gains de modèle.
- Tracer chaque run via `protocol.json` + signature SacreBLEU dans `sacrebleu_*.txt`.

---

## 2. Jeux de données et segmentation

### 2.1 Splits

| Split | Fichier manifest | Rôle |
|-------|------------------|------|
| **train** | `train.tsv` | Entraînement uniquement (jamais scoré à l’éval finale) |
| **valid** | `valid.tsv` | Sélection checkpoint + **BLEU dev** |
| **test** | `test.tsv` | **BLEU test** (métrique de rapport principal) |

Les splits suivent m-TEDx après `2_prepare` (OpenSLR-100). Pas de re-mélange aléatoire côté S3T.

### 2.2 Segmentation (axe indépendant)

| `segment_mode` | Manifests / WAV | Usage |
|----------------|-----------------|--------|
| **`utterance`** | `datasets/manifests/<pair>/`, `datasets/processed/<pair>/` | Comparaison Table 8 Pantagruel |
| **`sentence_like`** | `datasets/manifests_sentence/<pair>/`, `datasets/processed_sentence/<pair>/` | Runs expérimentaux S3T (fusion de segments) |

**Règle figée :** un checkpoint entraîné sur `sentence_like` ne doit **pas** être évalué sur des manifests `utterance` (et inversement).

### 2.3 Anti-fuite

`2_prepare` exécute `detect_leaks` (ids / texte cible normalisé) entre train et valid/test. Par défaut `--fail-on-leak` est actif.

### 2.4 Alignement prédictions / références

- **Une ligne = un segment** du manifest, **même ordre** que le TSV.
- Fichiers : `eval/dev_predictions.txt`, `eval/test_predictions.txt` (split **valid** nommé `dev` dans les artefacts).
- Références : texte cible anglais du manifest (`tgt_text`), **sans** normalisation NFKC/minuscules avant SacreBLEU (voir §3).

---

## 3. Métriques

### 3.1 Métrique principale

**SacreBLEU corpus BLEU** (`sacrebleu>=2.3`), paramètres par défaut de la bibliothèque. La **signature** est enregistrée dans `sacrebleu_dev.txt` / `sacrebleu_test.txt` et dans `protocol.json`.

### 3.2 Métriques secondaires (même passe)

- **chrF** corpus  
- **TER** corpus  

Elles sont dans `metrics.json` mais **ne remplacent pas** le BLEU pour les tableaux README / rapport.

### 3.3 Normalisation texte

| Variante | Hypothèse | Référence |
|----------|-----------|-----------|
| ST Transformer | Décodage SPM → chaîne | Décodage SPM des tokens cible |
| speechLLM | Texte généré LLM | `tgt_text` manifest brut |
| Gemini | Réponse API | `tgt_text` manifest brut |
| Cascade | Sortie Marian | `tgt_text` manifest brut |

**Aucune** étape d’éval ST n’applique `utils_text_eval.normalize_text_for_eval` (réservé aux scripts ASR locaux).

### 3.4 Dépendance

Verrouillage recommandé : `sacrebleu>=2.3,<3` dans `requirements.txt`. La version exacte est loggée dans `protocol.json` (`sacrebleu_package_version`).

---

## 4. Décodage par variante (v1 figée)

Les paramètres **réellement exécutés** doivent figurer dans `eval/protocol.json` → champ `decode`. Le champ `pipeline_spec.decode_target` indique l’objectif article quand le code n’est pas encore aligné.

### 4.1 ST end-to-end (`1_Transformer`, variante 5 déléguée)

| Paramètre | Valeur v1 (implémenté) | Config YAML type |
|-----------|------------------------|------------------|
| Mode | **Greedy** (pas de beam search dans le code) | `decode.beam_size: 5` (journalisé seulement) |
| Longueur max | `decode.max_len_b` (défaut **128**) | `base_*.yaml` |
| Checkpoint | `checkpoints/best.pt` | meilleur BLEU **dev** en train (greedy, 20 batches max) |
| Early stopping | `train.early_stopping_patience` (0 = off) | Arrêt si N évals dev consécutives sans gain BLEU (`4_train.py`, juin 2026) |

**Écart connu :** le papier Pantagruel / Table 8 mentionne beam 5 ; l’implémentation S3T v1 reste greedy. Passer à beam 5 = **nouvelle version de protocole** + relance de tous les runs ST comparables.

### 4.2 speechLLM B1 (`2_speechLLM`)

| Paramètre | Valeur figée |
|-----------|--------------|
| `beam_size` | **1** |
| `max_new_tokens` | **48** |
| Prompt | `Translate the French speech to English.` |
| Checkpoint | `best.pt` (projecteur ; encodeur si `freeze_encoder: false`) |

Runs de référence : `run_002` (encodeur gelé), `run_005` (encodeur dégelé).

### 4.3 Gemini API (`3_Gemini`)

| Paramètre | Valeur figée |
|-----------|--------------|
| Entrée | WAV 16 kHz via manifest |
| `temperature` | **0.0** |
| `max_output_tokens` | **256** |
| Modèle | `model.gemini_id` dans le YAML du run (ex. `gemini-2.5-flash`, `gemini-3.5-flash`) |

Pas d’entraînement ; coût estimé via `pricing.*` dans le YAML.

### 4.4 Cascade ASR→MT (`4_cascade`)

| Étape | Défaut figé |
|-------|-------------|
| ASR | Whisper `openai/whisper-large-v3`, `language=fr` |
| MT | Marian `Helsinki-NLP/opus-mt-fr-en`, `mt_max_length=256` |

Segments en échec : hypothèse vide, comptés dans `metrics.json` → `failures`.

---

## 5. Artefacts obligatoires par run évalué

```
runs/<langpair>/<run_id>/eval/
  dev_predictions.txt
  test_predictions.txt
  sacrebleu_dev.txt      # BLEU + signature
  sacrebleu_test.txt
  metrics.json           # scores + decode + config
  protocol.json          # version figée du protocole (NEW)
```

Optionnel : `*_review.tsv` (speechLLM / Gemini) via `export_eval_review`.

---

## 6. Suivi agrégé

```bash
python scripts_communs/update_experiments_tracking.py --run-dir runs/fr-en/<run_id>
# ou
python scripts_communs/update_experiments_tracking.py --all
```

Colonnes CSV : `bleu_dev`, `bleu_test`, `segment_mode`, `beam`, notes avec `sacrebleu_signature` si présent.

---

## 7. Quand changer de version de protocole ?

Incrémenter `EVAL_PROTOCOL_VERSION` dans `eval_protocol.py` et **relancer l’évaluation** (pas seulement le train) si l’un des points suivants change :

1. Version majeure de `sacrebleu` ou paramètres BLEU non par défaut  
2. Normalisation texte avant score  
3. Décodage (greedy → beam, `max_new_tokens`, température Gemini, modèles cascade)  
4. Segmentation (`utterance` ↔ `sentence_like`) ou manifests valid/test  
5. Règles d’alignement (ordre des lignes, filtrage de segments)

**Ne pas** comparer directement des BLEU issus de versions de protocole différentes.

---

## 8. Bench multi-variantes (éval seule)

Script d’orchestration (évaluations sans train) :

```bash
bash scripts/bench_evaluate_variants.sh              # sentence_like (défaut)
bash scripts/bench_evaluate_variants.sh utterance    # bench papier
bash scripts/bench_evaluate_variants.sh --dry-run
```

Prérequis : checkpoints / clé API selon la variante ; manifests cohérents avec le `segment_mode`.

---

## 9. Comparaison Pantagruel Table 8

- Papier : BLEU **test**, segmentation **utterance**, encodeur + décodeur fairseq/LeBenchmark.  
- S3T v1 : bench utterance **partiel** — cascade **37,41** et Gemini **33,72** test OK ; ST run_002 **3,79** (échec) ; **run_004 v2** **16,68** test (réplication partielle Table 8) ; ST en **greedy** (pas beam 5) ; encodeur **B-1k** seulement.  
- Les scores `sentence_like` sont **indicatifs** et ne remplacent pas la ligne Table 8 utterance.

---

## 10. Historique des versions

| Version | Date | Changements |
|---------|------|-------------|
| `2026-06-02-v1` | 2026-06-02 | Première version figée ; `protocol.json` ; greedy ST documenté ; speechLLM beam=1/48 ; Gemini temp=0 ; SacreBLEU défaut |
