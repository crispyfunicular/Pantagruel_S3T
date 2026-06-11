# Corpus personnel — évaluation hors m-TEDx

Corpus de **40 lectures françaises** (locuteur unique, voix enregistrée) avec **références anglaises**.
Objectif : mesurer la généralisation des modèles S3T sur un domaine et un locuteur **absents de l'entraînement m-TEDx**.

## Contenu

| Fichier | Description |
|---------|-------------|
| `N-K.wav` | Audio 16 kHz mono (`N` ∈ 1..20, `K` ∈ {1,2}) — **textes distincts** |
| `corpus_perso_ref_EN.txt` | 200 phrases EN, 40 paragraphes × 5 phrases |
| `corpus_perso_test.tsv` | Manifest généré (40 lignes) |
| `eval_profiles.yaml` | Profils modèles à activer / adapter |
| `configs/*.yaml` | Configs d'évaluation par variante |

**Mapping** : `N-K.wav` → paragraphe d'indice `(N-1)×2 + (K-1)`.

## Préparation (une fois)

```bash
source .venv/bin/activate
python scripts/build_corpus_perso_manifest.py
```

Vérification :

```bash
python scripts/run_corpus_perso_eval.py --dry-run
```

## Lancer les évaluations

1. Ouvrir [`eval_profiles.yaml`](eval_profiles.yaml).
2. Pour chaque modèle :
   - mettre `enabled: true` ;
   - vérifier le chemin `checkpoint` (ST / speechLLM) ;
   - ajuster `extra_args` si besoin (ex. `--beam-size 5`, `--limit 5` pour smoke test Gemini).
3. Lancer :

```bash
# Un seul modèle (ignore enabled)
python scripts/run_corpus_perso_eval.py --only gemini_35_v2 -v

# Tous les profils activés
python scripts/run_corpus_perso_eval.py -v

# Tous les profils (même disabled) — debug
python scripts/run_corpus_perso_eval.py --all --dry-run
```

## Sorties

- `runs/fr-en/eval_corpus_perso_<variante>/eval/metrics.json`
- `dev` et `test` pointent sur le **même manifest** → scores identiques (corpus unique de 40 clips).
- Métrique principale : **BLEU corpus** SacreBLEU (non comparable aux scores m-TEDx).

## Modèles pré-configurés

| ID profil | Variante | Référence m-TEDx | Checkpoint par défaut |
|-----------|----------|------------------|------------------------|
| `st_14k_v3` | ST L-14k v3 | run_020 @ 21,22 | `runs/.../run_020_.../best.pt` |
| `st_114k_v2` | ST L-114k v2 | run_016 @ 19,63 | `runs/.../run_016_.../best.pt` |
| `speechllm_14k` | speechLLM Phi-2 | run_012 @ 15,03 | `runs/.../run_012_.../best.pt` |
| `gemini_35_v2` | Gemini 3.5 Flash v2 | run_005 @ 41,09 | API (`GEMINI_API_KEY`) |
| `cascade_utterance` | Whisper → Marian | run_001 @ 37,41 | modèles pré-entraînés HF |

**Note** : rapatrier les checkpoints distants (Modyco/OVH) avant d'évaluer ST / speechLLM en local.

## Commandes manuelles (alternative)

```bash
python 1_Transformer/pipeline.py evaluate \
  --config corpus_perso/configs/st_large_14k_v3.yaml \
  --run-id eval_corpus_perso_st_14k_v3 \
  --checkpoint runs/fr-en/run_020_transformer_baseline_utterance_large_14k_v3/checkpoints/best.pt
```

## Limites méthodologiques

- Pas de transcript FR source → pas d'analyse ASR intermédiaire ni de comparaison source/hypothèse côté français.
- Corpus court (40 utterances), locuteur unique, style lecture — variance statistique élevée.
- Les scores **ne remplacent pas** le bench m-TEDx ; ils complètent l'analyse de généralisation.
