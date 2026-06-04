# Pipeline speechLLM (B1)

Traduction parole **fr→en** sur m-TEDx via **Pantagruel (gelé)** + **projecteur (entraîné)** + **LLM causal (gelé)**.

Référence : [embarrassingly_simple_approach.pdf](embarrassingly_simple_approach.pdf) (SLAM-ASR).

## Prérequis

- Stages données S3T `0`–`3` exécutés (`datasets/manifests/fr-en/*.tsv`).
- GPU recommandé ; pour debug pipeline, `microsoft/phi-2` dans `configs/fr-en/b1.yaml`.
- Accès Hugging Face pour Pantagruel et le LLM.

## Commandes

Depuis la racine du dépôt (venv activé) :

```bash
# Entraînement projecteur seul
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1.yaml \
  --run-id run_001_speechllm_b1

# Évaluation SacreBLEU (valid + test)
python 2_speechLLM/pipeline.py evaluate \
  --config 2_speechLLM/configs/fr-en/b1.yaml \
  --run-id run_001_speechllm_b1 \
  --beam-size 4

# Inférence WAV arbitraire
python 2_speechLLM/pipeline.py infer \
  --checkpoint runs/fr-en/run_001_speechllm_b1/checkpoints/best.pt \
  --input-audio path/to/audio.wav \
  --config 2_speechLLM/configs/fr-en/b1.yaml

# Train + evaluate
python 2_speechLLM/pipeline.py run \
  --config 2_speechLLM/configs/fr-en/b1.yaml \
  --run-id run_001_speechllm_b1
```

Dry-run :

```bash
python 2_speechLLM/pipeline.py train --config 2_speechLLM/configs/fr-en/b1.yaml --run-id test --dry-run
```

## Artifacts

Sous `runs/fr-en/<run_id>/` (ou `experiment.output_dir` dans le YAML) :

| Fichier | Rôle |
|---------|------|
| `config.yaml` | Copie de la config |
| `checkpoints/best.pt` | Poids entraînés (projecteur ; + encodeur si `freeze_encoder: false`) |
| `checkpoints/last.pt` | Dernier état (même format) |
| `train.log` | JSONL par update |
| `metrics.json` | Résumé entraînement |
| `eval/sacrebleu_*.txt` | Métriques signées |
| `eval/dev_predictions.txt` | Hypothèses dev |

## VRAM

- **Phi-2** : pilote pipeline sur GPU 12–16 Go (batch réduit dans `b1.yaml`).
- **7B chat** : prévoir quantisation (`bitsandbytes`) ou multi-GPU — à documenter dans la config du run.

## Format d'entraînement

```text
USER: <embeddings parole> <prompt> ASSISTANT: <traduction anglais>
```

La loss ne s'applique qu'aux tokens après `ASSISTANT:`.
