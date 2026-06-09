# Pipeline speechLLM (B1)

Traduction parole **fr→en** sur m-TEDx via **Pantagruel (gelé)** + **projecteur (entraîné)** + **LLM causal (gelé)**.

Référence : [embarrassingly_simple_approach.pdf](embarrassingly_simple_approach.pdf) (SLAM-ASR).

## Prérequis

- Stages données S3T `0`–`3` exécutés (`datasets/manifests/fr-en/*.tsv`).
- GPU recommandé ; pour debug pipeline, `microsoft/phi-2` dans `configs/fr-en/b1.yaml`.
- Accès Hugging Face pour Pantagruel et le LLM.
- B2bis (Qwen / Mistral) : voir [recap_decodeurs.md](recap_decodeurs.md) ; Mistral 7B requiert `bitsandbytes` (`load_in_4bit: true`).

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

## Configs B2bis (autres décodeurs LLM)

| Config | LLM | Format prompt | Quantisation |
|--------|-----|---------------|--------------|
| `configs/fr-en/b2bis_qwen25_3b.yaml` | Qwen2.5-3B-Instruct | `qwen_chatml` | non |
| `configs/fr-en/b2bis_mistral_7b.yaml` | Mistral-7B-Instruct-v0.3 | `mistral_inst` | 4-bit |

Chaque LLM nécessite un **projecteur réentraîné** (dimensions d'embedding différentes). Les checkpoints Phi-2 B1 ne sont pas réutilisables.

```bash
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b2bis_qwen25_3b.yaml \
  --run-id run_010_speechllm_b2bis_qwen25_3b
```

Champs YAML clés : `model.llm_name`, `prompt.format` (`phi2` | `qwen_chatml` | `mistral_inst`), `model.load_in_4bit`.

## VRAM

- **Phi-2** : pilote pipeline sur GPU 12–16 Go (batch réduit dans `b1.yaml`).
- **Qwen2.5-3B** : ~7 Go full-precision (config `b2bis_qwen25_3b.yaml`).
- **Mistral-7B** : ~5 Go en 4-bit (`b2bis_mistral_7b.yaml`, `pip install bitsandbytes`).

## Format d'entraînement

Le format dépend du LLM (`prompt.format` dans le YAML) :

| Format | Séquence |
|--------|----------|
| `phi2` (défaut B1) | `USER: <speech> <prompt> ASSISTANT: <traduction>` |
| `qwen_chatml` | `<\|im_start\|>user\n<speech><prompt>\n<\|im_start\|>assistant\n<traduction>` |
| `mistral_inst` | `[INST] <speech><prompt> [/INST] <traduction>` |

La loss ne s'applique qu'aux tokens de la traduction (après le marqueur assistant).
