# Runbook Experiments ST (PRD v2)

Ce document complete `PRD.md` avec un protocole operationnel pour lancer, suivre et reproduire les experiences ST.

---

## 1) Prerequis

- Python 3.10+
- GPU CUDA disponible
- Acces reseau pour OpenSLR et Hugging Face
- Espace disque recommande: >= 200 GB

Installation environnement:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchaudio transformers speechbrain sacrebleu sentencepiece tensorboard pyyaml pandas
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

Verrouiller l'environnement:

```bash
pip freeze > requirements.lock.txt
```

---

## 2) Convention de dossiers

```text
S3T/
  PRD.md
  README_experiments.md
  datasets/
    raw/
    processed/
    manifests/
  configs/
    fr-es/
    fr-en/
    fr-pt/
  runs/
    fr-es/
      run_001/
        config.yaml
        train.log
        checkpoints/
        eval/
          dev_predictions.txt
          test_predictions.txt
          sacrebleu_dev.txt
          sacrebleu_test.txt
          metrics.json
```

Convention nommage run:

`run_<id>_<langpair>_<seed>_<freeze>_<vocab>_<beam>`

Exemple:

`run_001_fr-es_seed42_freeze5k_vocab1k_beam5`

---

## 3) Standard de reproductibilite

Chaque run doit stocker:

- commit Git (`git rev-parse HEAD`)
- `requirements.lock.txt`
- config complete du run (`config.yaml`)
- seeds (`python`, `numpy`, `torch`, `cuda`)
- flags CuDNN (`deterministic`, `benchmark`)
- infos machine/GPU (`nvidia-smi`)
- commandes exactes executees (train/decode/eval)

Template seed a appliquer au debut de chaque script Python:

```python
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
```

---

## 4) Protocole data (sans fuite)

Regles obligatoires:

- respecter les splits OpenSLR (`train/valid/test`) sans re-melange
- entrainer SentencePiece uniquement sur les textes cibles de `train`
- exclure segments invalides: audio manquant, texte vide, duree hors bornes
- normalisation audio: WAV 16 kHz mono 16-bit PCM
- normalisation texte stable: Unicode NFKC, espaces normalises, strategie casse fixee

Exemple commandes (a adapter aux scripts du repo):

```bash
# 1) Telechargement corpus
mkdir -p datasets/raw
cd datasets/raw
wget -c https://www.openslr.org/resources/100/mtedx_fr-en.tgz
wget -c https://www.openslr.org/resources/100/mtedx_fr-pt.tgz
wget -c https://www.openslr.org/resources/100/mtedx_fr-es.tgz
tar -xzf mtedx_fr-en.tgz
tar -xzf mtedx_fr-pt.tgz
tar -xzf mtedx_fr-es.tgz
cd ../..

# 2) Preparation audio/manifestes
python scripts/prepare_mtedx.py \
  --input_root datasets/raw \
  --output_root datasets/processed \
  --manifests_root datasets/manifests \
  --sample_rate 16000 \
  --min_duration 1.0 \
  --max_duration 30.0 \
  --text_norm nfkc \
  --keep_case false
```

---

## 5) Tokenisation (SentencePiece)

Commande type par langue cible:

```bash
python scripts/train_spm.py \
  --train_text datasets/manifests/fr-es/train.target.txt \
  --model_prefix datasets/processed/spm/fr-es_1k \
  --vocab_size 1000 \
  --model_type unigram \
  --character_coverage 1.0
```

Variantes ablation:

- vocab `1000`
- vocab `5000`

---

## 6) Config run (template)

Exemple `configs/fr-es/base.yaml`:

```yaml
experiment:
  name: "fr-es_base"
  lang_pair: "fr-es"
  output_dir: "runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5"
  seed: 42
  deterministic: true

data:
  train_manifest: "datasets/manifests/fr-es/train.tsv"
  valid_manifest: "datasets/manifests/fr-es/valid.tsv"
  test_manifest: "datasets/manifests/fr-es/test.tsv"
  spm_model: "datasets/processed/spm/fr-es_1k.model"
  sample_rate: 16000
  min_duration_s: 1.0
  max_duration_s: 30.0

model:
  encoder_name: "PantagrueLLM/Pantagruel-Base"
  decoder_layers: 6
  decoder_heads: 8
  hidden_dim: 768
  dropout: 0.1

train:
  max_updates: 120000
  warmup_updates: 10000
  freeze_encoder_updates: 5000
  learning_rate_peak: 0.0002
  weight_decay: 0.01
  label_smoothing: 0.1
  batch_size: 8
  gradient_accumulation: 8
  gradient_clip_norm: 1.0
  amp_dtype: "bf16"
  eval_every_updates: 1000
  early_stopping_patience: 8
  best_checkpoint_metric: "bleu_dev"

decode:
  beam_size: 5
  max_len_a: 1.2
  max_len_b: 10
```

---

## 7) Train / Decode / Eval (commandes de reference)

Lancement train:

```bash
python scripts/train_st.py --config configs/fr-es/base.yaml
```

Generation dev/test:

```bash
python scripts/decode_st.py \
  --config configs/fr-es/base.yaml \
  --checkpoint runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/checkpoints/best.pt \
  --split valid \
  --output runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/eval/dev_predictions.txt

python scripts/decode_st.py \
  --config configs/fr-es/base.yaml \
  --checkpoint runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/checkpoints/best.pt \
  --split test \
  --output runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/eval/test_predictions.txt
```

Evaluation BLEU canonique:

```bash
sacrebleu \
  datasets/manifests/fr-es/valid.target.txt \
  -i runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/eval/dev_predictions.txt \
  -m bleu chrf ter \
  -w 2 \
  > runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/eval/sacrebleu_dev.txt

sacrebleu \
  datasets/manifests/fr-es/test.target.txt \
  -i runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/eval/test_predictions.txt \
  -m bleu chrf ter \
  -w 2 \
  > runs/fr-es/run_001_fr-es_seed42_freeze5k_vocab1k_beam5/eval/sacrebleu_test.txt
```

Important:

- garder exactement la meme commande SacreBLEU entre runs
- stocker la signature SacreBLEU dans les fichiers `sacrebleu_*.txt`
- choisir le meilleur checkpoint sur `BLEU dev` (loss seulement secondaire)

---

## 8) Plan baseline + mini-ablation

Ordre recommande:

1. baseline minimale: `freeze=full`, `beam=1`, `vocab=1k`
2. variante A: `freeze=5k`, `beam=1`, `vocab=1k`
3. variante B: `freeze=10k`, `beam=1`, `vocab=1k`
4. variante C: `freeze=5k`, `beam=5`, `vocab=1k`
5. variante D: `freeze=5k`, `beam=5`, `vocab=5k`

Regle de decision:

- promouvoir une variante seulement si gain `BLEU dev` stable sur au moins 2 seeds

---

## 9) Go/No-Go par phase

- Phase 1 go: environnement fige + GPU OK + seed policy active
- Phase 2 go: manifests propres (0 fuite, 0 vide, stats exportees)
- Phase 3 go: `forward/backward` stables sur dummy batch
- Phase 4 go: loss baisse et baseline depassee sur `BLEU dev`
- Phase 5 go: rapport final reproductible (config + commit + commande eval + signatures)

---

## 10) Fichier de suivi d'experiences

Creer `runs/experiments_tracking.csv`:

```csv
run_id,lang_pair,seed,freeze_updates,vocab_size,beam,bleu_dev,bleu_test,chrf_test,ter_test,train_hours,max_gpu_mem_gb,git_commit,status,notes
```

Mettre a jour apres chaque run pour comparer facilement avec la Table 8 Pantagruel.

---

## 11) Protocole FR->FR (smoke test Pantagruel / ASR)

Le checkpoint Hugging Face `PantagrueLLM/Speech_Text_Base_fr_1K_4GB` est un **encodeur de pre-entrainement** (pas de tete decodeur ASR/ST). Pour valider l'environnement et mesurer WER/CER sur un petit corpus local :

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Encodeur Pantagruel : forward audio uniquement (pas de transcription)
python scripts/quick_eval_hf_asr.py corpus_audio/ \
  --transcription pantagruel-encoder

# ASR proxy Whisper (protocole d'evaluation reproductible en attendant le finetune fairseq)
python scripts/quick_eval_hf_asr.py corpus_audio/ \
  --transcription whisper

# Fichier unique + reference manuelle
python scripts/quick_eval_hf_asr.py corpus_audio/au_nord_du_pays.wav \
  --transcription whisper \
  --reference "Au Nord du pays, vit une espèce de chat, dont la queue est très courte"
```

Sortie : rapport JSON sous `artifacts/quick_eval_*.json` (paires `.wav`/`.lab`, fichiers ignores, agregats WER/CER).

| Mode | Metrique | Role dans le pipeline S3T |
|------|----------|-----------------------------|
| `--transcription whisper` | WER / CER | Prototype du protocole d'evaluation locale FR->FR |
| `--transcription pantagruel-encoder` | forme des embeddings | Verifie le chargement HF avant finetune ST |
| `5_evaluate.py` / `6_infer.py` | SacreBLEU | Evaluation ST cible (FR->EN, etc.) |

Convention : les transcriptions sans audio associe dans `corpus_audio/` (ex. `11-*` a `20-*`) sont **ignorees** et listees dans le champ `skipped` du rapport.

---

## 12) Checklist de cloture d'un run

- [ ] config sauvegardee dans le dossier run
- [ ] commit Git enregistre
- [ ] log train complet conserve
- [ ] predictions dev/test conservees
- [ ] `sacrebleu_dev.txt` et `sacrebleu_test.txt` presentes
- [ ] signature SacreBLEU verifiee
- [ ] ligne ajoutee a `experiments_tracking.csv`

