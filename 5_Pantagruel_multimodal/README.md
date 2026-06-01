# 5_Pantagruel_multimodal — variante expérimentale

Encodeur Hugging Face **`PantagrueLLM/Speech_Text_Base_fr_1K_4GB`** (API `speech_text`, `mode=AUDIO`)
+ **décodeur Transformer** + SPM (même stack que `1_Transformer`, données `sentence_like`).

## Statut

| Composant | Statut |
|-----------|--------|
| Routeur `pipeline.py` (`spm`, `train`, `evaluate`, `infer`, `run`) | implémenté |
| Délégation `1_Transformer` 3–6 | implémenté |
| Config `configs/fr-en/base.yaml` | implémenté |

> Ce n'est **pas** le forward « full multimodal » texte+audio du pretrain Pantagruel : on fine-tune une tête ST classique sur l'encodeur Speech_Text gelé puis dégelé progressivement.

## Prérequis

- Étapes communes 0–2 avec `--segment-mode sentence_like`
- `datasets/manifests_sentence/fr-en/*.tsv` et `datasets/processed_sentence/fr-en/`
- `train.target.txt` : créé par `2_prepare`, ou **généré automatiquement** depuis `train.tsv` à l'étape `spm` si absent

## Usage (machine GPU)

```bash
cd ~/S3T && source .venv/bin/activate
CFG=5_Pantagruel_multimodal/configs/fr-en/base.yaml
RUN=run_001_pantagruel_multimodal

# 1) SPM (une fois ; --overwrite si régénération)
python 5_Pantagruel_multimodal/pipeline.py spm --config "$CFG"

# 2) Entraînement (~5000 updates par défaut dans base.yaml)
nohup python 5_Pantagruel_multimodal/pipeline.py train \
  --config "$CFG" --run-id "$RUN" -v \
  > logs/${RUN}_train.log 2>&1 &

# 3) Évaluation SacreBLEU
python 5_Pantagruel_multimodal/pipeline.py evaluate \
  --config "$CFG" --run-id "$RUN" -v

# Ou enchaîner spm → evaluate (sans infer)
python 5_Pantagruel_multimodal/pipeline.py run \
  --config "$CFG" --run-id "$RUN" \
  --from-stage spm --to-stage evaluate -v
```

## Rsync depuis le ThinkPad

```bash
rsync -avz \
  --exclude '.venv' --exclude 'runs' --exclude 'datasets/raw' \
  /home/morgane/git/GETALP/S3T/5_Pantagruel_multimodal/ \
  /home/morgane/git/GETALP/S3T/scripts_communs/st_common.py \
  /home/morgane/git/GETALP/S3T/1_Transformer/ \
  mpellissier@10.8.0.2:/home/mpellissier/S3T/
```

(Adapter : rsync `5_Pantagruel_multimodal/` vers `.../S3T/5_Pantagruel_multimodal/`, idem pour `1_Transformer/` et `scripts_communs/st_common.py`.)
