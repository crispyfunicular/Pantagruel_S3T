# Rapport détaillé — Comment `pantagruel_uni` a permis l’article `Pantagruel_2026.pdf`

Date: 2026-05-18  
Dépôt analysé: `/home/morgane/git/GETALP/fairseq`  
Branche étudiée: `origin/pantagruel_uni`  
Référence article: `Pantagruel_2026.pdf` (dans la racine du dépôt)

## 1) Résumé direct

La branche `pantagruel_uni` contient le travail d’ingénierie qui rend possible la plupart des contributions annoncées dans l’article:

- **modèles Pantagruel data2vec/JEPA** pour texte et parole;
- **ajout d’une perte MLM côté texte** en plus de la perte représentationnelle;
- **pipeline de pré-entraînement/fine-tuning speech-text** avec configurations dédiées;
- **préparation de corpus audio/texte** et conversions nécessaires;
- **infrastructure d’entraînement à grande échelle** (gestion du temps limite, average de poids, robustesse distributed).

En chiffres (comparaison avec `origin/main`):

- `origin/main...origin/pantagruel_uni` = `0 822`  
  (donc `pantagruel_uni` ajoute **822 commits** par rapport à `main`)
- Diff HEAD: **50 fichiers modifiés**, **2742 insertions**, **170 suppressions**.

## 2) Ce que dit l’article, et où c’est implémenté

L’article annonce explicitement:

- un cadre **data2vec 2.0 / JEPA** pour texte + parole;
- un schéma teacher-student avec prédiction en espace latent;
- pour le texte, une **combinaison data2vec + MLM**;
- des pré-entraînements/fine-tunings multi-corpus et multi-tâches.

Dans `pantagruel_uni`, cela apparaît surtout ici:

- `examples/data2vec/models/data2vec2.py`
- `fairseq/tasks/masked_lm.py`
- `examples/data2vec/tasks/multimodal.py`
- `examples/data2vec/models/modalities/modules.py`
- `examples/data2vec/tasks/s2t_finetuning.py`
- `fairseq_cli/train.py`
- `fairseq/optim/weight_averaging.py`
- `examples/speech_to_text/prep_mtedx_data.py`
- `examples/speech_to_text/prep_covost_data.py`
- `examples/pantagruel/configs/speech/finetuning/*.yaml`

## 3) Chronologie technique utile pour relier code et papier

## Phase A — socle d’entraînement data2vec/joint (2024)

Commits marquants:

- `07142ea3` (2024-02-09): ajout de **time limit** pour arrêter proprement les jobs.
- `c812fd6b`, `665e9f72`, `22024d93`, `7f5d1b39` (2024-02): migration/corrections **PyTorch 2 SDPA**.
- `69fb511d` (2024-04-05): lancement du **speech-text joint pre-training**.
- `6e77e3f5`, `8ea0490d`, `e2087c7c`, `ab1f853b` (2024-04): stratégie **dummy ops/tensors** pour stabiliser le distributed training (`find_unused_parameters`).
- `ef06dc19` (2024-04-25): ajout de **token_type_embeddings** pour joint training.
- `5c530a68` + `d12d2acc` (2024-05): préparation manifest multi-corpus speech.
- `f62cfeab` (2024-05-28): correction time limit + conversion HF.

Impact article:

- rend possible l’entraînement multi-GPU long (tableaux de pré-entraînement);
- permet les runs speech-text robustes servant aux résultats expérimentaux;
- fournit la base opérationnelle de la recette Pantagruel.

## Phase B — extension des objectifs Pantagruel (2024 Q4 → 2025)

Commits marquants:

- `287ef28d`, `8616e555`, `f2b61eb8`, `024ebdb4`, `547eea66`, `0fc28b48`: ajout/expérimentation de pertes (pantagruel multi, MMM, NCP, MoCo, contrastive).
- `1201114c` (2025-10-13): ajout **MLM + ST finetuning** dans data2vec.
- `790df414`, `25d57249`, `79eaac9c`, `9a1c5a3a`: calendrier/pondération de la composante MLM (warmup, decay, tri-stage).
- `9999a086` (2025-11-19): implémentation **`original_bert` MLM** pour Pantagruel text.
- `da093c41` (2025-11-24): ajout **rotary embeddings** pour Pantagruel unimodal.
- `fe495b26` (2025-12-01): ajout de **model weight averaging** en entraînement.
- `28321400` (2026-03-18): ajout explicite de la **MLM loss dans Pantagruel multimodal**.

Impact article:

- correspond directement à l’assertion clé du papier:  
  *“pour le texte, data2vec est complété par MLM”*;
- alimente les variantes rapportées (modèles textuels avec composante MLM);
- améliore la stabilité et la qualité des checkpoints utilisés pour les tableaux de résultats.

## Phase C — stabilisation finale et reproductibilité (2026)

Commits marquants:

- `c0c97142` (2026-03-22): correction d’erreurs en fine-tuning avec modèles pré-entraînés MLM.
- nombreux commits “add/modify configs” (fin 2025–début 2026): industrialisation des recettes expérimentales.

Impact article:

- verrouille la compatibilité prétrain → finetune;
- fiabilise les reproductions multi-runs nécessaires aux scores moyens reportés.

## 4) Mapping article → code (détaillé)

## 4.1 Objectif JEPA/data2vec (texte + parole)

Implémentation principale:

- `examples/data2vec/models/data2vec2.py`
  - support multi-modalité;
  - teacher-student + prédiction de représentations;
  - logique de masquage, EMA, décodage latent.

Éléments complémentaires:

- `examples/data2vec/models/modalities/base.py`  
  (encodage local/contextuel, gestion `local_grad_mult`, hooks positionnels)
- `examples/data2vec/models/modalities/modules.py`  
  (blocs attention, rotary, SDPA torch2)

Lien papier:

- correspond à la Figure 1 (teacher/student, masked tokens, EMA).

## 4.2 Ajout MLM pour texte (contribution centrale du papier)

Implémentation:

- `examples/data2vec/models/data2vec2.py`
  - paramètres `mlm_loss`, `mlm_num_layers`, `mlm_impl`, scheduling de poids MLM;
  - deux modes d’implémentation (`use_decoder_output` / `original_bert`);
  - combinaison de la loss MLM avec la loss data2vec.
- `fairseq/tasks/masked_lm.py`
  - préparation dataset compatible Pantagruel (incluant chemin “original_bert”).

Commits clés:

- `9999a086`, `894305b1`, `79eaac9c`, `9a1c5a3a`, `28321400`.

Lien papier:

- matérialise la phrase de l’article indiquant que **la perte MLM est ajoutée pour les modèles texte**.

## 4.3 Fine-tuning speech et speech-text

Implémentation:

- `examples/data2vec/tasks/s2t_finetuning.py`
- `fairseq/models/wav2vec/wav2vec2_asr.py`
- `examples/pantagruel/configs/speech/finetuning/*.yaml`
  - recettes base/large;
  - réglages `lr`, `weight_decay`, `max_tokens`, `freeze_finetune_updates`, `w2v_path`.

Commits clés:

- `1201114c`, `34502556`, `76c9e9c1`, `ff156225`, `c0c97142`.

Lien papier:

- rend possible les évaluations speech (ASR/NER/SLU/SER/ST) reportées dans les tableaux.

## 4.4 Préparation des données/corpus

Implémentation:

- `examples/speech_to_text/prep_mtedx_data.py`
  - génération manifests;
  - contrôle fréquence d’échantillonnage;
  - option speed perturbation.
- `examples/speech_to_text/prep_covost_data.py`
  - préparation CoVoST.
- `fairseq/data/audio/raw_audio_dataset.py`
  - garde-fous `sample_rate`;
  - traitement audio/mel.

Commits clés:

- `5c530a68`, `d12d2acc`, `9f88e967`, `600f6ec9`, `6852d33b`, `bbd76ed7`.

Lien papier:

- soutient la constitution des jeux d’entraînement/évaluation et la qualité des entrées audio.

## 4.5 Infrastructure d’entraînement long

Implémentation:

- `fairseq_cli/train.py`
  - arrêt anticipé piloté par **time_limit estimé**;
  - logique reprise propre des jobs.
- `fairseq/dataclass/configs.py`
  - nouveaux paramètres de contrôle (`time_limit`, etc.).
- `fairseq/optim/weight_averaging.py`
  - moyenne uniforme/EMA des poids en cours d’entraînement.

Commits clés:

- `07142ea3`, `2152b6c3`, `d55f571c`, `f62cfeab`, `fe495b26`.

Lien papier:

- indispensable pour mener les pré-entraînements longs mentionnés (grandes tables d’hyperparamètres).

## 5) Ce qui a le plus “produit” l’article

Par importance pratique:

1. **Couplage data2vec + MLM** (fichiers `data2vec2.py`, `masked_lm.py`)  
   → cœur méthodologique revendiqué pour le texte.
2. **Recettes de fine-tuning speech/text** (`examples/pantagruel/configs/...`)  
   → rend les benchmarks effectivement exécutables.
3. **Pipeline de préparation de données audio/manifest**  
   → permet de passer à l’échelle multi-corpus.
4. **Robustesse de training distribué + time limit**  
   → condition nécessaire pour finir les runs sur infra partagée.
5. **Weight averaging et ajustements de stabilité**  
   → meilleure qualité/consistance des checkpoints de comparaison.

## 6) Lecture “historique Git” importante

Le journal `pantagruel_uni` comporte beaucoup de commits “`add/modify config`”.
Interprétation:

- la branche a servi de **laboratoire expérimental** (itérations nombreuses);
- la valeur scientifique provient moins d’un seul commit “feature” que du **cycle cumulatif**:
  implémentation → réglages → corrections finetune → consolidation.

Autrement dit, l’article est appuyé par:

- une petite poignée de commits “noyau méthode”;
- une grande masse de commits “noyau expérimental” (configs, runs, stabilisation).

## 7) Conclusion opérationnelle

Oui, le code de `pantagruel_uni` a bien servi à écrire `Pantagruel_2026.pdf`:

- il contient les briques algorithmiques revendiquées (JEPA/data2vec, ajout MLM);
- il contient les outils d’exécution à grande échelle (time-limit, averaging, robustesse distributed);
- il contient la chaîne de préparation/fine-tuning qui permet de produire les résultats des tableaux.

Si vos résultats divergent, les causes probables prioritaires sont:

1. variante exacte de la perte (MLM impl/schedule)  
2. recette de config (lr, warmup, masking, EMA)  
3. préparation audio/manifest et contrôle sample rate  
4. checkpoint choisi (avec/sans weight averaging)  
5. contexte d’entraînement (arrêt sur time-limit, reprise, seed)

---

## Annexe — Points de preuve utilisés

- Divergence branche: `origin/main...origin/pantagruel_uni = 0 822`
- Diff HEAD: `50 files changed, 2742 insertions(+), 170 deletions(-)`
- Commits caractéristiques inspectés:
  - `07142ea3`, `69fb511d`, `ef06dc19`, `9999a086`, `da093c41`, `fe495b26`, `28321400`, `c0c97142`
- Fichiers pivots inspectés:
  - `examples/data2vec/models/data2vec2.py`
  - `fairseq/tasks/masked_lm.py`
  - `fairseq_cli/train.py`
  - `fairseq/optim/weight_averaging.py`
  - `examples/speech_to_text/prep_mtedx_data.py`
  - `examples/speech_to_text/prep_covost_data.py`
  - `examples/pantagruel/configs/speech/finetuning/large_mtedx_fr2en_lr3e-5.yaml`
