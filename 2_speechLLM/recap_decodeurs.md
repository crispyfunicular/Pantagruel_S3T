# Comparatif des décodeurs LLM pour speechLLM B2bis

Évaluation des candidats LLM pour la tête de génération dans la variante speechLLM (Pantagruel gelé → projecteur → LLM gelé). Référence = `microsoft/phi-2` (B1 existant).

**Retenus pour B2bis :** Qwen2.5-3B-Instruct et Mistral-7B-Instruct-v0.3.

---

## Tableau comparatif

| Critère | **Phi-2** *(baseline B1)* | **Qwen2.5-3B-Instruct** ✅ | **Mistral-7B-Instruct-v0.3** ✅ | TinyLlama-1.1B-Chat |
|---|---|---|---|---|
| **Taille** | 2.7 B | 3 B | 7.2 B | 1.1 B |
| **Dim. embeddings** | 2 560 | 2 048 | 4 096 | 2 048 |
| **VRAM (full-prec.)** | ~6 GB | ~7 GB | ~16 GB | ~2.5 GB |
| **VRAM (4-bit)** | ~2 GB | ~2.5 GB | ~5 GB | — |
| **Qualité attendue** | Correcte (résultats B1 : BLEU dev 19.99 / test 15.89) | Bonne à très bonne | Excellente (référence ouverte) | Faible (trop petit pour ST) |
| **Multilinguisme** | Faible (anglais) | **Fort** (conçu pour) | Moyen | Moyen |
| **Chat template** | `USER: … ASSISTANT:` (format libre) | `<\|im_start\|>user\n…<\|im_end\|>` (ChatML) | `[INST]…[/INST]` (Mistral v0.3) | `<\|system\|>…<\|user\|>…<\|assistant\|>` |
| **Accès HF** | Libre (MIT) | Libre (Apache 2.0) | Libre (Apache 2.0) | Libre (Apache 2.0) |
| **Compatibilité `transformers`** | ≥ 4.36 (déjà requis) | ≥ 4.37 | ≥ 4.36 | Bonne |
| **Quantisation nécessaire** | Non | Non (3B tient en VRAM) | Recommandée (4-bit pour 16 GB) | Non |
| **Rôle dans le projet** | **Référence B1** | **B2bis — alternative légère** | **B2bis — référence qualité** | Debug pipeline uniquement |

---

## Notes d'implémentation

### Changements requis par rapport à B1

- **Projecteur à réentraîner** pour chaque LLM (les dimensions d'embeddings diffèrent — les checkpoints B1 de Phi-2 ne sont pas réutilisables directement).
- **Format de prompt configurable** : Phi-2 utilise `USER: / ASSISTANT:` en texte libre ; Qwen2.5 et Mistral ont des templates de chat spécifiques. Le YAML doit exposer `prompt.user_prefix` et `prompt.infix_template`.
- **Quantisation** : Mistral-7B nécessite `bitsandbytes` (`load_in_4bit: true` dans le YAML) pour tenir dans ~5 GB de VRAM.

### Configs YAML (implémentées)

| LLM | Fichier config | `llm_name` HuggingFace |
|-----|---------------|------------------------|
| Qwen2.5-3B-Instruct | `configs/fr-en/b2bis_qwen25_3b.yaml` | `Qwen/Qwen2.5-3B-Instruct` |
| Mistral-7B-Instruct-v0.3 | `configs/fr-en/b2bis_mistral_7b.yaml` | `mistralai/Mistral-7B-Instruct-v0.3` |

### Protocole de comparaison B2bis

Tous les runs partagent le même protocole d'éval (conforme `eval_protocol.py`) :

- `beam_size: 1`, `max_new_tokens: 48`
- Même split `fr-en` m-TEDx (valid / test)
- Même seed (42) pour reproductibilité
- Tracking dans `runs/experiments_tracking.csv` avec colonne `llm_name`

---

## Résultats B1 (référence Phi-2)

| Run | Encodeur | LLM | Segments | SacreBLEU dev | SacreBLEU test |
|-----|----------|-----|----------|---------------|----------------|
| `run_002_speechllm_b1_sentence_long` | speech-base-1K (gelé) | Phi-2 (gelé) | sentence_like | **19.99** | **15.89** |
| `run_005_speechllm_b1_sentence_long_unfreeze_encoder` | speech-base-1K (dégelé) | Phi-2 (gelé) | sentence_like | 19.25 | 18.83 |
