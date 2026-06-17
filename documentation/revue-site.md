# Revue du site web Pantagruel S3T

Analyse de cohérence globale du dossier `docs/` — forme et fond — pour un public académique extérieur au projet.  
*Document de travail — les points corrigés sont barrés ; la remédiation est indiquée en dessous.*

---

## Synthèse

Le site est globalement bien conçu : identité visuelle homogène (Inter, palette par variante, cartes, nav sticky), contenu pédagogique sur plusieurs pages (`pipeline.html`, `corpus.html`, `resultats.html`), et interactions soignées (carrousel d’exemples, graphiques, accessibilité partielle). Les points restants concernent surtout la **lisibilité pour un non-initié** (toggle utterance / sentence_like).

---

## Problèmes de fond (contenu, cohérence)

### 1. Notes internes visibles sur la page d’accueil - OK

Dans `index.html`, la note BLEU en mode *utterance* mentionnait des runs « en cours » ou « en file » (`run_041`, `run_033`, `run_038`). Ce type de remarque de carnet de bord n’a pas sa place sur une page destinée à un public extérieur. Sur `resultats.html`, la même note avait déjà été nettoyée — incohérence entre les deux pages.

~~**Suggestion :** aligner la note d’`index.html` sur celle de `resultats.html` (supprimer les mentions de runs en cours).~~

➜ Note *utterance* d’`index.html` alignée sur `resultats.html` : suppression des mentions `run_041`, `run_033`, `run_038` et des marqueurs « en cours » / « en file ».

---

### 2. Navigation incohérente entre les pages - OK

| Page | Liens dans la nav (avant) |
|------|---------------------------|
| `index.html` | Pipeline · Résultats · Exemples · Corpus · Vocabulaire |
| `resultats.html` | Pipeline · Résultats · Vocabulaire |
| Pages variantes | Pipeline · Résultats · Vocabulaire |

`corpus.html` n’apparaissait que depuis l’accueil.

~~**Suggestion :** harmoniser la nav sur toutes les pages (au minimum : Pipeline · Résultats · Corpus · Vocabulaire ; Exemples en ancre depuis l’accueil ou lien dédié).~~

➜ Menu uniforme sur les **10 pages HTML** : Pipeline · Résultats · Exemples · Corpus · Vocabulaire · GitHub. Sur l’accueil, « Exemples » pointe vers `#exemples` ; sur les autres pages, vers `index.html#exemples`.

---

### 3. Page `corpus.html` peu intégrée - OK

Accessible surtout via la nav de l’accueil, sans fil d’Ariane alors que les autres pages secondaires en ont un.

~~**Suggestion :** ajouter un fil d’Ariane (`Accueil › Corpus`) et un lien Corpus dans toutes les barres de navigation.~~

➜ Fil d’Ariane `Accueil › Corpus` ajouté sur `corpus.html`. Lien Corpus présent dans toutes les barres de navigation (voir point 2).

---

### 4. Fil d’Ariane absent de certaines pages - OK

Présent sur `pipeline.html`, `resultats.html`, `vocabulaire.html`, pages variantes — absent de `index.html` (normal) et de `corpus.html`.

~~**Suggestion :** compléter `corpus.html`.~~

➜ Fil d’Ariane ajouté sur `corpus.html`, avec les styles `.breadcrumb` alignés sur `pipeline.html`.

---

### 5. Incohérences de données entre pages - OK

- Score Gemini : **41** (entier) sur `index.html` vs **41,1** sur `resultats.html`.
- Libellés : « Gemini 3.5 » vs « Gemini 3.5 Flash » selon les endroits.

~~**Suggestion :** unifier les scores (une décimale) et les noms affichés.~~

➜ Données `SEGMENTS` alignées sur `index.html` et `resultats.html` : score utterance `41.1` (affiché 41,1), libellé graphique `3 · Gemini 3.5 Flash`, libellé tableau `Gemini 3.5 Flash v2` (identique sur les deux pages).

---

### 6. `variante-5.html` : contexte introductif insuffisant -> TODO

La variante est marquée « expérimentale » mais le lecteur n’a pas dès le début une phrase claire sur l’état du travail (un seul run, 7,9 BLEU, réglages non optimisés).

**Suggestion :** ajouter un encadré d’introduction (statut, limites, comparaison honnête avec V1).

---

### 7. Liens du hero vers des fichiers `.md` sur GitHub - OK

Les boutons « Documentation complète », « Glossaire », « Exemples traduits » pointaient vers `variantes.md`, `vocabulaire.md`, `phrases.md` sur GitHub — rendu brut — alors que le site propose déjà des pages HTML soignées.

~~**Suggestion :** privilégier les pages HTML du site ; réserver les liens GitHub à une section « sources / dépôt » ou au lien GitHub principal.~~

➜ Hero d’`index.html` : `pipeline.html`, `vocabulaire.html`, `#exemples` (voir aussi point 16). Lien GitHub principal conservé dans la nav ; sources `.md` accessibles via le dossier `docs/` sur GitHub en note de bas de page des exemples.

---

## Problèmes de forme (UX, lisibilité)

### 8. Toggle `utterance` / `sentence_like` opaque — OK

Les libellés sont en jargon technique, sans explication pour un public non initié.

~~**Suggestion :** sous-titres du type *segment court (natif m-TEDx)* / *segments fusionnés* ; ou lien vers une entrée du glossaire (`vocabulaire.html#segmentation`).~~

➜ Lien discret « Découpage audio → glossaire » sous le toggle sur `index.html` et `resultats.html` (`vocabulaire.html#segmentation`).

---

### 9. Police monospace sur le toggle - OK

Le segmented control utilisait `JetBrains Mono`, ce qui renforçait l’impression « interface technique ».

~~**Suggestion :** police sans-serif comme le reste du site.~~

➜ Toggle `utterance` / `sentence_like` : `font-family: var(--sans)` sur `index.html` et `resultats.html`.

---

### 10. Colonnes « Paramètres clés » trop denses - OK

Les cellules du tableau sous le graphique BLEU accumulent plusieurs runs et abréviations en une seule ligne.

~~**Suggestion :** une ligne par run dans le tableau détaillé ; sur l’accueil, ne garder qu’un résumé en une phrase par variante.~~

➜ **`index.html`** : colonne « Paramètres clés » réduite à une phrase par variante (run principal + réglage essentiel). **`resultats.html`** : tableau scindé en `chartRows` (meilleur score par variante) et `tableRows` (une ligne par run) ; colonne **Run** ajoutée.

---

### 11. Schéma ASCII sur `index.html`

Le bloc `<pre class="pipeline">` était moins lisible sur mobile que le SVG de `pipeline.html`.

~~**Suggestion :** réutiliser un schéma SVG simplifié ou un lien « Voir le pipeline détaillé » vers `pipeline.html`.~~

➜ Schéma SVG vertical responsive (5 variantes + étapes communes) à la place du bloc ASCII ; lien « Voir le pipeline détaillé → » vers `pipeline.html`.

---

### 12. Section « À propos » inégale - OK

`index.html` avait une bio complète ; `resultats.html` se contentait d’un footer minimal.

~~**Suggestion :** réutiliser le bloc `page-end` / `about-card` de l’accueil (ou une version courte) sur `resultats.html` et éventuellement `pipeline.html`.~~

➜ Bloc `about-card` (bio + liens LinkedIn / site personnel) ajouté sur `resultats.html` et `pipeline.html`, identique à `index.html` / `corpus.html`.

---

### 13. Emojis dans les titres de `resultats.html` - OK

🏆, 📊, 📋 n’apparaissaient que sur cette page ; le reste du site utilise numéros ou texte.

~~**Suggestion :** remplacer par les icônes numérotées déjà utilisées ailleurs (ex. variante 1, 2…) ou supprimer les emojis.~~

➜ Emojis retirés des titres « Meilleur score par variante », « Résultats BLEU (test) » et « Tableau récapitulatif ». Les sections par variante conservent leurs pastilles numérotées (1–4).

---

### 14. Graphique scatter Transformer (variante 1) -> TODO

La légende HTML est claire, mais l’ordre de lecture chronologique n’est pas évident (axe X = numéro de run).

**Suggestion :** une phrase de légende du type « de gauche à droite : ordre approximatif des expériences » ; éventuellement relier les points d’un même encodeur par une ligne légère.

---

## Observations mineures

### 15. Tooltips hover-only sur `variante-1.html` et `variante-2.html` - OK

Sur mobile, les tooltips au survol ne fonctionnent pas. `pipeline.html` et `corpus.html` utilisent un clic pour maintenir la bulle ouverte.

~~**Suggestion :** aligner le comportement des tooltips sur toutes les pages.~~

➜ Clic pour maintenir la bulle ouverte (classe `is-open`), fermeture au clic extérieur — même logique JS que `pipeline.html` / `corpus.html`. Consigne utilisateur mise à jour sur les deux pages.

---

### 16. Fichiers `.md` exposés comme documentation principale - OK

`variantes.md`, `vocabulaire.md`, `phrases.md` sont utiles en interne ; pour le public, les pages HTML devraient être la porte d’entrée.

~~**Suggestion :** ne pas mettre en avant les `.md` depuis le hero ; les garder accessibles via GitHub pour les relecteurs techniques.~~

➜ Hero d’`index.html` : liens internes vers `pipeline.html`, `vocabulaire.html` et `#exemples`. Section exemples : lien vers `resultats.html` et `variante-5.html` ; accès aux `.md` relégué en note de bas de page vers le dossier `docs/` sur GitHub.

---

### 17. Footer de `resultats.html` incomplet - OK

Liens vers accueil et vocabulaire, mais pas vers pipeline ou corpus.

~~**Suggestion :** footer harmonisé sur toutes les pages (Pipeline · Résultats · Corpus · Vocabulaire · Accueil).~~

➜ Footer identique sur les **10 pages HTML** : `Stage LIG-GETALP · traduction de la parole FR→EN · Pipeline · Résultats · Corpus · Vocabulaire · Accueil`.

---

### 18. Fichiers audio manquants (si déploiement local) - OK

`index.html` référence `audio/9fxo9YJhnG8_*.wav` ; le dossier `docs/audio/` doit être alimenté avant publication.

~~**Suggestion :** vérifier que le dossier `docs/audio/` est bien versionné ou documenté pour le déploiement.~~

➜ Script [`scripts/extract_web_audio.py`](../scripts/extract_web_audio.py) : copie les 5 segments depuis `datasets/processed/fr-en/{train,valid,test}/` vers `docs/audio/`. Procédure documentée dans le README (en-tête page web).

---

## Tableau des priorités

| Priorité | Point | Impact | Statut |
|----------|-------|--------|--------|
| **Haute** | Expliquer utterance / sentence_like aux non-initiés | Accessibilité | À faire |

---

## Architecture actuelle (rappel)

```
index.html ──┬── pipeline.html ── variante-1..5.html
             ├── resultats.html
             ├── corpus.html
             ├── vocabulaire.html
             └── #exemples (carrousel) ↔ phrases.md

variantes.md · vocabulaire.md · phrases.md (sources texte, plus complètes que certaines pages HTML)
```

**Points forts à conserver :** palette par variante, carrousel d’exemples avec audio, explications sur Gemini/Cascade dans `resultats.html`, glossaire HTML, tooltips sur `pipeline.html` et `corpus.html`, fil d’Ariane sur les pages profondes, navigation et footer harmonisés.

---

*Généré pour accompagner la relecture du site stage LIG-GETALP — traduction parole FR→EN.*
