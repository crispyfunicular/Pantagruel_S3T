/**
 * Applique le thème avant le rendu (évite le flash).
 * Défaut : clair. Si l'utilisateurice a basculé le curseur : localStorage s3t-theme.
 */
(function () {
  try {
    if (localStorage.getItem('s3t-theme') === 'dark') {
      document.documentElement.dataset.theme = 'dark';
    }
  } catch (e) {
    /* stockage indisponible */
  }
})();
