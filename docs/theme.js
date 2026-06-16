/**
 * Bascule thème clair / sombre — préférence persistée dans localStorage.
 * Thème par défaut : clair.
 */
(function () {
  const STORAGE_KEY = 's3t-theme';
  const root = document.documentElement;

  function isDark() {
    return root.dataset.theme === 'dark';
  }

  function syncToggle() {
    const input = document.getElementById('theme-toggle');
    if (input) {
      input.checked = isDark();
    }
  }

  function refreshCharts() {
    if (typeof Chart === 'undefined') {
      return;
    }
    const instances = Chart.instances;
    if (!instances) {
      return;
    }
    const dark = isDark();
    const grid = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.05)';
    const tick = dark ? '#9ca3b8' : '#64748b';
    Object.values(instances).forEach(function (chart) {
      const scales = chart.options.scales || {};
      ['x', 'y'].forEach(function (axis) {
        const scale = scales[axis];
        if (!scale) {
          return;
        }
        if (scale.grid) {
          scale.grid.color = grid;
        }
        if (scale.ticks) {
          scale.ticks.color = tick;
        }
      });
      chart.update();
    });
  }

  function setTheme(dark) {
    if (dark) {
      root.dataset.theme = 'dark';
      try {
        localStorage.setItem(STORAGE_KEY, 'dark');
      } catch (e) {
        /* stockage indisponible */
      }
    } else {
      root.removeAttribute('data-theme');
      try {
        localStorage.setItem(STORAGE_KEY, 'light');
      } catch (e) {
        /* stockage indisponible */
      }
    }
    syncToggle();
    refreshCharts();
    document.dispatchEvent(new CustomEvent('s3t-theme-change', { detail: { dark: dark } }));
  }

  document.addEventListener('DOMContentLoaded', function () {
    syncToggle();
    const input = document.getElementById('theme-toggle');
    if (!input) {
      return;
    }
    input.addEventListener('change', function () {
      setTheme(input.checked);
    });
  });
})();
