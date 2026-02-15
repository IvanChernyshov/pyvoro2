// MathJax configuration for mkdocs-material.
//
// - mkdocs-material can use "instant navigation" (client-side page swaps).
// - MathJax typesets on the initial load, but needs an explicit re-typeset after
//   each page change when instant navigation is enabled.
// - We support both Markdown pages and notebook-rendered pages.

// MathJax v3 reads configuration from `window.MathJax` *before* it loads.
// This file must be included before the MathJax runtime script.
window.MathJax = {
  tex: {
    // Support both arithmatex output (\\(...\\), \\[...\\]) and raw $...$ in
    // notebook-rendered pages.
    inlineMath: [
      ['\\(', '\\)'],
      ['$', '$'],
    ],
    displayMath: [
      ['\\[', '\\]'],
      ['$$', '$$'],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Don't typeset inside code blocks.
    // NOTE: We intentionally don't include "script" here because some Markdown
    // pipelines can emit TeX in <script type="math/tex"> nodes.
    skipHtmlTags: ['noscript', 'style', 'textarea', 'pre', 'code'],
  },
};

// Re-typeset after each page swap (instant navigation).
// `document$` is a mkdocs-material global.
document$.subscribe(() => {
  // MathJax replaces the configuration object with the runtime API once loaded.
  if (!window.MathJax || !window.MathJax.typesetPromise) return;

  // Recommended by mkdocs-material to avoid caching issues when navigating.
  if (window.MathJax.startup?.output?.clearCache) {
    window.MathJax.startup.output.clearCache();
  }
  if (window.MathJax.typesetClear) window.MathJax.typesetClear();
  if (window.MathJax.texReset) window.MathJax.texReset();

  window.MathJax.typesetPromise();
});
