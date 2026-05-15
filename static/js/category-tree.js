// category-tree.js — toggle expand/collapse for left-side category sidebar.
// Clicking the toggle triangle expands/collapses; clicking the link itself
// still navigates to the section page.
(function () {
  function init() {
    var toggles = document.querySelectorAll('.cat-tree-toggle[data-toggle-tree]');
    toggles.forEach(function (toggle) {
      toggle.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var item = toggle.closest('.cat-tree-item');
        if (!item) return;
        if (item.classList.contains('collapsed')) {
          item.classList.remove('collapsed');
          item.classList.add('open');
        } else {
          item.classList.remove('open');
          item.classList.add('collapsed');
        }
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
