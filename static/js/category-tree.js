// category-tree.js — toggle expand/collapse for left-side category sidebar
(function () {
  function init() {
    var toggles = document.querySelectorAll('.cat-tree-toggle[data-toggle-tree]');
    toggles.forEach(function (toggle) {
      toggle.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var item = toggle.closest('.cat-tree-item');
        if (!item) return;
        item.classList.toggle('collapsed');
        item.classList.toggle('open');
      });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
