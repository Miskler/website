(function () {
  const timeline = document.getElementById('timeline');
  if (!timeline) return;

  let activeGroup = null;

  function clearHover() {
    timeline.classList.remove('element-hovered');
    if (activeGroup) {
      activeGroup.forEach(el => el.classList.remove('i-hovered'));
      activeGroup = null;
    }
  }

  function applyHover(group) {
    clearHover();
    timeline.classList.add('element-hovered');
    activeGroup = group;
    group.forEach(el => el.classList.add('i-hovered'));
  }

  // универсальный обработчик
  function onEnter(e) {
    const el = e.currentTarget;
    const groupId = el.dataset.hoverGroup;
    if (!groupId) return;

    const group = document.querySelectorAll(
      `[data-hover-group="${groupId}"]`
    );
    applyHover(group);
  }

  function onLeave() {
    clearHover();
  }

  // публичная функция, которую вызывает timeline.js
  window.registerTimelineHover = function (el, groupId) {
    el.dataset.hoverGroup = groupId;
    el.addEventListener('mouseenter', onEnter);
    el.addEventListener('mouseleave', onLeave);
  };
})();