const data = JSON.parse(
  document.getElementById('timeline-data').textContent
);
const container = document.getElementById('timeline');
const Lcontainer = document.getElementById('timeline-items');
const DAY = 86400000;
const PX_PER_DAY = 0.3;
function parseDate(s) {
  const [d, m, y] = s.split('.').map(Number);
  return new Date(y, m - 1, d);
}
function start(e) {
  return e.timeline.point
    ? parseDate(e.timeline.point)
    : parseDate(e.timeline.from);
}
function end(e) {
  return e.timeline.point
    ? start(e)
    : parseDate(e.timeline.to);
}
const dates = data.flatMap(e => [start(e), end(e)]);
const min = new Date(Math.min(...dates));
const y = d => (d - min) / DAY * PX_PER_DAY;
// Sort data by start date
data.sort((a, b) => start(a) - start(b));
// Compute min and max lane values from data
const minLane = Math.min(...data.map(e => e.lane));
const maxLane = Math.max(...data.map(e => e.lane));
// Use predefined sides and lanes from data
// Create elements
data.forEach((e, i) => {
  const y1 = y(start(e));
  const y2 = y(end(e));
  const sign = e.side === 'right' ? 1 : -1;
  const visualStep = 12; // Small step for visual staggering
  const offset = e.lane * visualStep;

  const element = document.createElement('div')
  if (e.timeline.point) {
    element.className = 'timeline-point';
    element.id = `event-${i}-point`;
    element.style.transform = 'translate(-50%, -50%)';
  } else {
    element.className = 'timeline-range';
    element.id = `event-${i}-range`;
    element.style.height = `${Math.max(6, y2 - y1)}px`;
    element.style.transform = 'translate(-50%, 0)';
  }
  element.style.left = `calc(50% + ${offset}px)`;
  element.style.top = `${y1}px`;
  element.style.setProperty('--clr', e.color);
  Lcontainer.appendChild(element);
  registerTimelineHover(element, i);


  const label = document.createElement('div');
  label.className = `timeline-label ${e.side}`;
  label.id = `event-${i}-label`;

  const labelLane = e.side === 'left'
    ? Math.abs(minLane)
    : Math.abs(maxLane);

  label.style.setProperty('--lane', labelLane);
  label.style.setProperty('--clr', e.color);

  // ВЕРТИКАЛЬНЫЙ ЦЕНТР
  const centerY = e.timeline.point
    ? y1
    : (y1 + y2) / 2;

  label.style.setProperty('--y', `${centerY}px`);
  label.textContent = e.title;

  container.appendChild(label);
  registerTimelineHover(label, i);
});