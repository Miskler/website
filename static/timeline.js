const data = JSON.parse(
  document.getElementById('timeline-data').textContent
);
const container = document.getElementById('timeline');
const DAY = 86400000;
const PX_PER_DAY = 0.8;
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
  const offset = sign * e.lane * visualStep;
  if (e.timeline.point) {
    const point = document.createElement('div');
    point.className = 'timeline-point';
    point.id = `event-${i}-point`;
    point.style.top = `${y1}px`;
    point.style.left = `calc(50% + ${offset}px)`;
    point.style.transform = 'translate(-50%, -50%)';
    point.style.background = e.color;
    container.appendChild(point);
  } else {
    const range = document.createElement('div');
    range.className = 'timeline-range';
    range.id = `event-${i}-range`;
    range.style.top = `${y1}px`;
    range.style.height = `${Math.max(6, y2 - y1)}px`;
    range.style.left = `calc(50% + ${offset}px)`;
    range.style.transform = 'translate(-50%, 0)';
    range.style.background = e.color;
    container.appendChild(range);
  }
  const label = document.createElement('div');
  label.className = `timeline-label ${e.side}`;
  label.id = `event-${i}-label`;
  const labelLane = e.side === 'left' ? Math.abs(minLane) : Math.abs(maxLane);
  label.style.setProperty('--lane', labelLane);
  label.style.setProperty('--y', `${y1}px`);
  label.textContent = e.title;
  container.appendChild(label);
});