const data = JSON.parse(
  document.getElementById('timeline-data').textContent
);
const container = document.getElementById('timeline-body');
const Lcontainer = document.getElementById('timeline-items');
const DAY = 86400000;
const PX_PER_DAY = parseFloat(document.getElementById('timeline').getAttribute('experience-per-day'));
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
let minX = Infinity;
let maxX = -Infinity;
let maxY = -Infinity

data.forEach((e, i) => {
  const y1 = y(start(e));
  const y2 = y(end(e));
  const visualStep = 16; // Small step for visual staggering
  const offset = e.lane * visualStep;

  minX = Math.min(minX, offset);
  maxX = Math.max(maxX, offset);
  maxY = Math.max(maxY, y2)

  const element = document.createElement('div')
  if (e.timeline.point) {
    element.className = 'timeline-point';
    element.id = `event-${i}-point`;
    element.style.transform = 'translate(-50%, -50%)';
  } else {
    element.className = 'timeline-range' + (e.timeline.now ? ' now' : '');
    element.id = `event-${i}-range`;
    element.style.height = `${Math.max(6, y2 - y1)}px`;
    element.style.transform = 'translate(-50%, 0)';
  }
  element.style.left = `calc(50% + ${offset}px)`;
  element.style.top = `${y1}px`;
  element.style.setProperty('--clr', e.color);
  Lcontainer.appendChild(element);
  registerTimelineHover(element, i, true, false);


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

  label.style.setProperty('--y', `${centerY+(e.labelShift ? e.labelShift : 0)}px`);
  const labelTitle = document.createElement('span');
  labelTitle.textContent = e.title;
  label.appendChild(labelTitle);

  if (e.description) {
    const labelDesc = document.createElement('div');
    labelDesc.innerHTML = e.description;
    labelDesc.className = 'description';
    label.appendChild(labelDesc);
  };

  container.prepend(label);
  registerTimelineHover(label, i, true, true);
});


const sFooter = document.getElementById('space-footer');
sFooter.style.top = maxY+"px"

// Compute bounding box from all descendant elements

// Create a new child element with the computed boundaries
const boundsElement = document.createElement('div');
boundsElement.style.position = 'absolute';
boundsElement.style.transform = "translateX(-50%)"

const marginOutPX = 15;
boundsElement.style.top = `${-marginOutPX}px`;
boundsElement.style.width = `${(Math.abs(minX)+maxX)+marginOutPX*2}px`;
boundsElement.style.height = `${maxY+marginOutPX*2}px`;
boundsElement.style.left = `calc(50% + `+(Math.abs(minX) != Math.abs(maxX) ? (maxX+minX)/2 : 0)+'px)';

boundsElement.className = `bounds-timeline`;
Lcontainer.appendChild(boundsElement);
registerTimelineHover(boundsElement, -1, false, true);