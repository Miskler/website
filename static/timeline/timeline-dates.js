function initTimeline(containerId) {
    const timelineElement = document.getElementById('timeline');
    const experiencePerDay = parseFloat(timelineElement.getAttribute('experience-per-day'));
    
    const dataScript = document.getElementById('timeline-data');
    const data = JSON.parse(dataScript.textContent);
    
    function parseDate(str) {
        const [day, month, year] = str.split('.').map(Number);
        return new Date(year, month - 1, day);
    }
    
    let minTime = Infinity;
    let maxTime = -Infinity;
    
    data.forEach(event => {
        if (event.timeline.point) {
            const time = parseDate(event.timeline.point).getTime();
            minTime = Math.min(minTime, time);
            maxTime = Math.max(maxTime, time);
        } else if (event.timeline.from && event.timeline.to) {
            const fromTime = parseDate(event.timeline.from).getTime();
            const toTime = parseDate(event.timeline.to).getTime();
            minTime = Math.min(minTime, fromTime);
            maxTime = Math.max(maxTime, toTime);
        }
    });
    
    if (minTime === Infinity || maxTime === -Infinity) {
        console.error('No valid dates found in the data.');
        return;
    }
    
    const minDate = new Date(minTime);
    const maxDate = new Date(maxTime);
    const totalDays = (maxTime - minTime) / (1000 * 60 * 60 * 24);
    const height = totalDays * experiencePerDay;
    
    function formatDate(date) {
        const day = date.getDate();
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        const year = date.getFullYear();
        return `${day}.${month}.${year}`;
    }
    
    const container = document.getElementById(containerId);
    const minPxMinor = parseFloat(container.getAttribute('data-min-px-minor')) || 20;
    const minPxMajor = parseFloat(container.getAttribute('data-min-px-major')) || 100;
    container.style.height = `${height}px`;
    
    // Calculate number of minor intervals
    let minorIntervals = Math.floor(height / minPxMinor);
    if (minorIntervals < 1) minorIntervals = 1;
    const minorSpacingPx = height / minorIntervals;
    
    // Calculate approximate major step
    const majorIntervals = Math.floor(height / minPxMajor);
    const majorSpacingPx = majorIntervals > 0 ? height / majorIntervals : height;
    let step = Math.round(majorSpacingPx / minorSpacingPx);
    if (step < 1) step = 1;
    
    // Compute potential major indices
    let majorIndices = [];
    for (let i = 0; i < minorIntervals; i += step) {
        majorIndices.push(i);
    }
    majorIndices.push(minorIntervals); // Always include the last
    
    // Check and adjust if the last intermediate major is too close to the end
    if (majorIndices.length > 2) {
        const penultimate = majorIndices[majorIndices.length - 2];
        const last = majorIndices[majorIndices.length - 1];
        const dist = (last - penultimate) * minorSpacingPx;
        if (dist < minPxMajor) {
            majorIndices.splice(majorIndices.length - 2, 1);
        }
    }
    
    // Draw ticks
    for (let i = 0; i <= minorIntervals; i++) {
        const pos = i * minorSpacingPx;
        const daysFromStart = i * (totalDays / minorIntervals);
        const date = new Date(minDate.getTime() + daysFromStart * 86400000);
        
        const isMajor = majorIndices.includes(i);
        
        if (isMajor) {
            // Draw long tick
            const tick = document.createElement('div');
            tick.className = 'major-line';
            tick.style.top = `${pos}px`;
            container.appendChild(tick);
            
            // Label
            const label = document.createElement('div');
            label.className = 'major-line-text';
            label.style.top = `${pos}px`;
            label.textContent = formatDate(date);
            container.appendChild(label);
        } else {
            // Draw short tick
            const tick = document.createElement('div');
            tick.className = 'minor-line';
            tick.style.top = `${pos}px`;
            container.appendChild(tick);
        }
    }
};