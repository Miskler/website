(function () {
  "use strict";

  const root = document.querySelector(".site-background");
  const surface = document.getElementById("site-background-surface");

  if (!root || !surface) {
    return;
  }

  const mode = pickMode();
  root.dataset.background = mode;

  const stop = mode === "life"
    ? startLifeBackground(surface)
    : startPerlinBackground(surface);

  window.addEventListener("pagehide", stop, { once: true });

  function pickMode() {
    const modes = ["life", "perlin"];

    if (window.crypto && typeof window.crypto.getRandomValues === "function") {
      const values = new Uint32Array(1);
      window.crypto.getRandomValues(values);
      return modes[values[0] % modes.length];
    }

    return modes[Math.floor(Math.random() * modes.length)];
  }

  function measureCell(surfaceElement) {
    const probe = document.createElement("span");
    probe.textContent = "M";
    probe.style.cssText = [
      "position:absolute",
      "left:0",
      "top:0",
      "visibility:hidden",
      "white-space:pre",
      "font:inherit",
      "line-height:inherit",
      "letter-spacing:inherit",
    ].join(";");

    surfaceElement.appendChild(probe);
    const rect = probe.getBoundingClientRect();
    probe.remove();

    return {
      width: rect.width || 8,
      height: rect.height || 13,
    };
  }

  function startLifeBackground(surfaceElement) {
    const aliveChars = ["█", "▓", "▒", "░", "■", "▪"];
    const deadChar = " ";

    const GLIDER = [
      [1, 0],
      [2, 1],
      [0, 2],
      [1, 2],
      [2, 2],
    ];

    const LWSS = [
      [1, 0], [4, 0],
      [0, 1],
      [0, 2], [4, 2],
      [0, 3], [1, 3], [2, 3], [3, 3],
    ];

    let cols = 0;
    let rows = 0;
    let charW = 8;
    let charH = 12;
    let grid = new Uint8Array(0);
    let next = new Uint8Array(0);
    let generation = 0;
    let lastTick = 0;
    let rafId = 0;
    let pointerX = 0;
    let pointerY = 0;
    let pointerActive = false;
    let pointerWasActive = false;
    let lastPointerCellX = -1;
    let lastPointerCellY = -1;

    const cleanup = [];

    function add(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      cleanup.push(() => target.removeEventListener(type, handler, options));
    }

    function resize() {
      const size = measureCell(surfaceElement);

      charW = size.width || 8;
      charH = size.height || 12;

      cols = Math.ceil(window.innerWidth / charW) + 2;
      rows = Math.ceil(window.innerHeight / charH) + 2;
      grid = new Uint8Array(cols * rows);
      next = new Uint8Array(cols * rows);
      generation = 0;

      randomize();
      render();
    }

    function randomize() {
      for (let i = 0; i < grid.length; i++) {
        grid[i] = Math.random() > 0.86 ? 1 : 0;
      }

      for (let i = 0; i < 18; i++) {
        spawnGlider();
      }

      for (let i = 0; i < 8; i++) {
        spawnLWSS();
      }
    }

    function index(x, y) {
      return y * cols + x;
    }

    function setCell(x, y, value = 1) {
      x = (x + cols) % cols;
      y = (y + rows) % rows;
      grid[index(x, y)] = value;
    }

    function clearArea(x, y, width, height) {
      for (let yy = 0; yy < height; yy++) {
        for (let xx = 0; xx < width; xx++) {
          setCell(x + xx, y + yy, 0);
        }
      }
    }

    function spawnPattern(pattern, x, y, flipX = false, flipY = false) {
      for (const [px, py] of pattern) {
        setCell(x + (flipX ? -px : px), y + (flipY ? -py : py), 1);
      }
    }

    function clientToCell(clientX, clientY) {
      return {
        x: Math.floor(clientX / window.innerWidth * cols),
        y: Math.floor(clientY / window.innerHeight * rows),
      };
    }

    function emitCellsFromPointer(strong = false) {
      if (!pointerActive || cols <= 0 || rows <= 0) {
        return;
      }

      const cell = clientToCell(pointerX, pointerY);
      const sameCell = cell.x === lastPointerCellX && cell.y === lastPointerCellY;

      if (sameCell && !strong) {
        return;
      }

      lastPointerCellX = cell.x;
      lastPointerCellY = cell.y;

      const radius = strong ? 4 : 2;
      const attempts = strong ? 34 : 12;

      for (let i = 0; i < attempts; i++) {
        const dx = Math.floor(Math.random() * (radius * 2 + 1)) - radius;
        const dy = Math.floor(Math.random() * (radius * 2 + 1)) - radius;
        const distance = Math.abs(dx) + Math.abs(dy);

        if (distance <= radius + 1 && Math.random() > distance / (radius + 2)) {
          setCell(cell.x + dx, cell.y + dy, 1);
        }
      }

      if (strong || Math.random() > 0.82) {
        spawnPattern(
          GLIDER,
          cell.x - 1,
          cell.y - 1,
          Math.random() > 0.5,
          Math.random() > 0.5
        );
      }
    }

    function spawnGlider() {
      const margin = 6;
      const cx = cols / 2;
      const cy = rows / 2;
      const edge = Math.floor(Math.random() * 4);

      let x;
      let y;

      if (edge === 0) {
        x = Math.floor(Math.random() * cols);
        y = margin;
      } else if (edge === 1) {
        x = cols - margin;
        y = Math.floor(Math.random() * rows);
      } else if (edge === 2) {
        x = Math.floor(Math.random() * cols);
        y = rows - margin;
      } else {
        x = margin;
        y = Math.floor(Math.random() * rows);
      }

      const wantsRight = x < cx;
      const wantsDown = y < cy;
      const flipX = !wantsRight;
      const flipY = !wantsDown;

      clearArea(x - 5, y - 5, 12, 12);
      spawnPattern(GLIDER, x, y, flipX, flipY);
    }

    function spawnLWSS() {
      const fromLeft = Math.random() > 0.5;
      const x = fromLeft ? 4 : cols - 10;
      const centerBand = Math.max(4, Math.floor(rows * 0.28));
      const minY = Math.max(2, Math.floor(rows / 2 - centerBand / 2));
      const y = minY + Math.floor(Math.random() * centerBand);

      clearArea(x - 5, y - 5, 18, 14);
      spawnPattern(LWSS, x, y, !fromLeft, false);
    }

    function injectShips() {
      if (generation % 18 === 0) spawnGlider();
      if (generation % 52 === 0) spawnLWSS();

      if (generation % 140 === 0) {
        const x = Math.floor(Math.random() * cols);
        const y = Math.floor(Math.random() * rows);
        clearArea(x - 6, y - 4, 14, 10);

        for (let i = 0; i < 18; i++) {
          setCell(
            x + Math.floor(Math.random() * 12) - 6,
            y + Math.floor(Math.random() * 8) - 4,
            1
          );
        }
      }
    }

    function countNeighbors(x, y) {
      let count = 0;

      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;

          const nx = (x + dx + cols) % cols;
          const ny = (y + dy + rows) % rows;

          count += grid[index(nx, ny)];
        }
      }

      return count;
    }

    function step() {
      let aliveCount = 0;
      let changedCount = 0;

      generation++;
      injectShips();
      emitCellsFromPointer(pointerWasActive && generation % 2 === 0);

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const i = index(x, y);
          const alive = grid[i];
          const neighbors = countNeighbors(x, y);

          if (alive) {
            next[i] = neighbors === 2 || neighbors === 3 ? 1 : 0;
          } else {
            next[i] = neighbors === 3 ? 1 : 0;
          }

          aliveCount += next[i];
          changedCount += next[i] !== alive ? 1 : 0;
        }
      }

      [grid, next] = [next, grid];

      const tooEmpty = aliveCount < grid.length * 0.018;
      const tooStatic = changedCount < grid.length * 0.004;

      if (tooEmpty || tooStatic) {
        for (let i = 0; i < 12; i++) spawnGlider();
        for (let i = 0; i < 5; i++) spawnLWSS();
      }
    }

    function render() {
      let out = "";
      const time = performance.now() / 180;

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          if (grid[index(x, y)]) {
            const charIndex = (x * 13 + y * 7 + time) | 0;
            out += aliveChars[Math.abs(charIndex) % aliveChars.length];
          } else {
            out += deadChar;
          }
        }

        out += "\n";
      }

      surfaceElement.textContent = out;
    }

    function loop(now) {
      if (now - lastTick > 90) {
        step();
        render();
        lastTick = now;
      }

      rafId = window.requestAnimationFrame(loop);
    }

    function onPointerMove(event) {
      pointerX = event.clientX;
      pointerY = event.clientY;
      pointerActive = true;
      pointerWasActive = true;
      emitCellsFromPointer(true);
    }

    function onPointerDown(event) {
      pointerX = event.clientX;
      pointerY = event.clientY;
      pointerActive = true;
      pointerWasActive = true;
      emitCellsFromPointer(true);
    }

    function onPointerLeave() {
      pointerActive = false;
      lastPointerCellX = -1;
      lastPointerCellY = -1;
    }

    add(window, "resize", resize);
    add(window, "pointermove", onPointerMove);
    add(window, "pointerdown", onPointerDown);
    add(window, "pointerleave", onPointerLeave);

    resize();
    rafId = window.requestAnimationFrame(loop);

    return function stopLifeBackground() {
      window.cancelAnimationFrame(rafId);
      while (cleanup.length > 0) {
        cleanup.pop()();
      }
    };
  }

  function startPerlinBackground(surfaceElement) {
    class PerlinNoise {
      constructor(seed = 1337) {
        this.permutation = new Uint8Array(512);
        const p = new Uint8Array(256);

        for (let i = 0; i < 256; i++) p[i] = i;

        let s = seed >>> 0;
        const rnd = () => {
          s = (s * 1664525 + 1013904223) >>> 0;
          return s / 4294967296;
        };

        for (let i = 255; i > 0; i--) {
          const j = Math.floor(rnd() * (i + 1));
          [p[i], p[j]] = [p[j], p[i]];
        }

        for (let i = 0; i < 512; i++) this.permutation[i] = p[i & 255];
      }

      fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
      }

      lerp(a, b, t) {
        return a + t * (b - a);
      }

      grad(hash, x, y, z) {
        const h = hash & 15;
        const u = h < 8 ? x : y;
        const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
      }

      noise(x, y, z = 0) {
        const X = Math.floor(x) & 255;
        const Y = Math.floor(y) & 255;
        const Z = Math.floor(z) & 255;

        x -= Math.floor(x);
        y -= Math.floor(y);
        z -= Math.floor(z);

        const u = this.fade(x);
        const v = this.fade(y);
        const w = this.fade(z);
        const p = this.permutation;

        const A = p[X] + Y;
        const AA = p[A] + Z;
        const AB = p[A + 1] + Z;
        const B = p[X + 1] + Y;
        const BA = p[B] + Z;
        const BB = p[B + 1] + Z;

        return this.lerp(
          this.lerp(
            this.lerp(this.grad(p[AA], x, y, z), this.grad(p[BA], x - 1, y, z), u),
            this.lerp(this.grad(p[AB], x, y - 1, z), this.grad(p[BB], x - 1, y - 1, z), u),
            v
          ),
          this.lerp(
            this.lerp(this.grad(p[AA + 1], x, y, z - 1), this.grad(p[BA + 1], x - 1, y, z - 1), u),
            this.lerp(this.grad(p[AB + 1], x, y - 1, z - 1), this.grad(p[BB + 1], x - 1, y - 1, z - 1), u),
            v
          ),
          w
        );
      }

      fbm(x, y, z) {
        let value = 0;
        let amplitude = 0.5;
        let frequency = 1;

        for (let i = 0; i < 5; i++) {
          value += amplitude * this.noise(x * frequency, y * frequency, z * frequency);
          frequency *= 2;
          amplitude *= 0.5;
        }

        return value;
      }
    }

    const perlin = new PerlinNoise(Date.now());
    const chars = "  ..::--==++**##%%@@";

    let cols = 0;
    let rows = 0;
    let charW = 8;
    let charH = 13;
    let lastText = "";
    let rafId = 0;
    const mouse = {
      x: window.innerWidth / 2,
      y: window.innerHeight / 2,
      active: false,
      power: 0,
    };

    const cleanup = [];

    function add(target, type, handler, options) {
      target.addEventListener(type, handler, options);
      cleanup.push(() => target.removeEventListener(type, handler, options));
    }

    function measure() {
      const size = measureCell(surfaceElement);
      charW = size.width || 8;
      charH = size.height || 13;
      cols = Math.ceil(window.innerWidth / charW) + 2;
      rows = Math.ceil(window.innerHeight / charH) + 2;
      lastText = "";
    }

    function render(time) {
      const t = time * 0.00022;
      const scale = 0.085;
      const lines = [];

      for (let y = 0; y < rows; y++) {
        let line = "";

        for (let x = 0; x < cols; x++) {
          const px = x * charW;
          const py = y * charH;
          const dx = px - mouse.x;
          const dy = py - mouse.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const cursorRadius = 170;
          const cursorForce = mouse.power * Math.max(0, 1 - distance / cursorRadius);

          const cursorScale = 1 + cursorForce * 3.2;
          const cursorTurbulence = perlin.fbm(
            x * scale * cursorScale + 100,
            y * scale * cursorScale - 100,
            0.42
          );

          const wave = perlin.fbm(x * scale, y * scale, t);
          const detail = perlin.fbm(x * scale * 2.7 + 37, y * scale * 2.7 - 21, t * 1.4);
          const glow = perlin.noise(x * 0.04 + 50, y * 0.04 - 50, t * 0.9);

          let v = wave * 0.95 + detail * 0.42 + glow * 0.22 + cursorTurbulence * cursorForce * 1.1;
          v = (v + 1) / 2;
          v = (v - 0.5) * 1.75 + 0.5;
          v += cursorForce * 0.26;
          v = Math.pow(v, 0.74);

          if (!Number.isFinite(v)) v = 0;
          v = Math.max(0, Math.min(0.999999, v));

          const i = Math.floor(v * chars.length);
          line += chars.charAt(i);
        }

        lines.push(line);
      }

      const text = lines.join("\n");

      if (text !== lastText) {
        surfaceElement.textContent = text;
        lastText = text;
      }

      const targetMousePower = mouse.active ? 1 : 0;
      mouse.power += (targetMousePower - mouse.power) * 0.08;
      rafId = window.requestAnimationFrame(render);
    }

    function onPointerMove(event) {
      mouse.x = event.clientX;
      mouse.y = event.clientY;
      mouse.active = true;
      mouse.power = 1;
    }

    function onPointerLeave() {
      mouse.active = false;
    }

    add(window, "resize", measure);
    add(window, "pointermove", onPointerMove);
    add(window, "pointerleave", onPointerLeave);

    measure();
    rafId = window.requestAnimationFrame(render);

    return function stopPerlinBackground() {
      window.cancelAnimationFrame(rafId);
      while (cleanup.length > 0) {
        cleanup.pop()();
      }
    };
  }
})();
