// Entry point. The simulation lives in a Web Worker (src/worker.js) so the
// CPU LBM doesn't block rendering or pointer interaction. The main thread
// owns the WebGL2 renderer and forwards UI to the worker.

import { Renderer } from './render.js';

// ---- version badge ----------------------------------------------------------
// The Pages workflow writes version.json next to index.html on each deploy
// with a monotonic `version` (= total commit count) plus commit sha and
// build time. We fetch it with cache:no-store + a timestamp query so the
// badge always reflects whatever bundle is actually being served.
(async () => {
  const badge = document.getElementById('version-badge');
  if (!badge) return;
  const tag    = badge.querySelector('.version-tag');
  const detail = badge.querySelector('.version-detail');
  try {
    const r = await fetch(`version.json?t=${Date.now()}`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const v = await r.json();
    const when = (v.buildTime || '').replace('T', ' ').replace(/Z$/, ' UTC');
    if (tag)    tag.textContent    = `v${v.version ?? '?'}`;
    if (detail) detail.textContent = `${v.shortSha || '?'} · ${when}`;
    badge.title = [
      `version v${v.version ?? '?'}`,
      `commit ${v.sha || '?'}`,
      `branch ${v.ref || '?'}`,
      `run #${v.runNumber ?? '?'}`,
      `built ${v.buildTime || '?'}`,
    ].join('\n');
  } catch {
    // Local dev or pre-deploy: no version.json on disk.
    if (tag)    tag.textContent    = 'dev';
    if (detail) detail.textContent = '';
    badge.title = 'No version.json -- running locally or before first Pages deploy.';
  }
})();

const NX = 40, NY = 28, NZ = 40;

const canvas = document.getElementById('view');
const renderer = new Renderer(canvas, NX, NY, NZ);

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

// Latest worker-supplied state (renderer reads these every frame).
let latestVol = null;
let solidState = { cx: NX * 0.5, cy: NY * 0.20, cz: NZ * 0.5, r: 4.0,
                    vx: 0, vy: 0, vz: 0 };
// Track a small pool of buffers we've already received so we can hand
// them back to the worker for reuse.
let inFlightBuffer = null;

worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'frame') {
    // Send the previous frame's buffer back if we haven't already.
    if (inFlightBuffer) {
      worker.postMessage({ type: 'releaseBuf', buffer: inFlightBuffer }, [inFlightBuffer]);
    }
    latestVol = m.vol;
    inFlightBuffer = null;
    solidState = m.solid;
  }
};

worker.postMessage({ type: 'init', nx: NX, ny: NY, nz: NZ, preset: 'drop' });

// ---- UI ---------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

$('reset').addEventListener('click', () => {
  worker.postMessage({ type: 'preset', name: $('preset').value });
});

let running = true;
$('pause').addEventListener('click', () => {
  running = !running;
  $('pause').textContent = running ? 'Pause' : 'Resume';
  worker.postMessage({ type: running ? 'resume' : 'pause' });
});

$('preset').addEventListener('change', (e) => {
  worker.postMessage({ type: 'preset', name: e.target.value });
});

function bindSlider(id, valId, name, format = (v) => v.toFixed(3)) {
  const el = $(id), out = $(valId);
  const apply = () => {
    const v = parseFloat(el.value);
    worker.postMessage({ type: 'param', name, value: v });
    out.textContent = format(v);
  };
  el.addEventListener('input', apply);
  apply();
}
bindSlider('sigma',    'sigmaVal',    'sigma');
bindSlider('gravity',  'gravityVal',  'gravity',  (v) => v.toFixed(4));
bindSlider('rhoRatio', 'rhoRatioVal', 'rhoRatio', (v) => v.toFixed(0));
bindSlider('solidRho', 'solidRhoVal', 'solidRho', (v) => v.toFixed(2));

// ---- pointer interaction ----------------------------------------------------
// State machine driven by all currently-down pointers:
//   1 pointer down on the sphere   -> drag-sphere
//   1 pointer down elsewhere       -> orbit
//   2 pointers down (any source)   -> pinch-zoom (suspends the other mode)
// Lifting back to 1 pointer resumes orbit from the remaining contact.
const pointers = new Map();   // pointerId -> {x, y}
let mode = null;
let orbitState = null;        // { lastX, lastY }
let dragState  = null;        // { hitDepth, initialOffset }
let pinchState = null;        // { dist0, zoom0 }

function canvasNorm(ev) {
  const r = canvas.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
}

function raySphereHit(origin, dir, c, r) {
  const ox = origin[0] - c[0];
  const oy = origin[1] - c[1];
  const oz = origin[2] - c[2];
  const b = ox * dir[0] + oy * dir[1] + oz * dir[2];
  const cc = ox * ox + oy * oy + oz * oz - r * r;
  const disc = b * b - cc;
  if (disc < 0) return -1;
  const t = -b - Math.sqrt(disc);
  return t > 0 ? t : -1;
}

function activePointers() { return Array.from(pointers.values()); }
function pinchDist() {
  const a = activePointers();
  return a.length < 2 ? 0 : Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
}

function endSphereDrag() {
  if (mode === 'drag-sphere') worker.postMessage({ type: 'drag', active: false });
  dragState = null;
}

function startOrbitFromFirstPointer() {
  const p = activePointers()[0];
  orbitState = { lastX: p.x, lastY: p.y };
  mode = 'orbit';
}

function startPinch() {
  endSphereDrag();
  orbitState = null;
  pinchState = { dist0: pinchDist(), zoom0: renderer.dist };
  mode = 'pinch';
}

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('pointerdown', (ev) => {
  canvas.setPointerCapture(ev.pointerId);
  pointers.set(ev.pointerId, { x: ev.clientX, y: ev.clientY });

  if (pointers.size >= 2) { startPinch(); return; }

  // Single pointer down: sphere hit-test, fall back to orbit.
  const [nx, ny] = canvasNorm(ev);
  const M = renderer.maxN;
  const sc = [solidState.cx / M, solidState.cy / M, solidState.cz / M];
  const sr = solidState.r / M;
  const ray = renderer.pointerRay(nx, ny);
  const t = raySphereHit(ray.origin, ray.dir, sc, sr);

  if (ev.button === 0 && t > 0) {
    const hit = [
      ray.origin[0] + ray.dir[0] * t,
      ray.origin[1] + ray.dir[1] * t,
      ray.origin[2] + ray.dir[2] * t,
    ];
    dragState = {
      hitDepth: t,
      initialOffset: [hit[0] - sc[0], hit[1] - sc[1], hit[2] - sc[2]],
    };
    mode = 'drag-sphere';
    worker.postMessage({
      type: 'drag', active: true,
      x: solidState.cx, y: solidState.cy, z: solidState.cz,
    });
  } else {
    orbitState = { lastX: ev.clientX, lastY: ev.clientY };
    mode = 'orbit';
  }
});

canvas.addEventListener('pointermove', (ev) => {
  const p = pointers.get(ev.pointerId);
  if (!p) return;
  p.x = ev.clientX; p.y = ev.clientY;

  if (mode === 'pinch') {
    const d = pinchDist();
    if (d > 0 && pinchState && pinchState.dist0 > 0) {
      // Ratio: spread fingers -> zoom in (smaller dist).
      renderer.dist = Math.max(0.6, Math.min(6.0,
                      pinchState.zoom0 * pinchState.dist0 / d));
    }
    return;
  }

  if (mode === 'orbit' && orbitState) {
    const dx = ev.clientX - orbitState.lastX;
    const dy = ev.clientY - orbitState.lastY;
    orbitState.lastX = ev.clientX;
    orbitState.lastY = ev.clientY;
    // Drag-the-scene convention: the spot under the cursor follows the
    // cursor, i.e. dragging right rotates the scene right which means
    // the camera moves left around the target.
    renderer.azimuth += dx * 0.008;
    renderer.elev    = Math.max(-1.4, Math.min(1.4, renderer.elev + dy * 0.008));
    return;
  }

  if (mode === 'drag-sphere' && dragState) {
    const [nx, ny] = canvasNorm(ev);
    const ray = renderer.pointerRay(nx, ny);
    const t = dragState.hitDepth;
    const wx = ray.origin[0] + ray.dir[0] * t - dragState.initialOffset[0];
    const wy = ray.origin[1] + ray.dir[1] * t - dragState.initialOffset[1];
    const wz = ray.origin[2] + ray.dir[2] * t - dragState.initialOffset[2];
    const M = renderer.maxN;
    worker.postMessage({
      type: 'drag', active: true,
      x: wx * M, y: wy * M, z: wz * M,
    });
  }
});

function endPointer(ev) {
  pointers.delete(ev.pointerId);
  if (pointers.size === 0) {
    endSphereDrag();
    orbitState = null;
    pinchState = null;
    mode = null;
    return;
  }
  if (mode === 'pinch' && pointers.size === 1) {
    // Resume orbit from the remaining finger.
    pinchState = null;
    startOrbitFromFirstPointer();
  }
}
canvas.addEventListener('pointerup', endPointer);
canvas.addEventListener('pointercancel', endPointer);

canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  renderer.dist = Math.max(0.6, Math.min(6.0,
                  renderer.dist * Math.exp(ev.deltaY * 0.0015)));
}, { passive: false });

// ---- main render loop -------------------------------------------------------
// Decoupled from the simulation; we just render the latest volume the
// worker has produced, at the display's refresh rate.
const simStub = { nx: NX, ny: NY, nz: NZ };

function frame(now) {
  if (latestVol) {
    renderer.uploadVolumeRaw(latestVol);
    // Hand the buffer back to the worker next message round-trip.
    inFlightBuffer = latestVol.buffer;
    latestVol = null;
  }
  renderer.draw(simStub, solidState, (now || 0) * 0.001);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
