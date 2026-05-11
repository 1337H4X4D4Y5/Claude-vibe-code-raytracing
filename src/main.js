// Entry point. The simulation lives in a Web Worker (src/worker.js) so the
// CPU LBM doesn't block rendering or pointer interaction. The main thread
// owns the WebGL2 renderer and forwards UI to the worker.

import { Renderer } from './render.js';

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
let pointerState = null;

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

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('pointerdown', (ev) => {
  canvas.setPointerCapture(ev.pointerId);
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
    pointerState = {
      type: 'drag-sphere',
      hitDepth: t,
      initialOffset: [hit[0] - sc[0], hit[1] - sc[1], hit[2] - sc[2]],
    };
    worker.postMessage({
      type: 'drag', active: true,
      x: solidState.cx, y: solidState.cy, z: solidState.cz,
    });
  } else {
    pointerState = { type: 'orbit', x: ev.clientX, y: ev.clientY };
  }
});

canvas.addEventListener('pointermove', (ev) => {
  if (!pointerState) return;
  if (pointerState.type === 'orbit') {
    const dx = ev.clientX - pointerState.x;
    const dy = ev.clientY - pointerState.y;
    pointerState.x = ev.clientX;
    pointerState.y = ev.clientY;
    renderer.azimuth -= dx * 0.008;
    renderer.elev    = Math.max(-1.4, Math.min(1.4, renderer.elev - dy * 0.008));
  } else if (pointerState.type === 'drag-sphere') {
    const [nx, ny] = canvasNorm(ev);
    const ray = renderer.pointerRay(nx, ny);
    const t = pointerState.hitDepth;
    const wx = ray.origin[0] + ray.dir[0] * t - pointerState.initialOffset[0];
    const wy = ray.origin[1] + ray.dir[1] * t - pointerState.initialOffset[1];
    const wz = ray.origin[2] + ray.dir[2] * t - pointerState.initialOffset[2];
    const M = renderer.maxN;
    worker.postMessage({
      type: 'drag', active: true,
      x: wx * M, y: wy * M, z: wz * M,
    });
  }
});

function endDrag() {
  if (pointerState?.type === 'drag-sphere') {
    worker.postMessage({ type: 'drag', active: false });
  }
  pointerState = null;
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

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
