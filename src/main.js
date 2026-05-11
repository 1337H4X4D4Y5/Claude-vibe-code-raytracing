// Entry point: wire the simulation, renderer, UI and pointer interaction
// into one main loop. The simulation runs CPU-side, the renderer runs on
// WebGL2.

import { Sim } from './sim.js';
import { RigidSphere } from './solid.js';
import { Renderer } from './render.js';

// ---- grid + scene setup -----------------------------------------------------
const NX = 40, NY = 28, NZ = 40;
const sim = new Sim(NX, NY, NZ);
sim.tau     = 0.55;
sim.tauPhi  = 0.7;
sim.sigma   = 0.012;
sim.gravity = 0.00135;
sim.rhoL    = 1.0;
sim.rhoG    = 0.1;
sim.W       = 4.0;

const solid = new RigidSphere(NX, NY, NZ, {
  cx: NX * 0.5, cy: NY * 0.20, cz: NZ * 0.5, r: 4.0, density: 1.6,
});

const canvas = document.getElementById('view');
const renderer = new Renderer(canvas, NX, NY, NZ);

let running = true;
let lastTime = 0;

resetTo('drop');

// ---- presets ----------------------------------------------------------------
function resetTo(name) {
  switch (name) {
    case 'drop':
      sim.initFlat(NY * 0.45);
      solid.cx = NX * 0.5; solid.cy = NY * 0.15; solid.cz = NZ * 0.5;
      solid.vx = solid.vy = solid.vz = 0;
      solid.wx = solid.wy = solid.wz = 0;
      solid.setRadius(4.0);
      solid.setDensity(1.6);
      break;
    case 'dam': {
      // Liquid column on one side.
      const { nx, ny, nz } = sim;
      const w7 = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8];
      sim.initFlat(NY + 1);  // start all gas, then carve the column
      for (let z = 1; z < nz - 1; z++) {
        for (let y = 1; y < ny - 1; y++) {
          for (let x = 1; x < nx - 1; x++) {
            const i = sim.idx(x, y, z);
            const inCol = x < nx * 0.35 && y > ny * 0.35;
            const p = inCol ? 1 : 0;
            sim.phi[i] = p;
            sim.phiPrev[i] = p;
            for (let k = 0; k < 7; k++) sim.h[i * 7 + k] = w7[k] * p;
          }
        }
      }
      solid.cx = NX * 0.7; solid.cy = NY * 0.6; solid.cz = NZ * 0.5;
      solid.vx = solid.vy = solid.vz = 0;
      solid.wx = solid.wy = solid.wz = 0;
      solid.setRadius(3.5);
      solid.setDensity(0.5);
      break;
    }
    case 'rise':
      sim.initFlat(NY * 0.25);
      solid.cx = NX * 0.5; solid.cy = NY * 0.85; solid.cz = NZ * 0.5;
      solid.vx = solid.vy = solid.vz = 0;
      solid.wx = solid.wy = solid.wz = 0;
      solid.setRadius(4.0);
      solid.setDensity(0.4);     // floats
      break;
  }
}

// ---- UI ---------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

$('reset').addEventListener('click', () => resetTo($('preset').value));
$('pause').addEventListener('click', () => {
  running = !running;
  $('pause').textContent = running ? 'Pause' : 'Resume';
});
$('preset').addEventListener('change', (e) => resetTo(e.target.value));

function bindSlider(id, valId, setter, format = (v) => v.toFixed(3)) {
  const el = $(id), out = $(valId);
  const apply = () => { const v = parseFloat(el.value); setter(v); out.textContent = format(v); };
  el.addEventListener('input', apply);
  apply();
}
bindSlider('sigma',   'sigmaVal',   (v) => { sim.sigma = v; });
bindSlider('gravity', 'gravityVal', (v) => { sim.gravity = v; }, (v) => v.toFixed(4));
bindSlider('rhoRatio', 'rhoRatioVal',
           (v) => { sim.rhoL = 1.0; sim.rhoG = 1.0 / v; },
           (v) => v.toFixed(0));
bindSlider('solidRho','solidRhoVal',(v) => { solid.setDensity(v); }, (v) => v.toFixed(2));

// ---- pointer interaction ----------------------------------------------------
// Left drag on empty space   -> orbit camera
// Left drag with sphere hit  -> drag sphere in screen-parallel plane
// Right drag                 -> orbit camera
// Wheel                       -> zoom
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

  // Convert grid coords to renderer's world coords (longest axis -> 1).
  const M = renderer.maxN;
  const sc = [solid.cx / M, solid.cy / M, solid.cz / M];
  const sr = solid.r / M;

  const ray = renderer.pointerRay(nx, ny);
  const t = raySphereHit(ray.origin, ray.dir, sc, sr);

  if (ev.button === 0 && t > 0) {
    // Begin sphere drag. Record the world-space hit point depth (along
    // view forward) so we can project later mouse positions onto that plane.
    const hit = [
      ray.origin[0] + ray.dir[0] * t,
      ray.origin[1] + ray.dir[1] * t,
      ray.origin[2] + ray.dir[2] * t,
    ];
    pointerState = {
      type: 'drag-sphere',
      hitDepth: t,
      ray,
      initialOffset: [
        hit[0] - sc[0],
        hit[1] - sc[1],
        hit[2] - sc[2],
      ],
    };
    solid.dragging = true;
    solid.dragTargetX = solid.cx;
    solid.dragTargetY = solid.cy;
    solid.dragTargetZ = solid.cz;
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
    // Project ray to the same depth (distance along ray from origin) as
    // the initial hit; this is the simplest stable scheme and keeps the
    // sphere on the screen plane it was clicked from.
    const t = pointerState.hitDepth;
    const wx = ray.origin[0] + ray.dir[0] * t - pointerState.initialOffset[0];
    const wy = ray.origin[1] + ray.dir[1] * t - pointerState.initialOffset[1];
    const wz = ray.origin[2] + ray.dir[2] * t - pointerState.initialOffset[2];
    const M = renderer.maxN;
    solid.dragTargetX = wx * M;
    solid.dragTargetY = wy * M;
    solid.dragTargetZ = wz * M;
  }
});

canvas.addEventListener('pointerup', () => {
  if (pointerState?.type === 'drag-sphere') solid.dragging = false;
  pointerState = null;
});
canvas.addEventListener('pointercancel', () => {
  if (pointerState?.type === 'drag-sphere') solid.dragging = false;
  pointerState = null;
});

canvas.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  renderer.dist = Math.max(0.6, Math.min(6.0,
                  renderer.dist * Math.exp(ev.deltaY * 0.0015)));
}, { passive: false });

// ---- main loop --------------------------------------------------------------
function frame(now) {
  const t = (now || 0) * 0.001;
  if (!lastTime) lastTime = t;
  const dt = Math.min(1 / 30, t - lastTime);
  lastTime = t;

  if (running) {
    // Sub-step the simulation. LBM is stable at dt = 1 in lattice units
    // regardless of wall-clock dt -- we just advance the body in sync.
    const substeps = 2;
    for (let s = 0; s < substeps; s++) {
      sim.step(solid);
      solid.absorbDeadCells(sim);
      solid.step(1.0, sim.gravity);
    }
  }

  renderer.uploadVolume(sim, solid);
  renderer.draw(sim, solid, t);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
