// Simulation Web Worker.
//
// Owns the Sim + RigidSphere. The main thread sends commands; we tick the
// sim continuously and post the packed volume buffer back per frame. The
// buffer's ArrayBuffer is transferred (zero-copy) and the worker keeps a
// small buffer pool so we don't churn the allocator.

import { Sim } from './sim.js';
import { RigidSphere } from './solid.js';

let NX = 40, NY = 28, NZ = 40;
let sim = null;
let solid = null;
let running = true;
let substeps = 2;

const bufferPool = [];
function getBuffer() {
  if (bufferPool.length) return bufferPool.pop();
  return new Uint8Array(NX * NY * NZ * 2);
}

function initSim() {
  sim = new Sim(NX, NY, NZ);
  sim.tau     = 0.55;
  sim.tauPhi  = 0.7;
  sim.sigma   = 0.012;
  sim.gravity = 0.00135;
  sim.rhoL    = 1.0;
  sim.rhoG    = 0.1;
  sim.W       = 4.0;

  solid = new RigidSphere(NX, NY, NZ, {
    cx: NX * 0.5, cy: NY * 0.20, cz: NZ * 0.5, r: 4.0, density: 1.6,
  });
}

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
      const { nx, ny, nz } = sim;
      const w7 = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8];
      sim.initFlat(NY + 1);
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
      solid.setDensity(0.4);
      break;
  }
}

function step() {
  if (running) {
    for (let s = 0; s < substeps; s++) {
      sim.step(solid);
      solid.absorbDeadCells(sim);
      solid.step(1.0, sim.gravity);
    }
  }
  const buf = getBuffer();
  sim.packVolume(buf, null);
  postMessage({
    type: 'frame',
    vol: buf,
    solid: {
      cx: solid.cx, cy: solid.cy, cz: solid.cz, r: solid.r,
      vx: solid.vx, vy: solid.vy, vz: solid.vz,
    },
  }, [buf.buffer]);
  // Continue ticking; the main thread will hand the buffer back via
  // `releaseBuf`. We don't block on it -- if a frame backs up we just
  // allocate a fresh buffer.
}

let stepScheduled = false;
function scheduleStep() {
  if (stepScheduled) return;
  stepScheduled = true;
  // setTimeout(_, 0) yields to the event loop so incoming messages
  // (drag targets, parameter changes) get processed each tick.
  setTimeout(() => { stepScheduled = false; step(); scheduleStep(); }, 0);
}

self.onmessage = (e) => {
  const m = e.data;
  switch (m.type) {
    case 'init':
      NX = m.nx; NY = m.ny; NZ = m.nz;
      substeps = m.substeps ?? 2;
      initSim();
      resetTo(m.preset ?? 'drop');
      scheduleStep();
      break;
    case 'preset':
      resetTo(m.name);
      break;
    case 'param':
      if (m.name === 'sigma') sim.sigma = m.value;
      else if (m.name === 'gravity') sim.gravity = m.value;
      else if (m.name === 'rhoRatio') { sim.rhoL = 1.0; sim.rhoG = 1.0 / m.value; }
      else if (m.name === 'solidRho') solid.setDensity(m.value);
      else if (m.name === 'substeps') substeps = m.value | 0;
      break;
    case 'drag':
      solid.dragging = !!m.active;
      if (m.active) {
        solid.dragTargetX = m.x;
        solid.dragTargetY = m.y;
        solid.dragTargetZ = m.z;
      }
      break;
    case 'pause':
      running = false; break;
    case 'resume':
      running = true; break;
    case 'releaseBuf':
      // Main thread giving us back a buffer for reuse.
      if (m.buffer.byteLength === NX * NY * NZ * 2) {
        bufferPool.push(new Uint8Array(m.buffer));
      }
      break;
  }
};
