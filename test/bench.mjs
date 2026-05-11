// Time a representative simulation step so we know where to focus.
// Production grid size: 40 x 28 x 40.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

const sim = new Sim(40, 28, 40);
sim.initFlat(14);
sim.tau = 0.55; sim.tauPhi = 0.7; sim.sigma = 0.012;
sim.gravity = 0.00135; sim.rhoL = 1.0; sim.rhoG = 0.1; sim.W = 4.0;

const solid = new RigidSphere(40, 28, 40, { cx: 20, cy: 6, cz: 20, r: 4, density: 1.6 });

// Warm up V8.
for (let i = 0; i < 20; i++) {
  sim.step(solid);
  solid.absorbDeadCells(sim);
  solid.step(1.0, sim.gravity);
}

const STEPS = 80;
const t0 = performance.now();
for (let i = 0; i < STEPS; i++) {
  sim.step(solid);
  solid.absorbDeadCells(sim);
  solid.step(1.0, sim.gravity);
}
const dt = (performance.now() - t0) / STEPS;
const N = 40 * 28 * 40;
const mlups = (N / dt) * 1e-3;
console.log(`grid ${40}x${28}x${40} = ${N} cells`);
console.log(`mean step: ${dt.toFixed(2)} ms`);
console.log(`MLUPS:    ${mlups.toFixed(2)}`);
console.log(`fps@2sub:  ${(1000 / (dt * 2)).toFixed(1)}`);
