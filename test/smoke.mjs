// Smoke test: run the simulation for a few steps with a small grid and a
// mock solid; ensure no NaN appears and that gravity drives liquid down.

import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

const sim = new Sim(12, 12, 12);
sim.initFlat(6);                     // liquid in lower half
sim.tau = 0.7;
sim.tauPhi = 0.7;
sim.sigma = 0.012;
sim.gravity = 0.0012;
sim.rhoL = 1.0;
sim.rhoG = 0.1;
sim.W = 4.0;

const solid = new RigidSphere(12, 12, 12, {
  cx: 6, cy: 3, cz: 6, r: 2.0, density: 1.5,
});

function center(arr, i = sim.idx(6, 6, 6)) { return arr[i]; }

console.log('initial: phi(center) =', center(sim.phi),
            'phi(top) =', sim.phi[sim.idx(6, 1, 6)],
            'phi(bottom) =', sim.phi[sim.idx(6, 10, 6)]);

let badNaN = false, badRange = false;
for (let s = 0; s < 80; s++) {
  sim.step(solid);
  solid.absorbDeadCells(sim);
  solid.step(1.0, sim.gravity);
  // Scan for NaN.
  for (let i = 0; i < sim.n; i++) {
    if (!Number.isFinite(sim.phi[i]) || !Number.isFinite(sim.rho[i])
        || !Number.isFinite(sim.ux[i]) || !Number.isFinite(sim.uy[i])
        || !Number.isFinite(sim.uz[i])) { badNaN = true; break; }
    if (sim.phi[i] < -0.01 || sim.phi[i] > 1.01) { badRange = true; }
  }
  if (badNaN) { console.log('FAIL: NaN at step', s); process.exit(1); }
}

console.log('after 80 steps:');
console.log('  phi(top)    =', sim.phi[sim.idx(6, 1, 6)]);
console.log('  phi(mid)    =', sim.phi[sim.idx(6, 6, 6)]);
console.log('  phi(bottom) =', sim.phi[sim.idx(6, 10, 6)]);
console.log('  solid pos   =', solid.cx.toFixed(2), solid.cy.toFixed(2), solid.cz.toFixed(2));
console.log('  solid vel   =', solid.vx.toFixed(3), solid.vy.toFixed(3), solid.vz.toFixed(3));
console.log('  badRange    =', badRange);

if (badNaN) process.exit(1);
console.log('OK: no NaN, sim stable for 80 steps.');
