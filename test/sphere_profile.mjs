// Same as smoke.mjs but prints vertical phi profile through time.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

const sim = new Sim(12, 12, 12);
sim.initFlat(6);
sim.tau = 0.6;
sim.tauPhi = 0.7;
sim.sigma = 0.012;
sim.gravity = 0.0012;
sim.rhoL = 1.0;
sim.rhoG = 0.1;
sim.W = 4.0;

const solid = new RigidSphere(12, 12, 12, {
  cx: 6, cy: 3, cz: 6, r: 2.0, density: 1.5,
});

function profile(step) {
  let avg = 0, count = 0;
  for (let i = 0; i < sim.n; i++) { avg += sim.phi[i]; count++; }
  avg /= count;
  const col = [];
  for (let y = 0; y < sim.ny; y++) {
    let s = 0, n = 0;
    for (let z = 1; z < sim.nz - 1; z++) {
      for (let x = 1; x < sim.nx - 1; x++) {
        s += sim.phi[sim.idx(x, y, z)]; n++;
      }
    }
    col.push((s / n).toFixed(2));
  }
  console.log(`step ${String(step).padStart(3)} solidY=${solid.cy.toFixed(2)} avg=${avg.toFixed(3)} | y=`, col.join(' '));
}

profile(0);
for (let s = 1; s <= 80; s++) {
  sim.step(solid);
  solid.absorbDeadCells(sim);
  solid.step(1.0, sim.gravity);
  if (s % 10 === 0) profile(s);
}
