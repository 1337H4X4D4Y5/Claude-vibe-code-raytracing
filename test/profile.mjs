// Probe vertical phi profile through time.
import { Sim } from '../src/sim.js';

const sim = new Sim(8, 12, 8);
sim.initFlat(6);
sim.tau = 0.7;
sim.tauPhi = 0.7;
sim.sigma = 0.012;
sim.gravity = 0.0012;
sim.rhoL = 1.0;
sim.rhoG = 0.1;
sim.W = 4.0;

const noSolid = null;

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
  console.log(`step ${String(step).padStart(3)} | avg=${avg.toFixed(3)} | y=`, col.join(' '));
}

profile(0);
for (let s = 1; s <= 60; s++) {
  sim.step(noSolid);
  if (s % 10 === 0) profile(s);
}
