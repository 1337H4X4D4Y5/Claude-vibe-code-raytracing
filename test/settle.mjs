// Does the ball ever come to rest? Run each preset for 400 steps and
// log position + velocity every 50 steps.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

function trial(label, setup) {
  const NX = 40, NY = 28, NZ = 40;
  const sim = new Sim(NX, NY, NZ);
  sim.tau = 0.8; sim.tauPhi = 0.7; sim.sigma = 0.008;
  sim.gravity = 0.0015; sim.rhoL = 1.0; sim.rhoG = 0.1; sim.W = 4.0;
  const solid = new RigidSphere(NX, NY, NZ, {
    cx: NX*0.5, cy: NY*0.2, cz: NZ*0.5, r: 4, density: 1.6,
  });
  setup(sim, solid, NX, NY, NZ);

  console.log(`=== ${label} ===`);
  let maxSeenU = 0;
  for (let s = 1; s <= 400; s++) {
    sim.step(solid);
    solid.absorbDeadCells(sim);
    solid.step(1.0, sim.gravity);
    let u = 0;
    for (let i = 0; i < sim.n; i++) {
      const m = Math.hypot(sim.ux[i], sim.uy[i], sim.uz[i]);
      if (m > u) u = m;
    }
    if (u > maxSeenU) maxSeenU = u;
    if (s % 50 === 0) {
      const vel = Math.hypot(solid.vx, solid.vy, solid.vz);
      console.log(`  s=${s.toString().padStart(3)} cy=${solid.cy.toFixed(2)} |v|=${vel.toFixed(4)} maxU=${u.toFixed(3)}`);
    }
  }
  console.log(`  peak maxU = ${maxSeenU.toFixed(3)}`);
}

trial('drop', (sim, solid, NX, NY, NZ) => {
  sim.initFlat(NY * 0.25);
  solid.cx = NX * 0.5; solid.cy = NY * 0.65; solid.cz = NZ * 0.5;
  solid.setRadius(4.0); solid.setDensity(1.4);
});

trial('rise', (sim, solid, NX, NY, NZ) => {
  sim.initFlat(NY * 0.25);
  solid.cx = NX * 0.5; solid.cy = NY * 0.65; solid.cz = NZ * 0.5;
  solid.setRadius(4.0); solid.setDensity(0.3);
});
