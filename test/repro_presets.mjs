// Reproduce dam break + buoyant rise stability bugs.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

function runScene(name, setup) {
  const NX = 40, NY = 28, NZ = 40;
  const sim = new Sim(NX, NY, NZ);
  sim.tau = 0.85; sim.tauPhi = 0.7; sim.sigma = 0.008;
  sim.gravity = 0.0015; sim.rhoL = 1.0; sim.rhoG = 0.1; sim.W = 4.0;

  const solid = new RigidSphere(NX, NY, NZ, {
    cx: NX * 0.5, cy: NY * 0.2, cz: NZ * 0.5, r: 4, density: 1.6,
  });

  setup(sim, solid, NX, NY, NZ);

  console.log(`=== ${name} ===`);
  console.log(`initial solid: cx=${solid.cx.toFixed(2)} cy=${solid.cy.toFixed(2)} cz=${solid.cz.toFixed(2)} r=${solid.r.toFixed(2)} rho=${solid.density.toFixed(2)}`);

  for (let s = 1; s <= 120; s++) {
    sim.step(solid);
    solid.absorbDeadCells(sim);
    solid.step(1.0, sim.gravity);

    // Check for NaN
    let nan = 0, infCount = 0, maxU = 0;
    for (let i = 0; i < sim.n; i++) {
      if (!Number.isFinite(sim.phi[i]) || !Number.isFinite(sim.rho[i]) ||
          !Number.isFinite(sim.ux[i]) || !Number.isFinite(sim.uy[i]) || !Number.isFinite(sim.uz[i])) {
        nan++;
      }
      const u = Math.hypot(sim.ux[i], sim.uy[i], sim.uz[i]);
      if (u > maxU) maxU = u;
    }
    const badSolid = !Number.isFinite(solid.cx) || !Number.isFinite(solid.cy) || !Number.isFinite(solid.cz);

    if (s % 20 === 0 || nan > 0 || badSolid || maxU > 1.0) {
      console.log(`  step ${s.toString().padStart(3)}: solid=(${solid.cx.toFixed(2)},${solid.cy.toFixed(2)},${solid.cz.toFixed(2)}) v=(${solid.vx.toFixed(3)},${solid.vy.toFixed(3)},${solid.vz.toFixed(3)}) maxU=${maxU.toFixed(3)} nan=${nan}`);
    }

    if (nan > 0 || badSolid) {
      console.log(`  BLEW UP at step ${s}`);
      break;
    }
  }
}

runScene('drop', (sim, solid, NX, NY, NZ) => {
  sim.initFlat(NY * 0.25);
  solid.cx = NX * 0.5; solid.cy = NY * 0.35; solid.cz = NZ * 0.5;
  solid.setRadius(4.0); solid.setDensity(2.0);
});

runScene('dam', (sim, solid, NX, NY, NZ) => {
  sim.initFlat(NY + 1);
  const w7 = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8];
  for (let z = 1; z < NZ - 1; z++) {
    for (let y = 1; y < NY - 1; y++) {
      for (let x = 1; x < NX - 1; x++) {
        const i = sim.idx(x, y, z);
        const inCol = x < NX * 0.35 && y > NY * 0.35;
        const p = inCol ? 1 : 0;
        sim.phi[i] = p; sim.phiPrev[i] = p;
        for (let k = 0; k < 7; k++) sim.h[i * 7 + k] = w7[k] * p;
      }
    }
  }
  sim.seedHydrostatic();
  solid.cx = NX * 0.7; solid.cy = NY * 0.6; solid.cz = NZ * 0.5;
  solid.setRadius(3.5); solid.setDensity(0.5);
});

runScene('rise', (sim, solid, NX, NY, NZ) => {
  sim.initFlat(NY * 0.25);
  solid.cx = NX * 0.5; solid.cy = NY * 0.65; solid.cz = NZ * 0.5;
  solid.setRadius(4.0); solid.setDensity(0.5);
});
