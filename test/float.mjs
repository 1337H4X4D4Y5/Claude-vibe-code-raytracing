// Can a light ball find equilibrium at the water surface?
// Boussinesq buoyancy threshold is rho_ref = 0.55.
// For density 0.3 ball in water (rho_phase = 1):
//   Submerged fraction at equilibrium = 0.3 / 0.45 = 0.667
//   With r=4, cap formula gives cy_eq ~ 8 with waterline at 7.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

function trial(label, startCy, density) {
  const NX = 40, NY = 28, NZ = 40;
  const sim = new Sim(NX, NY, NZ);
  sim.tau = 0.8; sim.tauPhi = 0.7; sim.sigma = 0.008;
  sim.gravity = 0.0015; sim.rhoL = 1.0; sim.rhoG = 0.1; sim.W = 4.0;
  sim.initFlat(NY * 0.25);   // waterline at y=7
  const solid = new RigidSphere(NX, NY, NZ, {
    cx: NX*0.5, cy: startCy, cz: NZ*0.5, r: 4, density,
  });
  console.log(`=== ${label} startCy=${startCy} density=${density} ===`);
  for (let s = 1; s <= 600; s++) {
    sim.step(solid);
    solid.absorbDeadCells(sim);
    solid.step(1.0, sim.gravity);
    if (s % 60 === 0) {
      console.log(`  s=${s.toString().padStart(3)} cy=${solid.cy.toFixed(2)} |v|=${Math.hypot(solid.vx, solid.vy, solid.vz).toFixed(4)}`);
    }
  }
}

trial('placed at floor   d=0.3', 21.5, 0.3);
trial('placed at surface d=0.3', 8,    0.3);
trial('placed at floor   d=1.4', 21.5, 1.4);
trial('placed near surface d=1.4', 9,  1.4);
