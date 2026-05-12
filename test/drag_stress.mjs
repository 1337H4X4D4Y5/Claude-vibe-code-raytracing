// Simulate the user dragging the sphere through water and through
// the interface, verify it doesn't explode.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

const NX = 40, NY = 28, NZ = 40;
const sim = new Sim(NX, NY, NZ);
sim.tau = 0.8; sim.tauPhi = 0.7; sim.sigma = 0.008;
sim.gravity = 0.0015; sim.rhoL = 1.0; sim.rhoG = 0.1; sim.W = 4.0;
sim.initFlat(NY * 0.25);

const solid = new RigidSphere(NX, NY, NZ, {
  cx: NX*0.5, cy: NY*0.65, cz: NZ*0.5, r: 4, density: 1.4,
});

solid.dragging = true;
// Drag the sphere on a small loop: down -> right -> up (through
// interface) -> left -> back. Each leg 100 steps.
const path = [
  { steps: 100, x: 20, y: 24, z: 20 },   // down to floor
  { steps: 100, x: 30, y: 24, z: 20 },   // along floor
  { steps: 200, x: 30, y:  4, z: 20 },   // up THROUGH interface
  { steps: 100, x: 20, y:  4, z: 20 },   // along top
  { steps: 200, x: 20, y: 18, z: 20 },   // back down through interface
];

let totalStep = 0;
let bad = false;
for (const leg of path) {
  solid.dragTargetX = leg.x;
  solid.dragTargetY = leg.y;
  solid.dragTargetZ = leg.z;
  for (let s = 0; s < leg.steps; s++) {
    sim.step(solid);
    solid.absorbDeadCells(sim);
    solid.step(1.0, sim.gravity);
    totalStep++;

    if (!Number.isFinite(solid.cy) || !Number.isFinite(solid.vy)) {
      console.log(`BLEW UP at step ${totalStep} target=(${leg.x},${leg.y},${leg.z})`);
      bad = true;
      break;
    }

    let maxU = 0;
    for (let i = 0; i < sim.n; i++) {
      const m = Math.hypot(sim.ux[i], sim.uy[i], sim.uz[i]);
      if (m > maxU) maxU = m;
    }
    if (maxU > 1.0) {
      console.log(`HIGH MAX U at step ${totalStep}: maxU=${maxU}`);
      bad = true;
      break;
    }
  }
  if (bad) break;
  const spd = Math.hypot(solid.vx, solid.vy, solid.vz);
  console.log(`step ${totalStep}: cy=${solid.cy.toFixed(2)} cx=${solid.cx.toFixed(2)} cz=${solid.cz.toFixed(2)} |v|=${spd.toFixed(4)} target=(${leg.x},${leg.y},${leg.z})`);
}
if (!bad) console.log('STABLE: dragged through interface and back without blowing up.');
