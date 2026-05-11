// Per-phase breakdown of one step.
import { Sim } from '../src/sim.js';
import { RigidSphere } from '../src/solid.js';

const sim = new Sim(40, 28, 40);
sim.initFlat(14);
sim.tau = 0.7; sim.tauPhi = 0.7; sim.sigma = 0.012;
sim.gravity = 0.00135; sim.rhoL = 1.0; sim.rhoG = 0.1; sim.W = 4.0;
const solid = new RigidSphere(40, 28, 40, { cx: 20, cy: 6, cz: 20, r: 4, density: 1.6 });

for (let i = 0; i < 20; i++) {
  sim.step(solid); solid.absorbDeadCells(sim); solid.step(1.0, sim.gravity);
}

const REPS = 80;
const phases = ['retagCells', 'handleFreshCells', 'computePhiGradients',
                'collidePhase', 'streamPhase',
                'collideHydro', 'streamHydroAndMacro'];
const times = Object.fromEntries(phases.map(p => [p, 0]));

for (let r = 0; r < REPS; r++) {
  for (const p of phases) {
    const t0 = performance.now();
    if (p === 'retagCells') sim.retagCells(solid);
    else if (p === 'handleFreshCells') sim.handleFreshCells(solid);
    else if (p === 'streamHydroAndMacro') sim.streamHydroAndMacro(solid);
    else sim[p]();
    times[p] += performance.now() - t0;
  }
}

let total = 0;
for (const p of phases) total += times[p];
console.log('phase                  ms/step    %');
for (const p of phases) {
  const ms = times[p] / REPS;
  console.log(`  ${p.padEnd(20)} ${ms.toFixed(2).padStart(6)}  ${(100 * times[p] / total).toFixed(1).padStart(5)}%`);
}
console.log(`  TOTAL               ${(total / REPS).toFixed(2).padStart(6)}`);
