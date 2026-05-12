// Rigid sphere with single-axis (angular) momentum, exchanging momentum with
// the LBM via the per-link contributions accumulated by Sim.streamHydro.
//
// The sphere is the simplest body that still exercises the moving-wall
// bounce-back code; the same interface (contains, surfaceVelocity,
// applyImpulseAtCell) works for any rigid shape.

export class RigidSphere {
  constructor(nx, ny, nz, opts = {}) {
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.cx = opts.cx ?? nx * 0.5;
    this.cy = opts.cy ?? ny * 0.3;
    this.cz = opts.cz ?? nz * 0.5;
    this.r  = opts.r  ?? Math.min(nx, ny, nz) * 0.10;

    this.vx = 0; this.vy = 0; this.vz = 0;
    this.wx = 0; this.wy = 0; this.wz = 0;          // angular velocity

    // Density relative to the LBM reference density rho_ref =
    // 0.5*(rho_L + rho_G). Because the LBM body force is Boussinesq
    // (rho_phase - rho_ref) g, the *effective* sink/float threshold a
    // body sees is rho_ref, not real water 1.0. So density 0.3
    // floats clearly, 1.4 sinks clearly.
    this.density = opts.density ?? 1.4;

    // Mouse-spring (when user is dragging).
    this.dragging = false;
    this.dragTargetX = 0;
    this.dragTargetY = 0;
    this.dragTargetZ = 0;

    // Per-step accumulators for momentum exchange.
    this.Fx = 0; this.Fy = 0; this.Fz = 0;
    this.Tx = 0; this.Ty = 0; this.Tz = 0;

    this.recomputeMass();
  }

  recomputeMass() {
    const V = (4 / 3) * Math.PI * this.r * this.r * this.r;
    this.volume = V;
    this.mass = this.density * V;
    this.I = (2 / 5) * this.mass * this.r * this.r;
  }

  setRadius(r) { this.r = r; this.recomputeMass(); }
  setDensity(d) { this.density = d; this.recomputeMass(); }

  // ----------------------------------------------- tag / velocity queries
  contains(x, y, z) {
    const dx = x + 0.5 - this.cx;
    const dy = y + 0.5 - this.cy;
    const dz = z + 0.5 - this.cz;
    return dx * dx + dy * dy + dz * dz <= this.r * this.r;
  }

  surfaceVelocity(x, y, z, out) {
    // u_body + omega x r
    const rx = x + 0.5 - this.cx;
    const ry = y + 0.5 - this.cy;
    const rz = z + 0.5 - this.cz;
    out[0] = this.vx + (this.wy * rz - this.wz * ry);
    out[1] = this.vy + (this.wz * rx - this.wx * rz);
    out[2] = this.vz + (this.wx * ry - this.wy * rx);
    return out;
  }

  // Momentum-exchange contribution (pushed by the LBM solver per fluid->
  // solid link). px,py,pz is the linear momentum transferred to the body.
  applyImpulseAtCell(x, y, z, px, py, pz) {
    this.Fx += px;
    this.Fy += py;
    this.Fz += pz;
    const rx = x + 0.5 - this.cx;
    const ry = y + 0.5 - this.cy;
    const rz = z + 0.5 - this.cz;
    this.Tx += ry * pz - rz * py;
    this.Ty += rz * px - rx * pz;
    this.Tz += rx * py - ry * px;
  }

  // ----------------------------------------------- dead cell sweep
  // Cells that *just* became solid hand their fluid momentum to the body
  // and are tagged solid by retagCells. We pick them up by scanning
  // tag/tagPrev after step() and reading the previous ux/uy/uz from sim.
  // The momentum is scaled by the cell's phase density so the body
  // sweeps gas as light and water as heavy (matches the phase-weighted
  // bounce-back impulse in streamHydroAndMacro).
  absorbDeadCells(sim) {
    const { nx, ny, nz, tag, tagPrev, ux, uy, uz, rho, phi, rhoL, rhoG } = sim;
    const dRho = rhoL - rhoG;
    for (let z = 1; z < nz - 1; z++) {
      for (let y = 1; y < ny - 1; y++) {
        for (let x = 1; x < nx - 1; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== 1 || tagPrev[i] !== 0) continue;  // SOLID && was FLUID
          const phaseWeight = rhoG + phi[i] * dRho;
          const r = rho[i] * phaseWeight;
          this.applyImpulseAtCell(x, y, z, r * ux[i], r * uy[i], r * uz[i]);
        }
      }
    }
  }

  // ----------------------------------------------- integration
  step(dt, gravity) {
    // Body gravity in world units (matches the fluid lattice gravity).
    const bodyG = gravity;        // +y is down here
    this.Fy += this.mass * bodyG;

    // No explicit Archimedes correction here: in practice the LBM
    // already delivers ~ V * rho_phase * g of effective buoyancy on a
    // body close to the interface (the integrated bounce-back over
    // the voxelised surface plus Guo-forcing contributions add up to
    // close to real Archimedes, not the textbook Boussinesq
    // prediction of (rho_phase - rho_ref) V g). Adding an extra
    // rho_ref V g would double-count and turn dense balls into
    // floaters.

    // Mouse-spring force (critically damped). Stiffness deliberately
    // gentle: the body's surface velocity feeds into the bounce-back's
    // moving-wall correction (~ 2 w rho (e.u_w)/cs^2 per link), and
    // u_w much above ~0.1 cs blows the LBM up. Soft k + the velocity
    // cap below keeps user drags inside the stable envelope.
    if (this.dragging) {
      const k  = 0.004 * this.mass;
      const cD = 2.0 * Math.sqrt(k * this.mass);
      this.Fx += k * (this.dragTargetX - this.cx) - cD * this.vx;
      this.Fy += k * (this.dragTargetY - this.cy) - cD * this.vy;
      this.Fz += k * (this.dragTargetZ - this.cz) - cD * this.vz;
    }

    // Linear + angular update with an acceleration cap. Bounding the
    // PER-STEP momentum change to a few gravities prevents two things
    // at once:
    //  (a) the LBM near the interface delivering 40x its theoretical
    //      buoyancy from voxelisation-driven instabilities;
    //  (b) the drag spring or a fast mouse motion injecting body
    //      velocity well past the fluid-Mach-stable u_w ~ 0.1.
    // Real fluid dynamics rarely exceed ~3g instantaneous accel, so
    // clipping there is generous for everything physical and just
    // sheds the spurious LBM spikes.
    const accCap = 3.5 * Math.abs(gravity || 0.001);
    {
      const ax = this.Fx / this.mass;
      const ay = this.Fy / this.mass;
      const az = this.Fz / this.mass;
      const a2 = ax * ax + ay * ay + az * az;
      const c2 = accCap * accCap;
      const s = a2 > c2 ? accCap / Math.sqrt(a2) : 1.0;
      this.vx += dt * ax * s;
      this.vy += dt * ay * s;
      this.vz += dt * az * s;
    }
    this.wx += dt * this.Tx / this.I;
    this.wy += dt * this.Ty / this.I;
    this.wz += dt * this.Tz / this.I;

    // Body Mach cap, as a final safety so a long-lasting one-sided
    // force can't accumulate past LBM stability.
    const BODY_UMAX = 0.08;
    const sp2 = this.vx * this.vx + this.vy * this.vy + this.vz * this.vz;
    if (sp2 > BODY_UMAX * BODY_UMAX) {
      const s = BODY_UMAX / Math.sqrt(sp2);
      this.vx *= s; this.vy *= s; this.vz *= s;
    }

    // Damping. The voxelised sphere is a non-spherical bounce-back
    // surface so even an at-rest fluid gives the body a tiny biased
    // force from imperfect symmetry; FP32 LBM drift adds its own
    // slow background. Without explicit damping these compound and the
    // ball rattles around forever.
    //
    // 8 % linear / 4 % angular damping per step puts terminal velocity
    // around g/0.08 ~ 0.01 cells per step, which is plenty to feel
    // gravity-driven motion but slow enough that compounding numerical
    // bias dies out on its own.
    const linDamp = 0.92;
    const omegaDamp = 0.96;
    this.vx *= linDamp; this.vy *= linDamp; this.vz *= linDamp;
    this.wx *= omegaDamp; this.wy *= omegaDamp; this.wz *= omegaDamp;

    // Snap to rest below the noise floor (~1 mcell/step) so the ball
    // visually settles instead of jittering. Threshold is well below
    // any gravity-driven terminal velocity above so real dynamics
    // still come through.
    const SNAP = 0.0008;
    if (Math.abs(this.vx) < SNAP) this.vx = 0;
    if (Math.abs(this.vy) < SNAP) this.vy = 0;
    if (Math.abs(this.vz) < SNAP) this.vz = 0;
    if (Math.abs(this.wx) < SNAP) this.wx = 0;
    if (Math.abs(this.wy) < SNAP) this.wy = 0;
    if (Math.abs(this.wz) < SNAP) this.wz = 0;

    this.cx += dt * this.vx;
    this.cy += dt * this.vy;
    this.cz += dt * this.vz;

    // Clamp inside the domain.
    const m = this.r + 1.5;
    if (this.cx < m)               { this.cx = m;               this.vx = Math.max(0, this.vx); }
    if (this.cx > this.nx - 1 - m) { this.cx = this.nx - 1 - m; this.vx = Math.min(0, this.vx); }
    if (this.cy < m)               { this.cy = m;               this.vy = Math.max(0, this.vy); }
    if (this.cy > this.ny - 1 - m) { this.cy = this.ny - 1 - m; this.vy = Math.min(0, this.vy); }
    if (this.cz < m)               { this.cz = m;               this.vz = Math.max(0, this.vz); }
    if (this.cz > this.nz - 1 - m) { this.cz = this.nz - 1 - m; this.vz = Math.min(0, this.vz); }

    // Reset accumulators for next step.
    this.Fx = this.Fy = this.Fz = 0;
    this.Tx = this.Ty = this.Tz = 0;
  }
}
