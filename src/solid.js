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

    // Density relative to a unit reference density.
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
  absorbDeadCells(sim) {
    const { nx, ny, nz, tag, tagPrev, ux, uy, uz, rho } = sim;
    for (let z = 1; z < nz - 1; z++) {
      for (let y = 1; y < ny - 1; y++) {
        for (let x = 1; x < nx - 1; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== 1 || tagPrev[i] !== 0) continue;  // SOLID && was FLUID
          const r = rho[i];
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

    // Mouse-spring force (critically damped).
    if (this.dragging) {
      const k  = 0.08 * this.mass;
      const cD = 2.0 * Math.sqrt(k * this.mass);
      this.Fx += k * (this.dragTargetX - this.cx) - cD * this.vx;
      this.Fy += k * (this.dragTargetY - this.cy) - cD * this.vy;
      this.Fz += k * (this.dragTargetZ - this.cz) - cD * this.vz;
    }

    // Linear + angular update.
    this.vx += dt * this.Fx / this.mass;
    this.vy += dt * this.Fy / this.mass;
    this.vz += dt * this.Fz / this.mass;
    this.wx += dt * this.Tx / this.I;
    this.wy += dt * this.Ty / this.I;
    this.wz += dt * this.Tz / this.I;

    // Mild angular damping (no real shear stress model in 2nd-order LBM).
    const omegaDamp = 0.985;
    this.wx *= omegaDamp; this.wy *= omegaDamp; this.wz *= omegaDamp;

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
