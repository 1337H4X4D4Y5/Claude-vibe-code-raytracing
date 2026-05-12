// 3D kinetic two-phase flow with fluid-solid coupling.
//
// Reimplementation of Wei Li & Mathieu Desbrun, SIGGRAPH 2023:
// "Fluid-Solid Coupling in Kinetic Two-Phase Flow Simulation".
//
// Architecture mirrors the paper:
//   * D3Q27 BGK distribution f for velocity / pressure
//   * D3Q7  Allen-Cahn distribution h for the conservative phase field
//   * Forced collision for surface tension and gravity (Guo et al.)
//   * Halfway bounce-back at fluid/solid links with the Ladd moving-wall
//     correction; momentum exchange feeds the rigid body
//   * Dead cells: fluid -> solid sweep transfers their momentum to the body
//   * Fresh cells: solid -> fluid sweep are seeded with the body's velocity
//     and a phase value averaged from fluid neighbours
//
// We keep things pure JS/CPU with TypedArrays; volume rendering is GPU.

import {
  Q27, Q7,
  EX27, EY27, EZ27, W27, OPP27,
  EX7,  EY7,  EZ7,  W7,  OPP7,
  CS2_27, INV_CS2_27, INV_2CS4_27,
  CS2_7,  INV_CS2_7,
  TAG_FLUID, TAG_SOLID, TAG_WALL,
} from './lattices.js';

export class Sim {
  constructor(nx, ny, nz) {
    this.nx = nx; this.ny = ny; this.nz = nz;
    const n = nx * ny * nz;
    this.n = n;

    this.f  = new Float32Array(n * Q27);
    this.f2 = new Float32Array(n * Q27);
    this.h  = new Float32Array(n * Q7);
    this.h2 = new Float32Array(n * Q7);

    this.rho = new Float32Array(n);
    this.ux  = new Float32Array(n);
    this.uy  = new Float32Array(n);
    this.uz  = new Float32Array(n);
    this.phi = new Float32Array(n);
    this.phiPrev = new Float32Array(n);

    this.tag      = new Uint8Array(n);
    this.tagPrev  = new Uint8Array(n);

    // Per-cell solid velocity (only meaningful for SOLID cells).
    this.usx = new Float32Array(n);
    this.usy = new Float32Array(n);
    this.usz = new Float32Array(n);

    // Phase gradient buffer (rebuilt each step).
    this.gx = new Float32Array(n);
    this.gy = new Float32Array(n);
    this.gz = new Float32Array(n);
    this.lap = new Float32Array(n);

    // Per-step external body force field (eg. interactive stirring).
    this.bodyFx = new Float32Array(n);
    this.bodyFy = new Float32Array(n);
    this.bodyFz = new Float32Array(n);

    // Precomputed lattice neighbour offsets in linear cell index:
    //   sIdx = i + neighOffset27[k]   <=>   x' = x - EX[k], y' = y - EY[k], ...
    //   (Negative sign because pull-scheme source is i - e_k.)
    // We also precompute a scaled offset so the inner loop can read the
    // distribution directly:  f2[i27 + fOffset27[k]] === f2[sIdx*Q27 + k].
    const nxny = nx * ny;
    this.neighOffset27 = new Int32Array(Q27);
    this.fOffset27     = new Int32Array(Q27);
    for (let k = 0; k < Q27; k++) {
      const off = -(EZ27[k] * nxny + EY27[k] * nx + EX27[k]);
      this.neighOffset27[k] = off;
      this.fOffset27[k]     = off * Q27 + k;
    }
    this.neighOffset7 = new Int32Array(Q7);
    this.hOffset7     = new Int32Array(Q7);
    for (let k = 0; k < Q7; k++) {
      const off = -(EZ7[k] * nxny + EY7[k] * nx + EX7[k]);
      this.neighOffset7[k] = off;
      this.hOffset7[k]     = off * Q7 + k;
    }

    // Physics parameters (overwritten by main).
    this.tau     = 0.6;
    this.tauPhi  = 0.7;
    this.sigma   = 0.012;
    this.gravity = 0.0010;     // applied in +y direction (down)
    this.rhoL    = 1.0;
    this.rhoG    = 0.05;
    this.W       = 4.0;
    this.mobility = 0.02;
  }

  idx(x, y, z) { return ((z * this.ny) + y) * this.nx + x; }

  rhoFromPhi(p) { return this.rhoG + p * (this.rhoL - this.rhoG); }

  // ------------------------------------------------------------------ init
  // Seeds phi with the equilibrium tanh interface profile, not a sharp
  // 0/1 step. A sharp step has total variation that the Allen-Cahn
  // anti-diffusion has to redistribute over the first few hundred
  // steps; that relaxation generates spurious flow which couples
  // through surface tension into the hydro and ends up pushing the
  // body around for the entire simulation. Initialising at the
  // equilibrium profile (phi = 1/2 (1 + tanh(2(y-y_w)/W))) means the
  // interface starts in (near-)mechanical equilibrium and the body
  // can actually come to rest.
  initFlat(waterlineY) {
    const { nx, ny, nz, W } = this;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const phiCol = 0.5 * (1 + Math.tanh(2 * (y - waterlineY) / W));
        for (let x = 0; x < nx; x++) {
          const i = this.idx(x, y, z);
          this.phi[i]     = phiCol;
          this.phiPrev[i] = phiCol;
          this.rho[i]     = 1.0;
          this.ux[i] = this.uy[i] = this.uz[i] = 0;
          const onWall = x === 0 || x === nx - 1 ||
                         y === 0 || y === ny - 1 ||
                         z === 0 || z === nz - 1;
          this.tag[i] = onWall ? TAG_WALL : TAG_FLUID;
          this.tagPrev[i] = this.tag[i];
          for (let k = 0; k < Q27; k++) this.f[i * Q27 + k] = W27[k];
          for (let k = 0; k < Q7;  k++) this.h[i * Q7  + k] = W7[k] * phiCol;
        }
      }
    }
    this.seedHydrostatic();
  }

  // Sets rho_LBM and the hydro distributions to satisfy the LBM
  // hydrostatic equation with the current phi field:
  //     d rho_LBM / d y = (rho_phase - rho_ref) * g / cs^2
  // This is the Boussinesq form, matched to the body force in
  // collideHydro. It gives only ((rho_phase - rho_ref) / rho_phase) of
  // real Archimedes buoyancy (so a body needs slightly different
  // density to float/sink than real-world numbers), but the smaller
  // pressure gradient keeps the LBM well clear of its Mach limit and
  // the body actually settles instead of being driven by fluid
  // oscillations.
  seedHydrostatic() {
    const { nx, ny, nz, phi, rho, f } = this;
    const rhoRef = 0.5 * (this.rhoL + this.rhoG);
    const gOverCs2 = this.gravity * INV_CS2_27;
    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) {
        let r = 1.0;
        for (let y = 0; y < ny; y++) {
          const i = this.idx(x, y, z);
          rho[i] = r;
          const i27 = i * Q27;
          for (let k = 0; k < Q27; k++) f[i27 + k] = W27[k] * r;
          const rhoPhi = this.rhoG + phi[i] * (this.rhoL - this.rhoG);
          r += (rhoPhi - rhoRef) * gOverCs2;
        }
      }
    }
  }

  // -------------------------------------------------------- gradient pass
  computePhiGradients() {
    const { nx, ny, nz, phi, gx, gy, gz, lap } = this;
    for (let z = 0; z < nz; z++) {
      const zm = z > 0 ? z - 1 : z, zp = z < nz - 1 ? z + 1 : z;
      for (let y = 0; y < ny; y++) {
        const ym = y > 0 ? y - 1 : y, yp = y < ny - 1 ? y + 1 : y;
        for (let x = 0; x < nx; x++) {
          const xm = x > 0 ? x - 1 : x, xp = x < nx - 1 ? x + 1 : x;
          const i = ((z * ny) + y) * nx + x;
          const c = phi[i];
          const px = phi[((z * ny) + y) * nx + xp];
          const mx = phi[((z * ny) + y) * nx + xm];
          const py = phi[((z * ny) + yp) * nx + x];
          const my = phi[((z * ny) + ym) * nx + x];
          const pz = phi[((zp * ny) + y) * nx + x];
          const mz = phi[((zm * ny) + y) * nx + x];
          gx[i] = 0.5 * (px - mx);
          gy[i] = 0.5 * (py - my);
          gz[i] = 0.5 * (pz - mz);
          lap[i] = px + mx + py + my + pz + mz - 6 * c;
        }
      }
    }
  }

  // -------------------------------------------------------- step driver
  step(solid) {
    this.retagCells(solid);
    this.handleFreshCells(solid);
    this.computePhiGradients();
    this.collidePhase();
    this.streamPhase();
    this.collideHydro();
    this.streamHydroAndMacro(solid);
  }

  // -------------------------------------------------------- tagging
  // Also snapshots phi into phiPrev for the next step's fresh-cell fallback.
  retagCells(solid) {
    const { nx, ny, nz, tag, tagPrev, usx, usy, usz, phi, phiPrev } = this;
    tagPrev.set(tag);
    phiPrev.set(phi);
    const tmp = [0, 0, 0];
    const nxny = nx * ny;
    for (let z = 0; z < nz; z++) {
      const zOnWall = z === 0 || z === nz - 1;
      const zBase = z * nxny;
      for (let y = 0; y < ny; y++) {
        const yOnWall = y === 0 || y === ny - 1;
        const yzBase = zBase + y * nx;
        for (let x = 0; x < nx; x++) {
          const i = yzBase + x;
          if (zOnWall || yOnWall || x === 0 || x === nx - 1) {
            tag[i] = TAG_WALL;
            usx[i] = usy[i] = usz[i] = 0;
            continue;
          }
          if (solid && solid.contains(x, y, z)) {
            tag[i] = TAG_SOLID;
            solid.surfaceVelocity(x, y, z, tmp);
            usx[i] = tmp[0]; usy[i] = tmp[1]; usz[i] = tmp[2];
          } else {
            tag[i] = TAG_FLUID;
            usx[i] = usy[i] = usz[i] = 0;
          }
        }
      }
    }
  }

  // -------------------------------------------------------- fresh cells
  // Cells that switched solid->fluid this step are reseeded by averaging
  // phi, rho and velocity from fluid neighbours.
  //
  // Previously these cells were seeded with the *body's* surface
  // velocity, which broke momentum conservation: the body kept its full
  // velocity AND new fluid appeared in its wake carrying a copy of
  // that velocity. Over many steps the system gained spurious energy
  // and the ball would never settle. Averaging from fluid neighbours
  // is what real "uncovering" looks like (fluid rushes in from the
  // surroundings) and is conservative.
  handleFreshCells(solid) {
    const { nx, ny, nz, tag, tagPrev,
            ux, uy, uz, rho, phi, phiPrev, f, h } = this;
    for (let z = 1; z < nz - 1; z++) {
      for (let y = 1; y < ny - 1; y++) {
        for (let x = 1; x < nx - 1; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== TAG_FLUID || tagPrev[i] !== TAG_SOLID) continue;

          let psum = 0, rsum = 0, vxs = 0, vys = 0, vzs = 0, pcnt = 0;
          for (let dz = -1; dz <= 1; dz++) {
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (!dx && !dy && !dz) continue;
                const j = (((z + dz) * ny) + (y + dy)) * nx + (x + dx);
                if (tag[j] === TAG_FLUID) {
                  psum += phi[j];
                  rsum += rho[j];
                  vxs  += ux[j];
                  vys  += uy[j];
                  vzs  += uz[j];
                  pcnt++;
                }
              }
            }
          }
          const p = pcnt > 0 ? psum / pcnt : phiPrev[i];
          const r = pcnt > 0 ? rsum / pcnt : 1.0;
          const vx = pcnt > 0 ? vxs / pcnt : 0;
          const vy = pcnt > 0 ? vys / pcnt : 0;
          const vz = pcnt > 0 ? vzs / pcnt : 0;
          phi[i] = p; phiPrev[i] = p;
          rho[i] = r;
          ux[i] = vx; uy[i] = vy; uz[i] = vz;

          const u2 = vx * vx + vy * vy + vz * vz;
          for (let k = 0; k < Q27; k++) {
            const eu = EX27[k] * vx + EY27[k] * vy + EZ27[k] * vz;
            f[i * Q27 + k] = W27[k] * r *
              (1 + INV_CS2_27 * eu + INV_2CS4_27 * eu * eu - 1.5 * u2);
          }
          for (let k = 0; k < Q7; k++) {
            const eu = EX7[k] * vx + EY7[k] * vy + EZ7[k] * vz;
            h[i * Q7 + k] = W7[k] * p * (1 + INV_CS2_7 * eu);
          }
        }
      }
    }
  }

  // -------------------------------------------------------- phase collide
  // Allen-Cahn LBM (Liang-style). Equilibrium:
  //   h_i^eq = w_i phi (1 + e_i.u / cs^2)
  // Forcing term (interface sharpening). The macroscopic equation we
  // want is
  //   dt phi + div(phi u) = M_phi * lap(phi) - M_phi * div(4 phi(1-phi)/W n_hat)
  // For a Guo-style forcing to recover the second divergence the source
  // term's first moment must be  M_phi * 4 phi(1-phi)/W * n_hat, with an
  // additional (1 - 1/(2 tau_phi)) half-correction so the trapezoidal
  // streaming step doesn't double-count it. Previously I had neither
  // factor, so the anti-diffusion was ~60x too strong and the interface
  // fingered into nonphysical spikes.
  collidePhase() {
    const { nx, ny, nz, h, h2, phi, ux, uy, uz, tag,
            gx, gy, gz, W, tauPhi } = this;
    const invTau = 1 / tauPhi;
    const nxny = nx * ny;
    const Mphi = CS2_7 * (tauPhi - 0.5);
    const guoHalf = 1.0 - 0.5 * invTau;
    const sharpC = 4.0 * Mphi * guoHalf / W;
    const sharpFac = INV_CS2_7;

    for (let z = 1; z < nz - 1; z++) {
      const zBase = z * nxny;
      for (let y = 1; y < ny - 1; y++) {
        const yzBase = zBase + y * nx;
        for (let x = 1; x < nx - 1; x++) {
          const i = yzBase + x;
          if (tag[i] !== TAG_FLUID) continue;
          const i7 = i * Q7;

          const p = phi[i];
          const uxi = ux[i], uyi = uy[i], uzi = uz[i];
          const ggx = gx[i], ggy = gy[i], ggz = gz[i];
          const gmag = Math.sqrt(ggx * ggx + ggy * ggy + ggz * ggz) + 1e-12;
          const sharpen = sharpC * p * (1 - p);
          const Sx = sharpen * sharpFac * ggx / gmag;
          const Sy = sharpen * sharpFac * ggy / gmag;
          const Sz = sharpen * sharpFac * ggz / gmag;

          for (let k = 0; k < Q7; k++) {
            const ekx = EX7[k], eky = EY7[k], ekz = EZ7[k];
            const wk = W7[k];
            const eu = ekx * uxi + eky * uyi + ekz * uzi;
            const heq = wk * p * (1 + INV_CS2_7 * eu);
            const Fk  = wk * (ekx * Sx + eky * Sy + ekz * Sz);
            const hk = h[i7 + k];
            h2[i7 + k] = hk - invTau * (hk - heq) + Fk;
          }
        }
      }
    }
  }

  // -------------------------------------------------------- phase stream
  // Pull-scheme stream of the D3Q7 phase distribution. The phi moment is
  // NOT updated here -- we defer it to streamHydroAndMacro so that
  // collideHydro continues to see the phi value matching the precomputed
  // gradient/laplacian buffers (otherwise force is computed with new phi
  // and old grad/lap, which destabilises the surface tension term when a
  // solid impacts the interface).
  //
  // For each fluid cell i and direction k, the source cell is i - e_k.
  // Wall/solid sources bounce: h_k(i, t+1) <- h2_OPP[k](i).
  streamPhase() {
    const { nx, ny, nz, h, h2, tag, neighOffset7, hOffset7 } = this;
    const nxny = nx * ny;

    for (let z = 1; z < nz - 1; z++) {
      const zBase = z * nxny;
      for (let y = 1; y < ny - 1; y++) {
        const yzBase = zBase + y * nx;
        for (let x = 1; x < nx - 1; x++) {
          const i = yzBase + x;
          if (tag[i] !== TAG_FLUID) continue;
          const i7 = i * Q7;
          for (let k = 0; k < Q7; k++) {
            const sIdx = i + neighOffset7[k];
            h[i7 + k] = (tag[sIdx] === TAG_FLUID)
              ? h2[i7 + hOffset7[k]]
              : h2[i7 + OPP7[k]];
          }
        }
      }
    }
  }

  // -------------------------------------------------------- hydro collide
  // BGK collision with Guo forcing. Forces:
  //   F_grav  = (rho(phi) - rho_ref) * g_vec
  //   F_st    = mu(phi) * grad(phi),
  //     mu = 4 beta phi (phi-1)(2 phi - 1) - kappa * Lap(phi)
  // Coefficients beta, kappa picked to recover surface tension sigma at
  // interface thickness W (Liang et al.):
  //   beta  = 12 sigma / W
  //   kappa = 1.5 sigma * W
  collideHydro() {
    const { nx, ny, nz, f, f2, rho, ux, uy, uz, phi, tag,
            tau, sigma, gravity, rhoL, rhoG, W,
            gx, gy, gz, lap, bodyFx, bodyFy, bodyFz } = this;
    const invTau = 1 / tau;
    const rhoRef = 0.5 * (rhoL + rhoG);
    const beta  = 12 * sigma / W;
    const kappa = 1.5 * sigma * W;
    // Guo source term constant: S_k = A * w_k * (ekF * (1 + eu/cs^2) - uF)
    // where A = (1 - 0.5/tau) / cs^2.
    const A = (1 - 0.5 * invTau) * INV_CS2_27;
    const nxny = nx * ny;

    for (let z = 1; z < nz - 1; z++) {
      const zBase = z * nxny;
      for (let y = 1; y < ny - 1; y++) {
        const yzBase = zBase + y * nx;
        for (let x = 1; x < nx - 1; x++) {
          const i = yzBase + x;
          if (tag[i] !== TAG_FLUID) continue;
          const i27 = i * Q27;

          const r = rho[i];
          const p = phi[i];
          const mu = 4 * beta * p * (p - 1) * (2 * p - 1) - kappa * lap[i];
          const rhoPhi = rhoG + p * (rhoL - rhoG);
          const fx = bodyFx[i] + mu * gx[i];
          const fy = bodyFy[i] + mu * gy[i] + (rhoPhi - rhoRef) * gravity;
          const fz = bodyFz[i] + mu * gz[i];

          const halfR = 0.5 / r;
          const uxe = ux[i] + fx * halfR;
          const uye = uy[i] + fy * halfR;
          const uze = uz[i] + fz * halfR;
          const u2  = uxe * uxe + uye * uye + uze * uze;
          const oneMinusU2half = 1 - 1.5 * u2;
          const uF = uxe * fx + uye * fy + uze * fz;

          for (let k = 0; k < Q27; k++) {
            const ekx = EX27[k], eky = EY27[k], ekz = EZ27[k];
            const wk = W27[k];
            const eu = ekx * uxe + eky * uye + ekz * uze;
            const feq = wk * r * (oneMinusU2half + eu * (INV_CS2_27 + INV_2CS4_27 * eu));
            const ekF = ekx * fx + eky * fy + ekz * fz;
            const Sk  = A * wk * (ekF * (1 + INV_CS2_27 * eu) - uF);
            const fk = f[i27 + k];
            f2[i27 + k] = fk - invTau * (fk - feq) + Sk;
          }
        }
      }
    }
  }

  // -------------------------------------------------------- hydro stream + macro
  // Fused pull-scheme stream of the D3Q27 hydrodynamic distribution with
  // the macroscopic moment update.
  //
  // For each fluid cell i and direction k, the source is i - e_k:
  //   * source fluid  ->  f_k(i, t+1) = f2_k(source)
  //   * source solid/wall -> halfway bounce-back with Ladd moving-wall:
  //         f_k(i, t+1) = f2_OPP[k](i) + 2 w_k rho_old (e_k . u_w) / cs^2
  //     and momentum is fed to the body:
  //         dF = -(f2_OPP[k](i) + f_k(i, t+1)) * e_k
  // Macroscopic moments (rho, u) accumulate as we go.
  //
  // Optimisations:
  //   - pull scheme: writes f[i*27+k] sequentially, no fill(0)
  //   - non-fluid cells skipped entirely (no copy-loop)
  //   - hoisted nx*ny and i*Q27 invariants
  //   - simplified Guo source term: S_k = A * w_k * (e_k.F * (1+e_k.u/cs^2) - u.F)
  //     with A = (1 - 0.5/tau) / cs^2 (constant per step)
  streamHydroAndMacro(solid) {
    const { nx, ny, nz, f, f2, h, rho, ux, uy, uz, phi, tag,
            usx, usy, usz, sigma, gravity, rhoL, rhoG, W,
            gx, gy, gz, lap, bodyFx, bodyFy, bodyFz,
            neighOffset27, fOffset27 } = this;
    const nxny = nx * ny;
    const rhoRef = 0.5 * (rhoL + rhoG);
    const beta  = 12 * sigma / W;
    const kappa = 1.5 * sigma * W;
    const dRho = rhoL - rhoG;

    for (let z = 1; z < nz - 1; z++) {
      const zBase = z * nxny;
      for (let y = 1; y < ny - 1; y++) {
        const yzBase = zBase + y * nx;
        for (let x = 1; x < nx - 1; x++) {
          const i = yzBase + x;
          if (tag[i] !== TAG_FLUID) continue;
          const i27 = i * Q27;
          const rhoOld = rho[i];

          let r = 0, mx = 0, my = 0, mz = 0;

          for (let k = 0; k < Q27; k++) {
            const sIdx = i + neighOffset27[k];
            const sTag = tag[sIdx];
            const ekx = EX27[k], eky = EY27[k], ekz = EZ27[k];

            let fk;
            if (sTag === TAG_FLUID) {
              fk = f2[i27 + fOffset27[k]];
            } else {
              // Bounce-back. fopp = what we pushed toward the obstacle.
              const fopp = f2[i27 + OPP27[k]];
              let corr = 0;
              if (sTag === TAG_SOLID) {
                const eu = ekx * usx[sIdx] + eky * usy[sIdx] + ekz * usz[sIdx];
                corr = 2 * W27[k] * rhoOld * eu * INV_CS2_27;
              }
              fk = fopp + corr;
              if (sTag === TAG_SOLID && solid) {
                // Scale impulse on the body by local phase density to
                // recover something like the real mass ratio between
                // gas and water (Boussinesq LBM has rho_LBM ~= 1 in
                // both, which makes the body feel gas with too much
                // inertia).
                const phaseWeight = rhoG + phi[i] * dRho;
                const wm = (fopp + fk) * phaseWeight;
                const sx = x - ekx, sy = y - eky, sz = z - ekz;
                solid.applyImpulseAtCell(sx, sy, sz, -wm * ekx, -wm * eky, -wm * ekz);
              }
            }
            f[i27 + k] = fk;
            r  += fk;
            mx += ekx * fk;
            my += eky * fk;
            mz += ekz * fk;
          }

          const rNew = r > 1e-6 ? r : 1e-6;
          rho[i] = rNew;

          // Update phi from the freshly-streamed h moments. We do this
          // here rather than in streamPhase so that collideHydro saw the
          // pre-stream phi (consistent with the precomputed grad/lap).
          let pSum = 0;
          for (let k = 0; k < Q7; k++) pSum += h[i * Q7 + k];
          const p = pSum < 0 ? 0 : pSum > 1 ? 1 : pSum;
          phi[i] = p;

          const mu = 4 * beta * p * (p - 1) * (2 * p - 1) - kappa * lap[i];
          const rhoPhi = rhoG + p * (rhoL - rhoG);
          const fx = bodyFx[i] + mu * gx[i];
          const fy = bodyFy[i] + mu * gy[i] + (rhoPhi - rhoRef) * gravity;
          const fz = bodyFz[i] + mu * gz[i];
          const invR = 1 / rNew;
          let uxn = (mx + 0.5 * fx) * invR;
          let uyn = (my + 0.5 * fy) * invR;
          let uzn = (mz + 0.5 * fz) * invR;

          // Mach-limit safety clamp. BGK is unstable above ~0.3 cs but
          // we cap much lower because the voxelised sphere is a slightly
          // asymmetric bounce-back surface and gives the body a tiny net
          // force even at rest; without a tight cap that bias drives a
          // positive-feedback fluid circulation that ends up carrying
          // the ball around the box. UMAX = 0.12 still permits any
          // realistic gravity-driven flow at these grid sizes.
          const u2 = uxn * uxn + uyn * uyn + uzn * uzn;
          const UMAX = 0.12, UMAX2 = UMAX * UMAX;
          if (u2 > UMAX2) {
            const s = UMAX / Math.sqrt(u2);
            uxn *= s; uyn *= s; uzn *= s;
          }
          ux[i] = uxn; uy[i] = uyn; uz[i] = uzn;
        }
      }
    }

    bodyFx.fill(0); bodyFy.fill(0); bodyFz.fill(0);
  }

  // Pack phi into a Uint8Array for upload to a 3D RG texture along with a
  // solid mask in the second channel.
  packVolume(out, solidContains) {
    const { nx, ny, nz, phi, tag } = this;
    let o = 0;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          let p = phi[i];
          if (p < 0) p = 0; else if (p > 1) p = 1;
          out[o++] = (p * 255) | 0;
          out[o++] = tag[i] === TAG_SOLID ? 255 : 0;
        }
      }
    }
  }
}
