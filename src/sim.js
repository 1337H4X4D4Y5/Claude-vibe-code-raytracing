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
  initFlat(waterlineY) {
    const { nx, ny, nz } = this;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = this.idx(x, y, z);
          const phi = y > waterlineY ? 1.0 : 0.0;  // +y is down; bottom = liquid
          this.phi[i]     = phi;
          this.phiPrev[i] = phi;
          this.rho[i]     = 1.0;
          this.ux[i] = this.uy[i] = this.uz[i] = 0;
          const onWall = x === 0 || x === nx - 1 ||
                         y === 0 || y === ny - 1 ||
                         z === 0 || z === nz - 1;
          this.tag[i] = onWall ? TAG_WALL : TAG_FLUID;
          this.tagPrev[i] = this.tag[i];
          for (let k = 0; k < Q27; k++) this.f[i * Q27 + k] = W27[k] * 1.0;
          for (let k = 0; k < Q7;  k++) this.h[i * Q7  + k] = W7[k]  * phi;
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
    this.streamHydro(solid);
    this.computeMacro();
  }

  // -------------------------------------------------------- tagging
  retagCells(solid) {
    const { nx, ny, nz, tag, tagPrev, usx, usy, usz } = this;
    tagPrev.set(tag);
    const tmp = [0, 0, 0];
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (x === 0 || x === nx - 1 ||
              y === 0 || y === ny - 1 ||
              z === 0 || z === nz - 1) {
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
  // Cells that switched solid->fluid this step are reseeded with
  // equilibrium populations using the body's local velocity, and their
  // phase is averaged from fluid neighbours.
  handleFreshCells(solid) {
    const { nx, ny, nz, tag, tagPrev,
            ux, uy, uz, rho, phi, phiPrev, f, h } = this;
    const tmp = [0, 0, 0];
    for (let z = 1; z < nz - 1; z++) {
      for (let y = 1; y < ny - 1; y++) {
        for (let x = 1; x < nx - 1; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== TAG_FLUID || tagPrev[i] !== TAG_SOLID) continue;
          if (solid) solid.surfaceVelocity(x, y, z, tmp);
          else { tmp[0] = tmp[1] = tmp[2] = 0; }
          const vx = tmp[0], vy = tmp[1], vz = tmp[2];

          let psum = 0, pcnt = 0;
          for (let dz = -1; dz <= 1; dz++) {
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (!dx && !dy && !dz) continue;
                const j = (((z + dz) * ny) + (y + dy)) * nx + (x + dx);
                if (tag[j] === TAG_FLUID) { psum += phi[j]; pcnt++; }
              }
            }
          }
          const p = pcnt > 0 ? psum / pcnt : phiPrev[i];
          phi[i] = p; phiPrev[i] = p;
          rho[i] = 1.0;
          ux[i] = vx; uy[i] = vy; uz[i] = vz;

          const u2 = vx * vx + vy * vy + vz * vz;
          for (let k = 0; k < Q27; k++) {
            const eu = EX27[k] * vx + EY27[k] * vy + EZ27[k] * vz;
            f[i * Q27 + k] = W27[k] * 1.0 *
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
  // Forcing term (interface sharpening):
  //   F_i = w_i (e_i . [4 phi (1-phi)/W * n_hat]) / cs^2
  collidePhase() {
    const { nx, ny, nz, h, h2, phi, ux, uy, uz, tag,
            gx, gy, gz, W, tauPhi } = this;
    const invTau = 1 / tauPhi;
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== TAG_FLUID) {
            for (let k = 0; k < Q7; k++) h2[i * Q7 + k] = h[i * Q7 + k];
            continue;
          }
          const p = phi[i];
          const uxi = ux[i], uyi = uy[i], uzi = uz[i];
          const gmag = Math.hypot(gx[i], gy[i], gz[i]) + 1e-12;
          const nx_ = gx[i] / gmag;
          const ny_ = gy[i] / gmag;
          const nz_ = gz[i] / gmag;
          const sharpen = 4.0 * p * (1.0 - p) / W;

          for (let k = 0; k < Q7; k++) {
            const eu = EX7[k] * uxi + EY7[k] * uyi + EZ7[k] * uzi;
            const heq = W7[k] * p * (1 + INV_CS2_7 * eu);
            const en  = EX7[k] * nx_ + EY7[k] * ny_ + EZ7[k] * nz_;
            const Fk  = W7[k] * sharpen * en * INV_CS2_7;
            h2[i * Q7 + k] = h[i * Q7 + k] - invTau * (h[i * Q7 + k] - heq) + Fk;
          }
        }
      }
    }
  }

  // -------------------------------------------------------- phase stream
  // Only fluid cells stream; walls and solids would otherwise leak phase
  // mass into adjacent fluid (their h was seeded by initFlat). At solid/
  // wall neighbours we bounce the distribution back into the fluid cell
  // (zero flux for phi across the rigid surface).
  streamPhase() {
    const { nx, ny, nz, h, h2, tag } = this;
    // Preserve non-fluid h values; only zero out fluid cells.
    for (let i = 0; i < this.n; i++) {
      if (tag[i] === TAG_FLUID) {
        for (let k = 0; k < Q7; k++) h[i * Q7 + k] = 0;
      }
    }
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== TAG_FLUID) continue;
          for (let k = 0; k < Q7; k++) {
            const xn = x + EX7[k], yn = y + EY7[k], zn = z + EZ7[k];
            const oob = xn < 0 || xn >= nx || yn < 0 || yn >= ny || zn < 0 || zn >= nz;
            if (oob) {
              h[i * Q7 + OPP7[k]] += h2[i * Q7 + k];
              continue;
            }
            const j = ((zn * ny) + yn) * nx + xn;
            if (tag[j] === TAG_SOLID || tag[j] === TAG_WALL) {
              h[i * Q7 + OPP7[k]] += h2[i * Q7 + k];
            } else {
              h[j * Q7 + k] += h2[i * Q7 + k];
            }
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

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== TAG_FLUID) {
            for (let k = 0; k < Q27; k++) f2[i * Q27 + k] = f[i * Q27 + k];
            continue;
          }
          const r = rho[i];
          const p = phi[i];
          const mu = 4 * beta * p * (p - 1) * (2 * p - 1) - kappa * lap[i];
          const rhoPhi = rhoG + p * (rhoL - rhoG);

          const fx = bodyFx[i] + mu * gx[i];
          const fy = bodyFy[i] + mu * gy[i] + (rhoPhi - rhoRef) * gravity;
          const fz = bodyFz[i] + mu * gz[i];

          const uxe = ux[i] + 0.5 * fx / r;
          const uye = uy[i] + 0.5 * fy / r;
          const uze = uz[i] + 0.5 * fz / r;
          const u2  = uxe * uxe + uye * uye + uze * uze;

          for (let k = 0; k < Q27; k++) {
            const eu = EX27[k] * uxe + EY27[k] * uye + EZ27[k] * uze;
            const feq = W27[k] * r * (1 + INV_CS2_27 * eu
                                        + INV_2CS4_27 * eu * eu
                                        - 1.5 * u2);
            const tx = INV_CS2_27 * (EX27[k] - uxe) + INV_CS2_27 * INV_CS2_27 * eu * EX27[k];
            const ty = INV_CS2_27 * (EY27[k] - uye) + INV_CS2_27 * INV_CS2_27 * eu * EY27[k];
            const tz = INV_CS2_27 * (EZ27[k] - uze) + INV_CS2_27 * INV_CS2_27 * eu * EZ27[k];
            const Sk = (1 - 0.5 * invTau) * W27[k] * (tx * fx + ty * fy + tz * fz);
            f2[i * Q27 + k] = f[i * Q27 + k] - invTau * (f[i * Q27 + k] - feq) + Sk;
          }
        }
      }
    }
  }

  // -------------------------------------------------------- hydro stream
  // Halfway bounce-back at solid/wall links with Ladd's moving-wall
  // correction. Momentum exchange (force on the body) is accumulated via
  // (f*_k + f_{-k}) e_k summed over fluid->solid links.
  streamHydro(solid) {
    const { nx, ny, nz, f, f2, tag, usx, usy, usz, rho } = this;
    f.fill(0);
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          if (tag[i] !== TAG_FLUID) {
            for (let k = 0; k < Q27; k++) f[i * Q27 + k] = f2[i * Q27 + k];
            continue;
          }
          for (let k = 0; k < Q27; k++) {
            const xn = x + EX27[k], yn = y + EY27[k], zn = z + EZ27[k];
            const fk = f2[i * Q27 + k];
            const oob = xn < 0 || xn >= nx || yn < 0 || yn >= ny || zn < 0 || zn >= nz;
            let solidLink = oob;
            let uxs = 0, uys = 0, uzs = 0;
            let j = -1;
            if (!oob) {
              j = ((zn * ny) + yn) * nx + xn;
              if (tag[j] === TAG_SOLID) {
                solidLink = true;
                uxs = usx[j]; uys = usy[j]; uzs = usz[j];
              } else if (tag[j] === TAG_WALL) {
                solidLink = true;
              }
            }
            if (solidLink) {
              const ko = OPP27[k];
              const eu = EX27[k] * uxs + EY27[k] * uys + EZ27[k] * uzs;
              const corr = 2 * W27[k] * rho[i] * eu * INV_CS2_27;
              const bounced = fk - corr;
              f[i * Q27 + ko] += bounced;
              if (solid && !oob && tag[j] === TAG_SOLID) {
                const m = fk + bounced;
                solid.applyImpulseAtCell(xn, yn, zn,
                                          m * EX27[k], m * EY27[k], m * EZ27[k]);
              }
            } else {
              f[j * Q27 + k] += fk;
            }
          }
        }
      }
    }
  }

  // -------------------------------------------------------- macro update
  computeMacro() {
    const { nx, ny, nz, f, h, rho, ux, uy, uz, phi, phiPrev, tag,
            sigma, gravity, rhoL, rhoG, W, gx, gy, gz, lap,
            bodyFx, bodyFy, bodyFz } = this;
    const rhoRef = 0.5 * (rhoL + rhoG);
    const beta  = 12 * sigma / W;
    const kappa = 1.5 * sigma * W;

    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = ((z * ny) + y) * nx + x;
          phiPrev[i] = phi[i];
          if (tag[i] !== TAG_FLUID) continue;

          let r = 0, mx = 0, my = 0, mz = 0;
          for (let k = 0; k < Q27; k++) {
            const fk = f[i * Q27 + k];
            r += fk;
            mx += EX27[k] * fk;
            my += EY27[k] * fk;
            mz += EZ27[k] * fk;
          }
          rho[i] = r > 1e-6 ? r : 1e-6;

          let p = 0;
          for (let k = 0; k < Q7; k++) p += h[i * Q7 + k];
          if (p < 0) p = 0; else if (p > 1) p = 1;
          phi[i] = p;

          const mu = 4 * beta * p * (p - 1) * (2 * p - 1) - kappa * lap[i];
          const rhoPhi = rhoG + p * (rhoL - rhoG);
          const fx = bodyFx[i] + mu * gx[i];
          const fy = bodyFy[i] + mu * gy[i] + (rhoPhi - rhoRef) * gravity;
          const fz = bodyFz[i] + mu * gz[i];
          ux[i] = (mx + 0.5 * fx) / rho[i];
          uy[i] = (my + 0.5 * fy) / rho[i];
          uz[i] = (mz + 0.5 * fz) / rho[i];
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
