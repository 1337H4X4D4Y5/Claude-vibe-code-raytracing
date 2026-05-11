# Kinetic Two-Phase Flow with Fluid-Solid Coupling

A 3D, browser-based reimplementation of Wei Li & Mathieu Desbrun,
*"Fluid-Solid Coupling in Kinetic Two-Phase Flow Simulation"*
([SIGGRAPH 2023, ACM ToG 42(4)](https://www.geometry.caltech.edu/pubs/LD23.pdf)).

The simulator is pure JavaScript (no build step, no dependencies); the
volume renderer is a single WebGL2 fragment shader that raymarches the
live phase field.

![scene description: a solid sphere falling into a tank of water, rendered with raymarched volume rendering]

## Running

The page uses ES modules, so it has to be served over HTTP:

```bash
python3 -m http.server 8080
# then visit http://localhost:8080
```

The smoke tests run under Node:

```bash
node test/smoke.mjs
node test/profile.mjs
node test/sphere_profile.mjs
```

## What is implemented

| Paper element                                  | This implementation                                |
| ---------------------------------------------- | -------------------------------------------------- |
| D3Q27 velocity LBM with BGK collision          | `src/sim.js` — `collideHydro` / `streamHydro`      |
| Guo body-force forcing                         | velocity shift + correction term in `collideHydro` |
| D3Q7 Allen-Cahn phase-field LBM                | `src/sim.js` — `collidePhase` / `streamPhase`      |
| Interfacial anti-diffusion `4φ(1-φ)/W · n̂`     | forcing term in `collidePhase`                     |
| Surface-tension potential `μ ∇φ`               | `mu = 4β φ(φ−1)(2φ−1) − κ ∇²φ` in `collideHydro`   |
| Density-aware gravity `(ρ(φ)−ρ_ref) g`         | added to body force                                |
| Halfway bounce-back with moving-wall correction | `streamHydro` solid-link branch                    |
| Momentum-exchange force/torque on rigid body   | `applyImpulseAtCell` called per fluid→solid link   |
| **Fresh cells** (solid → fluid)                | `handleFreshCells`: phi averaged from fluid        |
|                                                | neighbours, populations seeded with body velocity  |
| **Dead cells** (fluid → solid)                 | `solid.absorbDeadCells`: hands the cell's momentum |
|                                                | to the body before tagging it solid                |
| Volumetric rendering of the phase field        | `src/render.js` — WebGL2 raymarcher                |

## What is simplified vs LD23

* The paper uses a **2× finer** phase-field grid than the velocity grid. We
  use the same grid for both. This costs some interface sharpness, which
  is most visible at small grid sizes.
* The paper uses 3D meshes for solids and rasterises them per frame. We
  use an analytic sphere (`RigidSphere`), so the bounce-back is voxel-
  precise without mesh sampling. The fluid-side interface
  (`Sim.streamHydro` and `Sim.handleFreshCells` / `Solid.absorbDeadCells`)
  is mesh-agnostic — point any class with `contains`,
  `surfaceVelocity`, and `applyImpulseAtCell` at the sim and it works.
* No multi-relaxation-time (MRT) operator; just BGK. This is enough at the
  grid sizes here but the paper recommends MRT for large Reynolds
  numbers.
* The rendering is a Beer's-law volume integration with a Fresnel-mixed
  surface shading at the phi = 0.5 iso-surface, not a full
  refraction-aware ray tracer.

## File layout

```
index.html         entry document
style.css          dark UI styling
src/
  lattices.js      D3Q27 / D3Q7 direction tables + lattice constants
  sim.js           the LBM solver (hydro + phase + boundary handling)
  solid.js         rigid sphere with momentum-exchange integration
  render.js        WebGL2 raymarcher + matrix helpers
  main.js          UI wiring, scene presets, main loop
test/
  smoke.mjs        runs sim for 80 steps and checks for NaN
  profile.mjs      vertical phi profile through time (no solid)
  sphere_profile.mjs  vertical phi profile with a falling solid
```

## Math at a glance

**Hydro distributions** (BGK + Guo forcing):

```
f_i*(x, t) = f_i(x, t) − (f_i(x, t) − f_i^eq(ρ, u + F/(2ρ))) / τ + S_i(F)
f_i(x + e_i, t + 1) = f_i*(x, t)              if link is fluid-fluid
f_-i(x, t + 1) = f_i*(x, t) − 2 w_i ρ (e_i · u_s) / cs²   if link is solid
```

**Phase distributions** (Allen-Cahn):

```
h_i^eq = w_i φ (1 + e_i · u / cs²)
F_i    = w_i (e_i · (4 φ(1-φ)/W · n̂)) / cs²
h_i*(x, t) = h_i(x, t) − (h_i − h_i^eq) / τ_φ + F_i
```

**Macros**:

```
ρ = Σ f_i                u = (Σ e_i f_i + F/2) / ρ
φ = Σ h_i                ρ(φ) = ρ_g + φ (ρ_l − ρ_g)
```

**Surface-tension potential** (Liang-style):

```
μ = 4 β φ (φ − 1) (2φ − 1) − κ ∇²φ
β = 12 σ / W           κ = 1.5 σ W
F_st = μ ∇φ
```

## License

MIT.
