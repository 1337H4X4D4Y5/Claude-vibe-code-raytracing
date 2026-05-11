// Lattice direction tables shared by the 3D LBM solvers.
//
// D3Q27 (velocity / pressure) -- weights from He & Luo style enumeration:
//   rest:    8/27
//   face:    2/27   x 6
//   edge:    1/54   x 12
//   corner:  1/216  x 8
// All paired so OPP27[k] = k ^ 1 for k > 0.
//
// D3Q7 (phase field) -- weights:
//   rest:    1/4
//   face:    1/8    x 6
// Used by Li & Desbrun (SIGGRAPH 2023) for the phase distribution h_i. We
// keep the same pairing convention so the bounce-back code works for both.

export const Q27 = 27;
export const Q7  = 7;

export const EX27 = new Int8Array(Q27);
export const EY27 = new Int8Array(Q27);
export const EZ27 = new Int8Array(Q27);
export const W27  = new Float32Array(Q27);
export const OPP27 = new Int8Array(Q27);

export const EX7 = new Int8Array(Q7);
export const EY7 = new Int8Array(Q7);
export const EZ7 = new Int8Array(Q7);
export const W7  = new Float32Array(Q7);
export const OPP7 = new Int8Array(Q7);

(function build() {
  // D3Q27 -- ordered so directions come in opposite pairs.
  // index 0 is the rest particle.
  const dirs = [
    [ 0, 0, 0],                                                  // 0
    [ 1, 0, 0], [-1, 0, 0],                                      // 1, 2  (face)
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
    [ 1, 1, 0], [-1,-1, 0],                                      // 7..18 (edge)
    [ 1,-1, 0], [-1, 1, 0],
    [ 1, 0, 1], [-1, 0,-1],
    [ 1, 0,-1], [-1, 0, 1],
    [ 0, 1, 1], [ 0,-1,-1],
    [ 0, 1,-1], [ 0,-1, 1],
    [ 1, 1, 1], [-1,-1,-1],                                      // 19..26 (corner)
    [ 1, 1,-1], [-1,-1, 1],
    [ 1,-1, 1], [-1, 1,-1],
    [-1, 1, 1], [ 1,-1,-1],
  ];
  if (dirs.length !== Q27) throw new Error('D3Q27 setup');

  const weights = [
    8/27,
    2/27, 2/27, 2/27, 2/27, 2/27, 2/27,
    1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54,
    1/216,1/216,1/216,1/216,1/216,1/216,1/216,1/216,
  ];

  for (let i = 0; i < Q27; i++) {
    EX27[i] = dirs[i][0];
    EY27[i] = dirs[i][1];
    EZ27[i] = dirs[i][2];
    W27[i] = weights[i];
  }
  // Opposite pairs: 0 self, then k <-> k^1.
  for (let i = 0; i < Q27; i++) {
    if (i === 0) { OPP27[i] = 0; continue; }
    OPP27[i] = (i & 1) ? i + 1 : i - 1;
  }
  // Sanity check pairing.
  for (let i = 1; i < Q27; i++) {
    const j = OPP27[i];
    if (EX27[i] !== -EX27[j] || EY27[i] !== -EY27[j] || EZ27[i] !== -EZ27[j]) {
      throw new Error('OPP27 mismatch at ' + i);
    }
  }

  // D3Q7
  const d7 = [
    [ 0, 0, 0],
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
  ];
  const w7v = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8];
  for (let i = 0; i < Q7; i++) {
    EX7[i] = d7[i][0]; EY7[i] = d7[i][1]; EZ7[i] = d7[i][2];
    W7[i] = w7v[i];
  }
  OPP7[0] = 0;
  for (let i = 1; i < Q7; i++) OPP7[i] = (i & 1) ? i + 1 : i - 1;
})();

// Sound speeds and shorthands.
export const CS2_27 = 1/3;
export const INV_CS2_27 = 3.0;
export const INV_2CS4_27 = 4.5;
export const CS2_7 = 1/4;
export const INV_CS2_7 = 4.0;

// Cell tags.
export const TAG_FLUID = 0;
export const TAG_SOLID = 1;
export const TAG_WALL  = 2;
