// WebGL2 raymarched volume renderer for the phase field and solid mask.
//
// The simulation produces a 3D RG8 texture per frame: R = phi (liquid
// indicator), G = solid mask. A single fullscreen quad runs a fragment
// shader that intersects the camera ray with the unit cube, marches
// through the texture, accumulates alpha for the liquid, and shades
// gradient-based normals for both the liquid surface (refractive look)
// and the solid (Lambertian).

const VS_SRC = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FS_SRC = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vUv;
out vec4 fragColor;

uniform sampler3D uVol;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform vec3 uGridInv;     // 1.0 / (nx, ny, nz)
uniform vec3 uBoxMax;      // world-space extent of the tank
uniform vec3 uLight;       // normalized light direction (in world)
uniform float uTime;
uniform vec3 uSphereC;     // sphere center in world coords
uniform float uSphereR;    // sphere radius in world coords

vec2 intersectAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
  vec3 inv = 1.0 / rd;
  vec3 t1 = (bmin - ro) * inv;
  vec3 t2 = (bmax - ro) * inv;
  vec3 tmin = min(t1, t2);
  vec3 tmax = max(t1, t2);
  return vec2(max(max(tmin.x, tmin.y), tmin.z),
              min(min(tmax.x, tmax.y), tmax.z));
}

float intersectSphere(vec3 ro, vec3 rd, vec3 c, float r) {
  vec3 oc = ro - c;
  float b = dot(oc, rd);
  float cc = dot(oc, oc) - r * r;
  float d = b * b - cc;
  if (d < 0.0) return -1.0;
  d = sqrt(d);
  float t = -b - d;
  if (t < 0.0) t = -b + d;
  return t;
}

vec3 skyColor(vec3 rd) {
  // Soft gradient + sun. y axis = up in our view (camera up was set to
  // (0,-1,0) so that LBM +y is screen-down -- in world space "above the
  // tank" is rd.y < 0).
  float t = clamp(-rd.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 horizon = vec3(0.18, 0.22, 0.32);
  vec3 zenith  = vec3(0.02, 0.04, 0.10);
  vec3 sky = mix(horizon, zenith, t);
  float sun = pow(max(dot(rd, -uLight), 0.0), 64.0);
  return sky + vec3(1.0, 0.9, 0.7) * sun * 0.6;
}

// World coords are scaled so that the longest LBM axis maps to 1; here we
// convert world position to volume texture coordinates ([0,1]^3).
vec3 worldToUVW(vec3 wp) { return wp / uBoxMax; }

float samplePhi(vec3 wp) {
  return texture(uVol, worldToUVW(wp)).r;
}

vec3 gradPhi(vec3 wp) {
  vec3 e = uBoxMax * uGridInv;     // one cell in world units
  vec3 g;
  g.x = samplePhi(wp + vec3(e.x, 0, 0)) - samplePhi(wp - vec3(e.x, 0, 0));
  g.y = samplePhi(wp + vec3(0, e.y, 0)) - samplePhi(wp - vec3(0, e.y, 0));
  g.z = samplePhi(wp + vec3(0, 0, e.z)) - samplePhi(wp - vec3(0, 0, e.z));
  return g;
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 pNear = uInvViewProj * vec4(ndc, -1.0, 1.0);
  vec4 pFar  = uInvViewProj * vec4(ndc, +1.0, 1.0);
  vec3 wNear = pNear.xyz / pNear.w;
  vec3 wFar  = pFar.xyz  / pFar.w;
  vec3 rd = normalize(wFar - wNear);
  vec3 ro = uCamPos;

  // Tank bounds: rectangular box matching the grid's aspect ratio.
  vec2 tt = intersectAABB(ro, rd, vec3(0.0), uBoxMax);
  float tEnter = max(tt.x, 0.0);
  float tExit  = tt.y;
  if (tExit < tEnter) {
    fragColor = vec4(skyColor(rd), 1.0);
    return;
  }

  // Analytical sphere intersection -- gives crisp silhouette even when
  // we under-sample the volume.
  float tSphere = intersectSphere(ro, rd, uSphereC, uSphereR);

  // March.
  const int STEPS = 128;
  float tMaxVol = tExit;
  if (tSphere > 0.0 && tSphere < tMaxVol) tMaxVol = tSphere;
  float dt = (tMaxVol - tEnter) / float(STEPS);

  vec3 col = vec3(0.0);
  float alpha = 0.0;
  vec3 hitNormal = vec3(0.0);
  float hitDepth = -1.0;
  vec3 hitPos = vec3(0.0);
  vec3 hitPhi = vec3(0.0);
  float surfaceAlpha = 0.0;

  // Jitter to break banding.
  float jit = fract(sin(dot(vUv, vec2(12.9898, 78.233))) * 43758.5453);

  vec3 waterTint = vec3(0.18, 0.55, 0.78);
  float absorb = 2.6;  // Beer's law coefficient

  for (int i = 0; i < STEPS; i++) {
    float t = tEnter + (float(i) + jit) * dt;
    if (t > tMaxVol) break;
    vec3 wp = ro + t * rd;
    // Sample.
    float phi = samplePhi(wp);

    // Surface detection: cross 0.5 from below.
    if (phi > 0.5 && alpha < 0.001 && hitDepth < 0.0) {
      hitDepth = t;
      hitPos = wp;
      hitNormal = -normalize(gradPhi(wp) + vec3(1e-6));
    }

    // Accumulate liquid extinction inside the bulk.
    if (phi > 0.05) {
      float density = phi * absorb * dt;
      float trans = exp(-density);
      col += (1.0 - alpha) * (1.0 - trans) * waterTint * 0.35;
      alpha += (1.0 - alpha) * (1.0 - trans);
    }

    if (alpha > 0.985) break;
  }

  // If we hit the surface, shade it like a glossy water surface with
  // Fresnel + reflected sky and refracted murk.
  if (hitDepth > 0.0) {
    vec3 n = hitNormal;
    if (dot(n, rd) > 0.0) n = -n;
    float fres = pow(1.0 - max(dot(-rd, n), 0.0), 4.0);
    vec3 reflDir = reflect(rd, n);
    vec3 refrDir = refract(rd, n, 1.0 / 1.33);
    vec3 reflCol = skyColor(reflDir);
    float lambert = max(dot(n, -uLight), 0.0);
    vec3 deep = mix(waterTint * 0.4, waterTint * 0.9, lambert);
    vec3 surfCol = mix(deep, reflCol, fres * 0.85 + 0.15);
    // Specular highlight.
    vec3 h = normalize(-uLight + -rd);
    float spec = pow(max(dot(h, n), 0.0), 96.0);
    surfCol += vec3(1.0) * spec * 0.5;
    col = mix(surfCol, col, alpha * 0.4);
    alpha = max(alpha, 0.85);
  }

  // Solid sphere.
  if (tSphere > 0.0 && tSphere <= tExit) {
    vec3 sp = ro + tSphere * rd;
    vec3 n = normalize(sp - uSphereC);
    float lambert = max(dot(n, -uLight), 0.0);
    float rim = pow(1.0 - max(dot(-rd, n), 0.0), 3.0);
    vec3 base = vec3(0.95, 0.78, 0.35);
    vec3 sCol = base * (0.18 + 0.85 * lambert) + vec3(1.0, 0.85, 0.6) * rim * 0.3;
    // Specular.
    vec3 h = normalize(-uLight - rd);
    sCol += vec3(1.0) * pow(max(dot(h, n), 0.0), 32.0) * 0.6;
    col = col + (1.0 - alpha) * sCol;
    alpha = 1.0;
  }

  // Background.
  if (alpha < 1.0) {
    col += (1.0 - alpha) * skyColor(rd);
  }

  // Gentle tonemap.
  col = col / (col + vec3(1.0));
  col = pow(col, vec3(1.0 / 2.2));
  fragColor = vec4(col, 1.0);
}`;

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error('Shader compile failed:\n' + log + '\n--- SOURCE ---\n' + src);
  }
  return sh;
}

function linkProgram(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error('Program link failed: ' + log);
  }
  return p;
}

// -- minimal 4x4 matrix helpers (column-major as expected by uniformMatrix4fv)
function mat4Identity() {
  return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}
function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      o[i * 4 + j] =
        a[0 * 4 + j] * b[i * 4 + 0] +
        a[1 * 4 + j] * b[i * 4 + 1] +
        a[2 * 4 + j] * b[i * 4 + 2] +
        a[3 * 4 + j] * b[i * 4 + 3];
    }
  }
  return o;
}
function mat4Perspective(fovy, aspect, zn, zf) {
  const f = 1 / Math.tan(fovy * 0.5);
  const nf = 1 / (zn - zf);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (zf + zn) * nf, -1,
    0, 0, 2 * zf * zn * nf, 0
  ]);
}
function mat4LookAt(eye, target, up) {
  const z0 = eye[0] - target[0], z1 = eye[1] - target[1], z2 = eye[2] - target[2];
  let zl = Math.hypot(z0, z1, z2); if (zl < 1e-9) zl = 1;
  const zx = z0 / zl, zy = z1 / zl, zz = z2 / zl;
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz); if (xl < 1e-9) xl = 1;
  xx /= xl; xy /= xl; xz /= xl;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1
  ]);
}
function mat4Invert(m) {
  // From gl-matrix.
  const a00 = m[0],  a01 = m[1],  a02 = m[2],  a03 = m[3];
  const a10 = m[4],  a11 = m[5],  a12 = m[6],  a13 = m[7];
  const a20 = m[8],  a21 = m[9],  a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00*a11 - a01*a10, b01 = a00*a12 - a02*a10, b02 = a00*a13 - a03*a10;
  const b03 = a01*a12 - a02*a11, b04 = a01*a13 - a03*a11, b05 = a02*a13 - a03*a12;
  const b06 = a20*a31 - a21*a30, b07 = a20*a32 - a22*a30, b08 = a20*a33 - a23*a30;
  const b09 = a21*a32 - a22*a31, b10 = a21*a33 - a23*a31, b11 = a22*a33 - a23*a32;
  let det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (!det) return mat4Identity();
  det = 1.0 / det;
  return new Float32Array([
    ( a11*b11 - a12*b10 + a13*b09) * det,
    (-a01*b11 + a02*b10 - a03*b09) * det,
    ( a31*b05 - a32*b04 + a33*b03) * det,
    (-a21*b05 + a22*b04 - a23*b03) * det,
    (-a10*b11 + a12*b08 - a13*b07) * det,
    ( a00*b11 - a02*b08 + a03*b07) * det,
    (-a30*b05 + a32*b02 - a33*b01) * det,
    ( a20*b05 - a22*b02 + a23*b01) * det,
    ( a10*b10 - a11*b08 + a13*b06) * det,
    (-a00*b10 + a01*b08 - a03*b06) * det,
    ( a30*b04 - a31*b02 + a33*b00) * det,
    (-a20*b04 + a21*b02 - a23*b00) * det,
    (-a10*b09 + a11*b07 - a12*b06) * det,
    ( a00*b09 - a01*b07 + a02*b06) * det,
    (-a30*b03 + a31*b01 - a32*b00) * det,
    ( a20*b03 - a21*b01 + a22*b00) * det
  ]);
}

export class Renderer {
  constructor(canvas, nx, ny, nz) {
    this.canvas = canvas;
    this.nx = nx; this.ny = ny; this.nz = nz;
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
    if (!gl) throw new Error('WebGL2 is required for this demo.');
    this.gl = gl;

    const vs = compile(gl, gl.VERTEX_SHADER, VS_SRC);
    const fs = compile(gl, gl.FRAGMENT_SHADER, FS_SRC);
    this.prog = linkProgram(gl, vs, fs);

    this.uniforms = {
      uVol:         gl.getUniformLocation(this.prog, 'uVol'),
      uInvViewProj: gl.getUniformLocation(this.prog, 'uInvViewProj'),
      uCamPos:      gl.getUniformLocation(this.prog, 'uCamPos'),
      uGridInv:     gl.getUniformLocation(this.prog, 'uGridInv'),
      uBoxMax:      gl.getUniformLocation(this.prog, 'uBoxMax'),
      uLight:       gl.getUniformLocation(this.prog, 'uLight'),
      uTime:        gl.getUniformLocation(this.prog, 'uTime'),
      uSphereC:     gl.getUniformLocation(this.prog, 'uSphereC'),
      uSphereR:     gl.getUniformLocation(this.prog, 'uSphereR'),
    };

    // World extent: longest grid axis maps to 1; other axes are scaled
    // proportionally so the simulated sphere and the rendered sphere
    // agree.
    const maxN = Math.max(nx, ny, nz);
    this.boxMax = [nx / maxN, ny / maxN, nz / maxN];
    this.maxN = maxN;

    // Fullscreen quad.
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1,-1,  1,-1,  -1, 1,
       1,-1,  1, 1,  -1, 1
    ]), gl.STATIC_DRAW);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.vao = vao;

    // 3D texture.
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, tex);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG8, nx, ny, nz, 0,
                  gl.RG, gl.UNSIGNED_BYTE, null);
    this.tex = tex;
    this.volBuf = new Uint8Array(nx * ny * nz * 2);

    // Camera state (orbit around tank center, up = (0,-1,0) so that LBM
    // +y appears down).
    this.azimuth = Math.PI * 0.25;
    this.elev    = -0.25;     // slight downward tilt
    this.dist    = 2.4;
    this.target  = [this.boxMax[0] * 0.5,
                    this.boxMax[1] * 0.5,
                    this.boxMax[2] * 0.5];
    this.upWorld = [0, -1, 0];
  }

  uploadVolume(sim, solid) {
    sim.packVolume(this.volBuf, (x, y, z) => solid && solid.contains(x, y, z));
    this.uploadVolumeRaw(this.volBuf);
  }

  // For the worker pipeline: caller hands us a packed Uint8Array of size
  // nx*ny*nz*2 (R=phi*255, G=solid mask) and we upload directly.
  uploadVolumeRaw(buf) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_3D, this.tex);
    gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0,
                     this.nx, this.ny, this.nz,
                     gl.RG, gl.UNSIGNED_BYTE, buf);
  }

  resize() {
    const { canvas, gl } = this;
    const w = Math.round(canvas.clientWidth  * (window.devicePixelRatio || 1));
    const h = Math.round(canvas.clientHeight * (window.devicePixelRatio || 1));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);
    return canvas.width / canvas.height;
  }

  cameraEye() {
    const cx = this.target[0] + this.dist * Math.cos(this.elev) * Math.sin(this.azimuth);
    const cy = this.target[1] - this.dist * Math.sin(this.elev);  // y inverted
    const cz = this.target[2] + this.dist * Math.cos(this.elev) * Math.cos(this.azimuth);
    return [cx, cy, cz];
  }

  buildMatrices(aspect) {
    const eye = this.cameraEye();
    const proj = mat4Perspective(Math.PI / 3, aspect, 0.05, 20.0);
    const view = mat4LookAt(eye, this.target, this.upWorld);
    const vp   = mat4Mul(proj, view);
    const inv  = mat4Invert(vp);
    return { eye, inv };
  }

  // Returns world ray for current pointer for picking.
  pointerRay(nx, ny) {
    const aspect = this.canvas.width / this.canvas.height;
    const { eye, inv } = this.buildMatrices(aspect);
    const ndcX = nx * 2 - 1;
    const ndcY = (1 - ny) * 2 - 1;
    const near = [ndcX, ndcY, -1, 1];
    const far  = [ndcX, ndcY,  1, 1];
    const proj = (m, v) => {
      const o = [0, 0, 0, 0];
      for (let i = 0; i < 4; i++) {
        o[i] = m[i] * v[0] + m[4 + i] * v[1] + m[8 + i] * v[2] + m[12 + i] * v[3];
      }
      return [o[0] / o[3], o[1] / o[3], o[2] / o[3]];
    };
    const wn = proj(inv, near);
    const wf = proj(inv, far);
    const dx = wf[0] - wn[0], dy = wf[1] - wn[1], dz = wf[2] - wn[2];
    const l = Math.hypot(dx, dy, dz) || 1;
    return { origin: eye, dir: [dx / l, dy / l, dz / l] };
  }

  draw(sim, solid, t) {
    const gl = this.gl;
    const aspect = this.resize();
    const { eye, inv } = this.buildMatrices(aspect);

    gl.clearColor(0.02, 0.03, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.tex);
    gl.uniform1i(this.uniforms.uVol, 0);
    gl.uniformMatrix4fv(this.uniforms.uInvViewProj, false, inv);
    gl.uniform3fv(this.uniforms.uCamPos, eye);
    gl.uniform3f(this.uniforms.uGridInv, 1 / sim.nx, 1 / sim.ny, 1 / sim.nz);
    gl.uniform3fv(this.uniforms.uBoxMax, this.boxMax);
    // Light direction (from light TO surface). Up in world is -y (camera
    // up is (0,-1,0)) so a +y component means the light is above the scene.
    const L = [0.35, 0.75, 0.5];
    const Ll = Math.hypot(L[0], L[1], L[2]);
    gl.uniform3f(this.uniforms.uLight, L[0] / Ll, L[1] / Ll, L[2] / Ll);
    gl.uniform1f(this.uniforms.uTime, t);
    gl.uniform3f(this.uniforms.uSphereC,
                 solid.cx / this.maxN, solid.cy / this.maxN, solid.cz / this.maxN);
    gl.uniform1f(this.uniforms.uSphereR, solid.r / this.maxN);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
