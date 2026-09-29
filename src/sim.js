// ---------------------------------------------------------------------------
// Water balloon ballistics simulation
// Position-based "double density relaxation" fluid (Clavet et al. 2005),
// latex membrane with propagating tear, hydrodynamic-ram energy deposition,
// secondary spray + mist particles, ground wetting.  Units: metres, seconds.
// ---------------------------------------------------------------------------
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const RHO_W = 1000, RHO_AIR = 1.2;
const WET_RES = 256, WET_EXT = 1.6; // wetness map covers [-EXT, EXT]^2 on the ground
const FX_CAP = 14000;
const P_ATM = 101325;
const DT_REF = 4e-4;   // step size at which the relaxation stiffness was tuned
const V_JET = 18;     // m/s: faster fluid atomises into spray

class BalloonSim {
  constructor(params) { this.reset(params); }

  reset(params) {
    const p = (this.p = Object.assign({}, params));
    this.rng = mulberry32(p.seed || 1337);
    const R = (this.R = p.radius);
    this.ry = 0.95;                         // slight vertical squash of a hanging balloon
    this.C = [0, p.height, 0];
    const V = (4 / 3) * Math.PI * R * R * R * this.ry;
    const s = (this.s = Math.cbrt(V / p.count));
    this.h = 1.9 * s;
    // resolution-independent material: the relaxation displacement scales with h, so the stiffness
    // (sound speed) and the XSPH damping (kinematic viscosity ~ c h^2/dt) are normalised to a
    // reference spacing; the finer the particles, the more they need to reach the same physics
    const S_REF = 0.0086, rr = S_REF / s;
    this.resK = Math.min(2.2, rr * rr);
    this.resV = Math.min(5.0, Math.pow(rr, 6.5));
    this.rp = 0.45 * s;                     // collision radius vs ground

    // --- fill the balloon with a jittered lattice --------------------------------
    const pts = [];
    const inset = 1 - (0.62 * s) / R;
    const n = Math.ceil(R / s) + 1;
    for (let i = -n; i <= n; i++)
      for (let j = -n; j <= n; j++)
        for (let k = -n; k <= n; k++) {
          const x = (i + (j & 1) * 0.5) * s, y = j * s * 0.866, z = (k + (i & 1) * 0.5) * s;
          const q = Math.hypot(x / R, y / (R * this.ry), z / R);
          if (q < inset) pts.push(x, y, z);
        }
    const N = (this.N = pts.length / 3);
    this.pos = new Float32Array(N * 3);
    this.prev = new Float32Array(N * 3);
    this.vel = new Float32Array(N * 3);
    this.alive = new Uint8Array(N).fill(1);
    this.nb = new Uint16Array(N);
    this.nbSmooth = new Float32Array(N);
    this.groundT = new Float32Array(N);
    this.foam = new Float32Array(N);   // entrained micro-bubbles (0..1): cavity collapse, floor impact
    this.rho = new Float32Array(N);
    this.P = new Float32Array(N);
    this.PN = new Float32Array(N);
    this.dx = new Float32Array(N * 3);
    this.dv = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      this.pos[i * 3] = pts[i * 3] + this.C[0] + (this.rng() - 0.5) * 0.5 * s;
      this.pos[i * 3 + 1] = pts[i * 3 + 1] + this.C[1] + (this.rng() - 0.5) * 0.5 * s;
      this.pos[i * 3 + 2] = pts[i * 3 + 2] + this.C[2] + (this.rng() - 0.5) * 0.5 * s;
    }
    // neighbour structures
    this.TABLE = 1 << 15;
    this.cellStart = new Int32Array(this.TABLE + 1);
    this.cellIdx = new Int32Array(N);
    this.sorted = new Int32Array(N);
    this.pairCap = N * 45;
    this.pa = new Int32Array(this.pairCap);
    this.pb = new Int32Array(this.pairCap);
    this.pq = new Float32Array(this.pairCap);
    this.npairs = 0;

    // --- bullet --------------------------------------------------------------------
    const cal = p.caliber * 1e-3;
    this.bullet = {
      r: cal / 2,
      len: cal * 3.6,
      mass: 0.0097 * Math.pow(p.caliber / 7.62, 3),
      y: this.C[1] + p.offset * R * this.ry * 0.85,
      z: 0,
      x: this.C[0] - R - 0.3,
      v: p.speed,
      v0: p.speed,
      inWater: false,
    };
    const b = this.bullet;
    const dy = (b.y - this.C[1]) / (R * this.ry);
    const half = R * Math.sqrt(Math.max(0, 1 - dy * dy));
    this.xIn = this.C[0] - half;
    this.xOut = this.C[0] + half;
    this.tImpact = (this.xIn - b.x) / b.v;
    const len = (x) => { const d = [x - this.C[0], b.y - this.C[1], b.z - this.C[2]]; const l = Math.hypot(...d); return d.map((v) => v / l); };
    this.e1 = len(this.xIn);
    this.e2 = len(this.xOut);

    // --- membrane ----------------------------------------------------------------
    this.mem = { cavClosed: false, tIn: -1, tOut: -1, th1: 0, th2: 0, alpha: 4, gone: false, tGone: -1 };

    // --- secondary particles --------------------------------------------------------
    this.fx = {
      n: 0, head: 0,
      pos: new Float32Array(FX_CAP * 3), vel: new Float32Array(FX_CAP * 3),
      age: new Float32Array(FX_CAP), life: new Float32Array(FX_CAP),
      size: new Float32Array(FX_CAP), type: new Uint8Array(FX_CAP), alive: new Uint8Array(FX_CAP),
    };
    this.wet = new Float32Array(WET_RES * WET_RES);
    this.wetDirty = true;

    this.t = 0;
    this.vmaxBulk = 0;
    this.steps = 0;
    this.version = 0;

    // --- settle: find rest density, relax lattice inside the membrane --------------
    this.buildPairs();
    this.computeDensity();
    let acc = 0, cnt = 0;
    for (let i = 0; i < N; i++) if (this.nb[i] >= 16) { acc += this.rho[i]; cnt++; }
    this.rho0 = cnt ? acc / cnt : 3;
    for (let it = 0; it < 30; it++) {
      this.prev.set(this.pos);
      this.buildPairs();
      this.relax(1, DT_REF);
      this.constrain(DT_REF);
    }
    this.vel.fill(0);
    this.buildPairs();
    // re-measure rest density on the settled packing so the fluid starts in equilibrium
    this.relaxDensityOnly();
    let mx = 0; for (let i = 0; i < N; i++) mx = Math.max(mx, this.nb[i]);
    acc = 0; cnt = 0;
    for (let i = 0; i < N; i++) if (this.nb[i] >= 0.85 * mx) { acc += this.rho[i]; cnt++; }
    if (cnt) this.rho0 = acc / cnt;
    // pre-roll with gravity so the hanging balloon starts at rest in equilibrium
    const g0 = p.gravity, dt0 = DT_REF;
    for (let it = 0; it < 70; it++) {
      for (let i = 0; i < N; i++) this.vel[i * 3 + 1] -= g0 * dt0;
      this.prev.set(this.pos);
      for (let i = 0; i < N * 3; i++) this.pos[i] += this.vel[i] * dt0;
      this.buildPairs();
      this.relax(1, dt0);
      this.constrain(dt0);
      for (let i = 0; i < N * 3; i++) this.vel[i] = (this.pos[i] - this.prev[i]) / dt0 * 0.6;
    }
    this.vel.fill(0);
    this.buildPairs();
    for (let i = 0; i < N; i++) this.nbSmooth[i] = this.nb[i];
  }

  // ---- membrane geometry ------------------------------------------------------------
  angTo(dx, dy, dz, e) { return Math.acos(Math.max(-1, Math.min(1, dx * e[0] + dy * e[1] + dz * e[2]))); }
  memIntact(dx, dy, dz) { // unit direction from centre
    const m = this.mem;
    if (m.gone) return false;
    if (m.tIn >= 0 && this.angTo(dx, dy, dz, this.e1) < m.th1) return false;
    if (m.tOut >= 0 && this.angTo(dx, dy, dz, this.e2) < m.th2) return false;
    if (Math.acos(Math.max(-1, Math.min(1, dy))) > m.alpha) return false;
    return true;
  }

  // ---- neighbour search -------------------------------------------------------------------
  buildPairs() {
    const { N, pos, alive, TABLE, cellStart, cellIdx, sorted } = this;
    // Verlet neighbour list: candidates are gathered within h*(1+SKIN) and reused until some
    // particle has moved more than half the skin; each step only refreshes distances
    const SKIN = 0;   // measured: fast spray particles force a rebuild nearly every step, so lists are rebuilt each step
    if (SKIN > 0 && this.vpos && !this.pairsDirty) {
      const vp = this.vpos, lim2 = (0.5 * SKIN * this.h) ** 2;
      let ok = true;
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        const dx = pos[i * 3] - vp[i * 3], dy = pos[i * 3 + 1] - vp[i * 3 + 1], dz = pos[i * 3 + 2] - vp[i * 3 + 2];
        if (dx * dx + dy * dy + dz * dz > lim2) { ok = false; break; }
      }
      if (ok) { this.refreshPairs(); return; }
    }
    this.pairsDirty = false;
    if (SKIN > 0) { this.vpos = this.vpos && this.vpos.length === pos.length ? this.vpos : new Float32Array(pos.length); this.vpos.set(pos); }
    this.rebuilds = (this.rebuilds || 0) + 1;
    const inv = 1 / (this.h * (1 + SKIN));
    cellStart.fill(0);
    const hash = (x, y, z) => ((Math.imul(x, 92837111) ^ Math.imul(y, 689287499) ^ Math.imul(z, 283923481)) >>> 0) & (TABLE - 1);
    for (let i = 0; i < N; i++) {
      if (!alive[i]) { cellIdx[i] = -1; continue; }
      const c = hash(Math.floor(pos[i * 3] * inv), Math.floor(pos[i * 3 + 1] * inv), Math.floor(pos[i * 3 + 2] * inv));
      cellIdx[i] = c; cellStart[c + 1]++;
    }
    for (let c = 0; c < TABLE; c++) cellStart[c + 1] += cellStart[c];
    const fill = this._fill || (this._fill = new Int32Array(TABLE));
    fill.set(cellStart.subarray(0, TABLE));
    for (let i = 0; i < N; i++) if (cellIdx[i] >= 0) sorted[fill[cellIdx[i]]++] = i;

    const h2 = (this.h * (1 + SKIN)) ** 2, hq2 = this.h * this.h;
    let np = 0;
    const { pa, pb, pq, nb } = this;
    nb.fill(0);
    // half-shell stencil: own cell (j > i) + 13 forward cells, visited in spatially sorted order
    const OFF = this._off || (this._off = (() => {
      const o = [];
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) for (let oz = -1; oz <= 1; oz++)
        if (ox > 0 || (ox === 0 && oy > 0) || (ox === 0 && oy === 0 && oz > 0)) o.push(ox, oy, oz);
      return o;
    })());
    const hh = this.h, cap = this.pairCap;
    for (let si = 0; si < N; si++) {
      const i = sorted[si];
      if (si >= cellStart[TABLE]) break;
      const xi = pos[i * 3], yi = pos[i * 3 + 1], zi = pos[i * 3 + 2];
      const cx = Math.floor(xi * inv), cy = Math.floor(yi * inv), cz = Math.floor(zi * inv);
      const c0 = cellIdx[i];
      for (let k = cellStart[c0], e = cellStart[c0 + 1]; k < e; k++) {
        const j = sorted[k];
        if (j <= i) continue;
        const dx = pos[j * 3] - xi, dy = pos[j * 3 + 1] - yi, dz = pos[j * 3 + 2] - zi;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < h2 && np < cap) { pa[np] = i; pb[np] = j; pq[np] = Math.sqrt(d2) / hh; np++; if (d2 < hq2) { nb[i]++; nb[j]++; } }
      }
      for (let t = 0; t < 39; t += 3) {
        const c = hash(cx + OFF[t], cy + OFF[t + 1], cz + OFF[t + 2]);
        if (c === c0) continue;
        for (let k = cellStart[c], e = cellStart[c + 1]; k < e; k++) {
          const j = sorted[k];
          const dx = pos[j * 3] - xi, dy = pos[j * 3 + 1] - yi, dz = pos[j * 3 + 2] - zi;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < h2 && np < cap) { pa[np] = i; pb[np] = j; pq[np] = Math.sqrt(d2) / hh; np++; if (d2 < hq2) { nb[i]++; nb[j]++; } }
        }
      }
    }
    this.npairs = np;
  }

  refreshPairs() {
    const { pos, pa, pb, pq, nb, npairs } = this;
    const ih = 1 / this.h;
    nb.fill(0);
    for (let k = 0; k < npairs; k++) {
      const i = pa[k], j = pb[k];
      const dx = pos[j * 3] - pos[i * 3], dy = pos[j * 3 + 1] - pos[i * 3 + 1], dz = pos[j * 3 + 2] - pos[i * 3 + 2];
      const q = Math.sqrt(dx * dx + dy * dy + dz * dz) * ih;
      pq[k] = q;
      if (q < 1) { nb[i]++; nb[j]++; }
    }
  }

  computeDensity() {
    const { rho, npairs, pa, pb, pq } = this;
    rho.fill(1); // self contribution
    for (let k = 0; k < npairs; k++) {
      if (pq[k] >= 1) continue;
      const w = 1 - pq[k], w2 = w * w;
      rho[pa[k]] += w2; rho[pb[k]] += w2;
    }
  }

  relaxDensityOnly() {
    const { rho, pa, pb, pos } = this; const h = this.h;
    rho.fill(1);
    for (let k = 0; k < this.npairs; k++) {
      const i = pa[k], j = pb[k];
      const q = Math.hypot(pos[j * 3] - pos[i * 3], pos[j * 3 + 1] - pos[i * 3 + 1], pos[j * 3 + 2] - pos[i * 3 + 2]) / h;
      if (q < 1) { const w = 1 - q; rho[i] += w * w; rho[j] += w * w; }
    }
  }
  // ---- double density relaxation (Jacobi, symmetric) ----------------------------------
  relax(iters, dt) {
    const { pos, rho, P, PN, dx, pa, pb, alive, N } = this;
    // stiffness scales with dt^2 (Clavet): physical pressure, not per-step displacement
    const f = Math.min(1, (dt * dt) / (DT_REF * DT_REF)) * this.resK;
    const h = this.h, kP = (this.kP || 0.055) * f, kN = (this.kN || 0.02) * f, rho0 = this.rho0, tc = this.tc || 0.4;
    for (let it = 0; it < iters; it++) {
      rho.fill(1);
      const rn = this.PN; rn.fill(1);
      for (let k = 0; k < this.npairs; k++) {
        const i = pa[k], j = pb[k];
        const ddx = pos[j * 3] - pos[i * 3], ddy = pos[j * 3 + 1] - pos[i * 3 + 1], ddz = pos[j * 3 + 2] - pos[i * 3 + 2];
        const q = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / h;
        this.pq[k] = q;
        if (q >= 1) continue;
        const w = 1 - q, w2 = w * w;
        rho[i] += w2; rho[j] += w2; rn[i] += w2 * w; rn[j] += w2 * w;
      }
      for (let i = 0; i < N; i++) {
        let pr = kP * (rho[i] - rho0);
        if (pr < -tc * kP * rho0) pr = -tc * kP * rho0; // limited tension (cohesion / surface tension)
        P[i] = pr; PN[i] = kN * rn[i];
      }
      dx.fill(0);
      for (let k = 0; k < this.npairs; k++) {
        const q = this.pq[k];
        if (q >= 1 || q < 1e-6) continue;
        const i = pa[k], j = pb[k];
        const w = 1 - q;
        const D = 0.5 * h * ((P[i] + P[j]) * w + (PN[i] + PN[j]) * w * w) * 0.5;
        const inv = 1 / (q * h);
        const ux = (pos[j * 3] - pos[i * 3]) * inv, uy = (pos[j * 3 + 1] - pos[i * 3 + 1]) * inv, uz = (pos[j * 3 + 2] - pos[i * 3 + 2]) * inv;
        dx[i * 3] -= D * ux; dx[i * 3 + 1] -= D * uy; dx[i * 3 + 2] -= D * uz;
        dx[j * 3] += D * ux; dx[j * 3 + 1] += D * uy; dx[j * 3 + 2] += D * uz;
      }
      const lim = 0.3 * this.s;
      for (let i = 0; i < N; i++) {
        if (!alive[i]) continue;
        let a = dx[i * 3], b = dx[i * 3 + 1], c = dx[i * 3 + 2];
        const l = Math.hypot(a, b, c);
        if (l > lim) { const f = lim / l; a *= f; b *= f; c *= f; }
        pos[i * 3] += a; pos[i * 3 + 1] += b; pos[i * 3 + 2] += c;
      }
    }
  }

  constrain(dt) {
    const { pos, prev, alive, N, C, R } = this;
    const ry = this.ry, rp = this.rp;
    const shell = 1 - (0.55 * this.s) / R;
    const memActive = !this.mem.gone;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const o = i * 3;
      if (memActive) {
        const lx = (pos[o] - C[0]) / R, ly = (pos[o + 1] - C[1]) / (R * ry), lz = (pos[o + 2] - C[2]) / R;
        const q = Math.hypot(lx, ly, lz);
        if (q > shell) {
          const dxn = lx / q, dyn = ly / q, dzn = lz / q;
          // only the part of the ray from the centre that is still covered by rubber holds water
          if (q < 1.25 && this.memIntact(dxn, dyn, dzn)) {
            const f = shell / q;
            pos[o] = C[0] + lx * f * R; pos[o + 1] = C[1] + ly * f * R * ry; pos[o + 2] = C[2] + lz * f * R;
          }
        }
      }
      if (pos[o + 1] < rp) {
        // impact on the floor: the rim of the spreading sheet throws a crown of droplets
        const vy = (pos[o + 1] - prev[o + 1]) / dt;
        if (vy < -1.5) this.foam[i] = Math.min(0.6, this.foam[i] + (-vy - 1.5) * 0.12);
        if (vy < -1.2 && this.gc && this.rng() < 0.45) {
          let rx = pos[o] - this.gc[0], rz = pos[o + 2] - this.gc[1];
          const rl = Math.hypot(rx, rz) || 1; rx /= rl; rz /= rl;
          const a = -vy, jit = () => (this.rng() - 0.5) * 0.4;
          const hs = a * (1.0 + 1.1 * this.rng()), up = a * (0.35 + 0.75 * this.rng());
          this.emit(0, pos[o], rp + 0.002, pos[o + 2], (rx + jit()) * hs, up, (rz + jit()) * hs, 0.0007 + 0.0018 * this.rng(), 3);
        }
        // stagnation pressure turns the fall into radial spreading of the sheet
        if (vy < -0.3 && this.gc) {
          let rx = pos[o] - this.gc[0], rz = pos[o + 2] - this.gc[1];
          const rl = Math.hypot(rx, rz);
          if (rl > 1e-4) { const add = 0.3 * -vy * dt / rl; prev[o] -= rx * add; prev[o + 2] -= rz * add; }
        }
        pos[o + 1] = rp;
        // ground friction: damp tangential motion of this step
        const fr = Math.min(1, dt / 0.025);   // thin viscous boundary layer: the sheet still spreads
        pos[o] = pos[o] - (pos[o] - prev[o]) * fr;
        pos[o + 2] = pos[o + 2] - (pos[o + 2] - prev[o + 2]) * fr;
        if (prev[o + 1] < rp) prev[o + 1] = rp;
        // crown: the thickened rim of the fast-spreading sheet lifts off the floor
        if (this.gc && this.nb[i] < 15) {
          let rx = pos[o] - this.gc[0], rz = pos[o + 2] - this.gc[1];
          const rl = Math.hypot(rx, rz);
          if (rl > 0.03) {
            const vr = ((pos[o] - prev[o]) * rx + (pos[o + 2] - prev[o + 2]) * rz) / (rl * dt);
            if (vr > 1.0) prev[o + 1] = rp - 0.35 * (vr - 1.0) * (0.5 + this.rng()) * dt;
          }
        }
      }
    }
  }

  // ---- energy deposition from the projectile --------------------------------------------
  applyBullet(xa, xb) {
    const b = this.bullet, p = this.p;
    if (xb < this.xIn || xa > this.xOut) return;
    const x0 = Math.max(xa, this.xIn), x1 = Math.min(xb, this.xOut);
    if (x1 <= x0) return;
    // drag force per unit length = energy deposited per unit length
    const A = Math.PI * b.r * b.r;
    const Cd = 0.3 * p.energy;
    const F = 0.5 * RHO_W * Cd * A * b.v * b.v;
    const ac = Math.max(1.1 * this.s, 4 * b.r);
    const L = Math.log(Math.max(1.5, this.R / ac));
    const U = Math.sqrt(F / (Math.PI * RHO_W * ac * ac * L));
    const { pos, vel, prev, alive, N } = this;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const o = i * 3;
      const px = pos[o];
      if (px < x0 || px >= x1) continue;
      const ry = pos[o + 1] - b.y, rz = pos[o + 2] - b.z;
      const d = Math.hypot(ry, rz) + 1e-6;
      const dd = Math.max(d, ac);
      let vr = U * (ac / dd) * Math.exp(-0.35 * dd / ac);
      vr *= 0.85 + 0.3 * this.rng();
      const near = Math.exp(-d / ac);
      this.foam[i] = Math.max(this.foam[i], 0.7 * Math.exp(-(d * d) / (ac * ac)));
      // radial cavity flow + forward drag of fluid near the shot line
      vel[o + 1] += vr * ry / d;
      vel[o + 2] += vr * rz / d;
      vel[o] += vr * (0.55 * near + 0.08) * (0.7 + 0.6 * this.rng());
      // back-splash near the entry, forward jet near the exit
      const fromIn = (px - this.xIn) / (this.xOut - this.xIn);
      if (fromIn < 0.18) vel[o] -= vr * 0.9 * (1 - fromIn / 0.18) * near;
      if (fromIn > 0.8) vel[o] += vr * 0.8 * ((fromIn - 0.8) / 0.2) * near;
      const cap = 0.22 * b.v;
      const sp = Math.hypot(vel[o], vel[o + 1], vel[o + 2]);
      if (sp > cap) { const f = cap / sp; vel[o] *= f; vel[o + 1] *= f; vel[o + 2] *= f; }
    }
    // bullet decelerates
    b.v = Math.max(0.2 * b.v0, b.v - (F * (x1 - x0)) / (b.mass * b.v));
  }

  // ---- secondary particles ----------------------------------------------------------
  emit(type, x, y, z, vx, vy, vz, size, life) {
    const f = this.fx;
    const i = f.head;
    f.head = (f.head + 1) % FX_CAP;
    if (!f.alive[i]) f.n++;
    f.alive[i] = 1; f.type[i] = type;
    f.pos[i * 3] = x; f.pos[i * 3 + 1] = y; f.pos[i * 3 + 2] = z;
    f.vel[i * 3] = vx; f.vel[i * 3 + 1] = vy; f.vel[i * 3 + 2] = vz;
    f.age[i] = 0; f.life[i] = life; f.size[i] = size;
  }
  randCone(ax, ay, az, spread) {
    // random unit vector within a cone around (ax,ay,az)
    const r = this.rng;
    let ux = r() * 2 - 1, uy = r() * 2 - 1, uz = r() * 2 - 1;
    const dot = ux * ax + uy * ay + uz * az;
    ux -= dot * ax; uy -= dot * ay; uz -= dot * az;
    const ul = Math.hypot(ux, uy, uz) + 1e-9;
    const ang = spread * Math.sqrt(r());
    const c = Math.cos(ang), s = Math.sin(ang);
    return [ax * c + (ux / ul) * s, ay * c + (uy / ul) * s, az * c + (uz / ul) * s];
  }
  emitHole(dt, which) {
    const m = this.mem, p = this.p, b = this.bullet;
    const t0 = which === 0 ? m.tIn : m.tOut;
    if (t0 < 0) return;
    const age = this.t - t0;
    if (age > 0.004) return;
    const X = which === 0 ? this.xIn : this.xOut;
    const k = Math.exp(-age / 0.0012);
    const vb = b.v0 * Math.sqrt(p.energy);
    const baseRate = (which === 0 ? 0.9e6 : 1.8e6) * p.mist * Math.min(1.5, p.energy * b.v0 / 600); // particles per second of physical time
    let cnt = baseRate * k * dt;
    cnt = Math.floor(cnt) + (this.rng() < cnt % 1 ? 1 : 0);
    const ax = which === 0 ? -0.75 : 1, ay = 0, az = 0;
    const e = which === 0 ? this.e1 : this.e2;
    let dx = ax + e[0] * 0.5, dy = ay + e[1] * 0.5, dz = az + e[2] * 0.5;
    const l = Math.hypot(dx, dy, dz); dx /= l; dy /= l; dz /= l;
    for (let c = 0; c < cnt; c++) {
      const mist = this.rng() < 0.45;
      const spread = which === 0 ? 0.95 : 0.55;
      const d = this.randCone(dx, dy, dz, spread * (mist ? 1.25 : 1));
      const sp = vb * (which === 0 ? 0.035 : 0.075) * (0.25 + 0.75 * Math.pow(this.rng(), 0.6)) * (0.5 + 0.5 * k);
      const jx = (this.rng() - 0.5) * 0.01, jy = (this.rng() - 0.5) * 0.01, jz = (this.rng() - 0.5) * 0.01;
      if (mist) this.emit(1, X + jx, b.y + jy, b.z + jz, d[0] * sp, d[1] * sp, d[2] * sp, 0.004 + 0.004 * this.rng(), 0.35 + 0.6 * this.rng());
      else this.emit(0, X + jx, b.y + jy, b.z + jz, d[0] * sp, d[1] * sp, d[2] * sp, 0.0006 + 0.0016 * Math.pow(this.rng(), 2), 3);
    }
  }
  atomise(i, sp) {
    // a fluid parcel moving faster than the sheet can hold together shatters into drops + mist
    const o = i * 3, { pos, vel } = this;
    this.alive[i] = 0; this.pairsDirty = true;
    const r = this.rng;
    const nd = 3 + Math.floor(r() * 3);
    for (let k = 0; k < nd; k++) {
      const j = 0.18;
      this.emit(0, pos[o] + (r() - 0.5) * this.s, pos[o + 1] + (r() - 0.5) * this.s, pos[o + 2] + (r() - 0.5) * this.s,
        vel[o] * (1 + (r() - 0.5) * j), vel[o + 1] * (1 + (r() - 0.5) * j) + (r() - 0.5) * sp * 0.08, vel[o + 2] * (1 + (r() - 0.5) * j) + (r() - 0.5) * sp * 0.08,
        0.0008 + 0.0022 * Math.pow(r(), 1.5), 3);
    }
    if (this.p.mist > 0 && r() < 0.8 * this.p.mist)
      this.emit(1, pos[o], pos[o + 1], pos[o + 2], vel[o] * 0.8, vel[o + 1] * 0.8, vel[o + 2] * 0.8, 0.004 + 0.004 * r(), 0.4 + 0.6 * r());
  }
  updateFx(dt) {
    const f = this.fx, g = this.p.gravity;
    let n = 0;
    for (let i = 0; i < FX_CAP; i++) {
      if (!f.alive[i]) continue;
      const o = i * 3;
      f.age[i] += dt;
      if (f.type[i] === 0) {
        const vx = f.vel[o], vy = f.vel[o + 1], vz = f.vel[o + 2];
        const sp = Math.hypot(vx, vy, vz);
        const k = (3 * RHO_AIR * 0.47) / (8 * RHO_W * f.size[i]);
        const damp = 1 / (1 + k * sp * dt);
        f.vel[o] = vx * damp; f.vel[o + 1] = (vy - g * dt) * damp; f.vel[o + 2] = vz * damp;
      } else {
        const damp = Math.exp(-dt / 0.03);
        f.vel[o] *= damp; f.vel[o + 1] = f.vel[o + 1] * damp - 0.03 * g * dt; f.vel[o + 2] *= damp;
        f.size[i] += dt * 0.05;
      }
      f.pos[o] += f.vel[o] * dt; f.pos[o + 1] += f.vel[o + 1] * dt; f.pos[o + 2] += f.vel[o + 2] * dt;
      if (f.pos[o + 1] < 0) {
        if (f.type[i] === 0) { this.deposit(f.pos[o], f.pos[o + 2], f.size[i] * 6, 0.02); f.alive[i] = 0; continue; }
        f.pos[o + 1] = 0; f.vel[o + 1] = 0;
      }
      if (f.age[i] > f.life[i]) { f.alive[i] = 0; continue; }
      n++;
    }
    f.n = n;
  }

  deposit(x, z, radius, amount) {
    const cell = (2 * WET_EXT) / WET_RES;
    const cx = (x + WET_EXT) / cell, cz = (z + WET_EXT) / cell;
    const rr = Math.max(1, radius / cell);
    const i0 = Math.max(0, Math.floor(cx - rr * 2)), i1 = Math.min(WET_RES - 1, Math.ceil(cx + rr * 2));
    const j0 = Math.max(0, Math.floor(cz - rr * 2)), j1 = Math.min(WET_RES - 1, Math.ceil(cz + rr * 2));
    for (let j = j0; j <= j1; j++)
      for (let i = i0; i <= i1; i++) {
        const d2 = ((i + 0.5 - cx) ** 2 + (j + 0.5 - cz) ** 2) / (rr * rr);
        const w = Math.exp(-d2 * 1.5) * amount;
        this.wet[j * WET_RES + i] = Math.min(4, this.wet[j * WET_RES + i] + w);
      }
    this.wetDirty = true;
  }

  // ---- time stepping ------------------------------------------------------------------
  chooseDt() {
    let dt = (0.45 * this.s) / Math.max(this.vmaxBulk, 0.6);
    dt = Math.min(Math.max(dt, 2e-5), 5e-4);
    const b = this.bullet;
    if (b.x < this.xOut + 0.05) {
      if (b.x < this.xIn - 0.03) dt = Math.min(dt, Math.max(((this.xIn - 0.03) - b.x) / b.v, 1e-6) + 1e-7, 2e-4);
      else dt = Math.min(dt, 0.6 * this.s / b.v);
    }
    return dt;
  }

  // keep particles in Morton (Z-curve) order of their cells so that neighbours are close in
  // memory: the pair loops then run from cache instead of chasing random indices
  reorder() {
    const { N } = this, inv = 1 / this.h;
    const keys = this._keys || (this._keys = new Float64Array(N));
    const spread = (v) => { v &= 1023; v = (v | (v << 16)) & 0x030000FF; v = (v | (v << 8)) & 0x0300F00F; v = (v | (v << 4)) & 0x030C30C3; v = (v | (v << 2)) & 0x09249249; return v; };
    for (let i = 0; i < N; i++) {
      if (!this.alive[i]) { keys[i] = 2 ** 40 * 2 + i; continue; }
      const cx = Math.floor(this.pos[i * 3] * inv) + 512, cy = Math.floor(this.pos[i * 3 + 1] * inv) + 512, cz = Math.floor(this.pos[i * 3 + 2] * inv) + 512;
      keys[i] = (spread(cx) | (spread(cy) << 1) | (spread(cz) << 2)) * 16384 + i;
      if (keys[i] < 0) keys[i] += 2 ** 30 * 16384;
    }
    keys.sort();
    const perm = this._perm || (this._perm = new Int32Array(N));
    for (let k = 0; k < N; k++) perm[k] = keys[k] % 16384;
    const tmp3 = this._tmp3 || (this._tmp3 = new Float32Array(N * 3)), tmp1 = this._tmp1 || (this._tmp1 = new Float32Array(N));
    for (const name of ['pos', 'prev', 'vel']) {
      const a = this[name];
      for (let k = 0; k < N; k++) { const i = perm[k]; tmp3[k * 3] = a[i * 3]; tmp3[k * 3 + 1] = a[i * 3 + 1]; tmp3[k * 3 + 2] = a[i * 3 + 2]; }
      a.set(tmp3);
    }
    for (const name of ['nbSmooth', 'groundT', 'foam']) {
      const a = this[name];
      for (let k = 0; k < N; k++) tmp1[k] = a[perm[k]];
      a.set(tmp1);
    }
    for (const name of ['alive', 'nb']) {
      const a = this[name], c = a.slice();
      for (let k = 0; k < N; k++) a[k] = c[perm[k]];
    }
    this.pairsDirty = true;
  }

  step(dt) {
    if (((this.steps || 0) % 25) === 0 && this.N <= 16384) this.reorder();
    const p = this.p, b = this.bullet, m = this.mem;
    const { N, pos, prev, vel, alive, nb, nbSmooth } = this;
    const g = p.gravity;

    // bullet
    const xa = b.x;
    b.x += b.v * dt;
    if (m.tIn < 0 && b.x >= this.xIn) m.tIn = this.t + dt * ((this.xIn - xa) / (b.x - xa));
    if (m.tOut < 0 && b.x >= this.xOut) m.tOut = this.t + dt * ((this.xOut - xa) / (b.x - xa));
    this.applyBullet(xa, b.x);

    // membrane tear kinematics (latex retraction ~ tens of m/s)
    const tn = this.t + dt;
    const vr = p.snap;
    if (m.tIn >= 0) m.th1 = 0.05 + (vr * Math.max(0, tn - m.tIn)) / this.R;
    if (m.tOut >= 0) m.th2 = 0.05 + (vr * Math.max(0, tn - m.tOut)) / this.R;
    if (m.tIn >= 0) m.alpha = Math.PI + 0.3 - (vr * 0.8 * Math.max(0, tn - m.tIn - 0.0009)) / this.R;
    if (!m.gone && m.tIn >= 0 && (m.alpha < 0.28 || m.th1 + m.th2 > 2 * Math.PI)) { m.gone = true; m.tGone = tn; }

    // external forces
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const o = i * 3;
      vel[o + 1] -= g * dt;
      if (nbSmooth[i] < 5) { // isolated drops feel aerodynamic drag
        const sp = Math.hypot(vel[o], vel[o + 1], vel[o + 2]);
        const k = (3 * RHO_AIR * 0.47) / (8 * RHO_W * 0.5 * this.s);
        const d = 1 / (1 + k * sp * dt);
        vel[o] *= d; vel[o + 1] *= d; vel[o + 2] *= d;
      }
    }
    // temporary cavity: the tunnel left by the bullet is near-vacuum, so atmospheric pressure
    // on the outside decelerates the outward flow and then collapses the cavity (Rayleigh-type)
    if (m.tIn >= 0) {
      const age = tn - m.tIn;
      const pe = P_ATM * Math.exp(-age / 0.007) * p.cavity;
      if (pe > 50 && !m.cavClosed) {
        const ac = Math.max(1.1 * this.s, 4 * b.r);
        // has the tunnel pinched shut?  (enough fluid back on the shot line in the middle)
        if (age > 0.0006) {
          let onAxis = 0;
          for (let i = 0; i < N; i++) {
            if (!alive[i] || nbSmooth[i] < 6) continue;
            const o = i * 3;
            if (Math.abs(pos[o] - this.C[0]) > 0.6 * this.R) continue;
            if (Math.hypot(pos[o + 1] - b.y, pos[o + 2] - b.z) < 0.55 * ac) onAxis++;
          }
          if (onAxis > 6 * Math.pow(this.resK, 1.5)) m.cavClosed = true;
        }
        const L = Math.log(Math.max(1.5, this.R / ac));
        const xa0 = this.xIn - 0.3 * this.R, xb0 = this.xOut + 0.3 * this.R;
        for (let i = 0; i < N; i++) {
          if (!alive[i] || nbSmooth[i] < 6) continue;
          const o = i * 3;
          if (pos[o] < xa0 || pos[o] > xb0) continue;
          const ry = pos[o + 1] - b.y, rz = pos[o + 2] - b.z;
          const d = Math.hypot(ry, rz) + 1e-6;
          const vrad = (vel[o + 1] * ry + vel[o + 2] * rz) / d;
          if (d < 0.8 * ac && vrad < 0) continue; // cavity closed here
          let a = Math.min(4000, pe / (RHO_W * Math.max(d, ac) * L));
          // air rushing in through the holes softens the collapse phase
          if (vrad < 0) a *= 0.3;
          vel[o + 1] -= (a * dt * ry) / d;
          vel[o + 2] -= (a * dt * rz) / d;
        }
      }
    }
    prev.set(pos);
    for (let i = 0; i < N * 3; i++) pos[i] += vel[i] * dt;

    this.buildPairs();
    this.relax(1, dt);
    this.constrain(dt);

    // velocities + XSPH viscosity
    const inv = 1 / dt;
    for (let i = 0; i < N * 3; i++) vel[i] = (pos[i] - prev[i]) * inv;
    const dv = this.dv; dv.fill(0);
    const c = Math.min(0.2, 0.06 * this.resV) * Math.min(1, dt / DT_REF);
    for (let k = 0; k < this.npairs; k++) {
      const q = this.pq[k]; if (q >= 1) continue;
      const i = this.pa[k], j = this.pb[k], w = (1 - q) * c;
      for (let a = 0; a < 3; a++) {
        const d = (vel[j * 3 + a] - vel[i * 3 + a]) * w;
        dv[i * 3 + a] += d; dv[j * 3 + a] -= d;
      }
    }
    let vmax = 0;
    for (let i = 0; i < N; i++) {
      if (!alive[i]) continue;
      const o = i * 3;
      const f = 1 / (1 + 0.1 * nb[i] * 0);
      vel[o] += dv[o] / Math.max(1, nb[i] * 0.35) * f;
      vel[o + 1] += dv[o + 1] / Math.max(1, nb[i] * 0.35) * f;
      vel[o + 2] += dv[o + 2] / Math.max(1, nb[i] * 0.35) * f;
      nbSmooth[i] += (nb[i] - nbSmooth[i]) * 0.35;
      if (this.foam[i] > 0) this.foam[i] *= Math.exp(-dt / 0.45);
      const sp = Math.hypot(vel[o], vel[o + 1], vel[o + 2]);
      if (sp > V_JET) { this.atomise(i, sp); continue; }
      if (nb[i] >= 6 && sp > vmax) vmax = sp;
      // ground settling -> absorbed into the wet film
      // slow water lying in thin rivulets on the floor becomes part of the thin film (wet map)
      const thin = nb[i] < 14;
      if (pos[o + 1] <= this.rp + 1e-3 && sp < (thin ? 0.9 : 0.35)) {
        this.groundT[i] += dt;
        if (this.groundT[i] > (thin ? 0.03 : 0.06)) { alive[i] = 0; this.pairsDirty = true; this.deposit(pos[o], pos[o + 2], this.s * 2.2, 0.35); continue; }
      } else this.groundT[i] = Math.max(0, this.groundT[i] - dt);
      if (Math.abs(pos[o]) > 6 || Math.abs(pos[o + 2]) > 6 || pos[o + 1] > 8) { alive[i] = 0; this.pairsDirty = true; }
      // touching the ground wets it
      if (pos[o + 1] <= this.rp + 1e-3 && this.rng() < 0.08) this.deposit(pos[o], pos[o + 2], this.s * 1.5, 0.05);
      // aerodynamic break-up of fast, thin sheets -> mist and fine drops
      const vth = pos[o + 1] < 4 * this.s ? 2.6 : 6;   // the spreading floor sheet fringes into drops at a lower speed
      if (sp > vth && nb[i] < 10 && p.mist > 0) {
        const prob = (sp - vth) * dt * 90 * p.mist;
        if (this.rng() < prob) {
          const low = vth < 3;
          const mist = this.rng() < (low ? 0.15 : 0.6);
          const j = 0.35;
          // crown splash: the rim of the floor sheet lifts off at an angle
          const lift = low ? Math.hypot(vel[o], vel[o + 2]) * (0.25 + 0.5 * this.rng()) : 0;
          this.emit(mist ? 1 : 0, pos[o], pos[o + 1], pos[o + 2],
            vel[o] * (0.7 + j * this.rng()), vel[o + 1] * (0.7 + j * this.rng()) + lift, vel[o + 2] * (0.7 + j * this.rng()),
            mist ? 0.003 + 0.004 * this.rng() : 0.0005 + 0.0012 * this.rng(), mist ? 0.3 + 0.5 * this.rng() : 3);
        }
      }
    }
    this.vmaxBulk = vmax;
    { let gx = 0, gz = 0, gn = 0; for (let i = 0; i < N; i++) if (alive[i]) { gx += pos[i * 3]; gz += pos[i * 3 + 2]; gn++; } this.gc = gn ? [gx / gn, gz / gn] : null; }

    this.emitHole(dt, 0);
    this.emitHole(dt, 1);
    this.updateFx(dt);

    this.t = tn;
    this.steps++;
    this.version++;
  }

  // advance to physical time T with a step budget; returns true when reached
  advanceTo(T, maxSteps) {
    let s = 0;
    while (this.t < T - 1e-9 && s < maxSteps) {
      let dt = this.chooseDt();
      if (this.t + dt > T) dt = Math.max(T - this.t, 1e-6);
      this.step(dt);
      s++;
    }
    return this.t >= T - 1e-9;
  }

  snapshot() {
    const f = this.fx;
    return {
      t: this.t, pos: this.pos.slice(), prev: this.prev.slice(), vel: this.vel.slice(), alive: this.alive.slice(),
      nbSmooth: this.nbSmooth.slice(), groundT: this.groundT.slice(), foam: this.foam.slice(), wet: this.wet.slice(),
      bullet: Object.assign({}, this.bullet), mem: Object.assign({}, this.mem), vmax: this.vmaxBulk, rng: this.rngState(), gc: this.gc ? this.gc.slice() : null,
      fx: { n: f.n, head: f.head, pos: f.pos.slice(), vel: f.vel.slice(), age: f.age.slice(), life: f.life.slice(), size: f.size.slice(), type: f.type.slice(), alive: f.alive.slice() },
    };
  }
  restore(S) {
    this.t = S.t; this.pos.set(S.pos); this.prev.set(S.prev); this.vel.set(S.vel); this.alive.set(S.alive);
    this.gc = S.gc ? S.gc.slice() : null; this.pairsDirty = true; this.nbSmooth.set(S.nbSmooth); this.groundT.set(S.groundT); this.foam.set(S.foam); this.wet.set(S.wet); this.wetDirty = true;
    Object.assign(this.bullet, S.bullet); Object.assign(this.mem, S.mem); this.vmaxBulk = S.vmax;
    this.rng = mulberry32(S.rng);
    const f = this.fx; f.n = S.fx.n; f.head = S.fx.head;
    for (const k of ['pos', 'vel', 'age', 'life', 'size', 'type', 'alive']) f[k].set(S.fx[k]);
    this.buildPairs();
    this.version++;
  }
  rngState() { const v = Math.floor(this.rng() * 4294967296); this.rng = mulberry32(v); return v; }
}

if (typeof module !== 'undefined') module.exports = { BalloonSim, WET_RES, WET_EXT, FX_CAP };
