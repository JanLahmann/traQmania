// Track + car renderer. Prerenders the track surface to an offscreen canvas,
// interpolates car states between websocket frames, and draws effects.

export const KIND_COLORS = {
  quantum: "#7a5cff",
  mlp: "#2fbf71",
  human: "#ff9f1c",
};

const SURFACE_COLORS = {
  asphalt: "#2a2e37",
  concrete: "#3a3d44",
  dirt: "#4a3b2a",
};

const EDGE_COLORS = {
  "kerb-red": "#e5484d",
  "kerb-blue": "#3e63dd",
  "kerb-white": "#dfe3ea",
};

const RAY_ANGLES = [-Math.PI / 3, 0, Math.PI / 3]; // matches [observation].ray_angles_deg
const RAY_MAX_DIST = 30; // [observation].ray_max_dist — rays assumed normalized [0,1]

function themeColor(map, key, fallback) {
  if (typeof key === "string" && key.startsWith("#")) return key;
  return map[key] || fallback;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpAngle(a, b, t) {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

/** Cumulative arc lengths of a polyline (closed handled by caller). */
function arcLengths(pts) {
  const s = [0];
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i - 1][0];
    const dy = pts[i][1] - pts[i - 1][1];
    s.push(s[i - 1] + Math.hypot(dx, dy));
  }
  return s;
}

/** Point and unit normal on a closed centerline at arc fraction f in [0,1). */
function pointAtFraction(pts, cumS, totalLen, f) {
  const target = ((f % 1) + 1) % 1 * totalLen;
  let lo = 0;
  let hi = cumS.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cumS[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  const i1 = Math.max(1, lo);
  const i0 = i1 - 1;
  const seg = Math.max(cumS[i1] - cumS[i0], 1e-9);
  const t = (target - cumS[i0]) / seg;
  const x = lerp(pts[i0][0], pts[i1][0], t);
  const y = lerp(pts[i0][1], pts[i1][1], t);
  const tx = (pts[i1][0] - pts[i0][0]) / seg;
  const ty = (pts[i1][1] - pts[i0][1]) / seg;
  return { x, y, nx: -ty, ny: tx };
}

/** Offset a closed centerline by signed distance d along its normals. */
function offsetRing(pts, d) {
  const n = pts.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const next = pts[(i + 1) % n];
    let tx = next[0] - prev[0];
    let ty = next[1] - prev[1];
    const len = Math.hypot(tx, ty) || 1;
    tx /= len;
    ty /= len;
    out[i] = [pts[i][0] - ty * d, pts[i][1] + tx * d];
  }
  return out;
}

export class RaceRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.track = null;
    this.trackLayer = document.createElement("canvas");
    this.transform = { s: 1, ox: 0, oy: 0 };
    this.prev = null; // previous state msg (+recv timestamp)
    this.cur = null; // latest state msg
    this.effects = []; // {kind, carId, start}
    this.showRays = true;
    this.running = false;
    this._dirtyLayer = true;

    const ro = new ResizeObserver(() => this._resize());
    ro.observe(canvas.parentElement || canvas);
    this._resize();
  }

  setTrack(payload) {
    const center = payload.centerline || [];
    const hw = payload.half_width || 6;
    const cumS = arcLengths(
      center.length ? [...center, center[0]] : center,
    );
    this.track = {
      ...payload,
      left: payload.left && payload.left.length ? payload.left : offsetRing(center, +hw),
      right: payload.right && payload.right.length ? payload.right : offsetRing(center, -hw),
      cumS,
      totalLen: payload.total_length || cumS[cumS.length - 1] || 1,
    };
    this.prev = null;
    this.cur = null;
    this.effects.length = 0;
    this._fitCamera();
    this._dirtyLayer = true;
  }

  pushState(msg) {
    this.prev = this.cur;
    this.cur = { msg, recv: performance.now() };
  }

  addEffect(kind, carId) {
    this.effects.push({ kind, carId, start: performance.now() });
  }

  start() {
    if (this.running) return;
    this.running = true;
    const loop = () => {
      if (!this.running) return;
      this._draw();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  // -- internals -------------------------------------------------------------

  _resize() {
    const host = this.canvas.parentElement || document.body;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.floor(host.clientWidth * dpr));
    const h = Math.max(1, Math.floor(host.clientHeight * dpr));
    if (w === this.canvas.width && h === this.canvas.height) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.trackLayer.width = w;
    this.trackLayer.height = h;
    this._fitCamera();
    this._dirtyLayer = true;
  }

  _fitCamera() {
    if (!this.track) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const ring of [this.track.left, this.track.right]) {
      for (const [x, y] of ring) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    const pad = 0.06;
    const bw = Math.max(maxX - minX, 1e-6);
    const bh = Math.max(maxY - minY, 1e-6);
    const s = Math.min(
      (this.canvas.width * (1 - 2 * pad)) / bw,
      (this.canvas.height * (1 - 2 * pad)) / bh,
    );
    // world y up -> screen y down (flip)
    this.transform = {
      s,
      ox: this.canvas.width / 2 - s * (minX + bw / 2),
      oy: this.canvas.height / 2 + s * (minY + bh / 2),
    };
  }

  _applyWorld(ctx) {
    const { s, ox, oy } = this.transform;
    ctx.setTransform(s, 0, 0, -s, ox, oy);
  }

  _prerenderTrack() {
    const ctx = this.trackLayer.getContext("2d");
    const t = this.track;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.trackLayer.width, this.trackLayer.height);
    if (!t || !t.centerline.length) return;
    this._applyWorld(ctx);

    const surface = themeColor(SURFACE_COLORS, t.theme && t.theme.surface, "#2a2e37");
    const edge = themeColor(EDGE_COLORS, t.theme && t.theme.edge, "#dfe3ea");

    const ringPath = (path, ring) => {
      path.moveTo(ring[0][0], ring[0][1]);
      for (let i = 1; i < ring.length; i++) path.lineTo(ring[i][0], ring[i][1]);
      path.closePath();
    };

    // surface: annulus between the two boundary rings (evenodd is robust to
    // winding direction)
    const annulus = new Path2D();
    ringPath(annulus, t.left);
    ringPath(annulus, t.right);
    ctx.fillStyle = surface;
    ctx.fill(annulus, "evenodd");

    // boundary kerbs: dashed accent over a solid dark base line
    for (const ring of [t.left, t.right]) {
      const p = new Path2D();
      ringPath(p, ring);
      ctx.strokeStyle = "#14161c";
      ctx.lineWidth = 0.9;
      ctx.setLineDash([]);
      ctx.stroke(p);
      ctx.strokeStyle = edge;
      ctx.lineWidth = 0.55;
      ctx.setLineDash([1.6, 1.6]);
      ctx.stroke(p);
    }
    ctx.setLineDash([]);

    // faint centerline
    const cl = new Path2D();
    ringPath(cl, t.centerline);
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 0.25;
    ctx.setLineDash([1, 2]);
    ctx.stroke(cl);
    ctx.setLineDash([]);

    // checkpoints as thin lines across the track; the first is the start line
    const cps = t.checkpoints || [];
    const closed = [...t.centerline, t.centerline[0]];
    cps.forEach((frac, idx) => {
      const p = pointAtFraction(closed, t.cumS, t.totalLen, frac);
      const hw = t.half_width;
      ctx.beginPath();
      ctx.moveTo(p.x - p.nx * hw, p.y - p.ny * hw);
      ctx.lineTo(p.x + p.nx * hw, p.y + p.ny * hw);
      if (idx === 0) {
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.0;
      } else {
        ctx.strokeStyle = "rgba(255,255,255,0.28)";
        ctx.lineWidth = 0.35;
      }
      ctx.stroke();
    });

    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  /** Interpolated car list at render time (lerp between the last two states). */
  _carsNow() {
    if (!this.cur) return [];
    const cars = this.cur.msg.cars || [];
    if (!this.prev) return cars;
    const span = Math.max(this.cur.recv - this.prev.recv, 1);
    const alpha = Math.min(Math.max((performance.now() - this.cur.recv) / span, 0), 1);
    const prevById = new Map((this.prev.msg.cars || []).map((c) => [c.id, c]));
    return cars.map((c) => {
      const p = prevById.get(c.id);
      if (!p) return c;
      return {
        ...c,
        x: lerp(p.x, c.x, alpha),
        y: lerp(p.y, c.y, alpha),
        theta: lerpAngle(p.theta, c.theta, alpha),
        v: lerp(p.v, c.v, alpha),
      };
    });
  }

  _draw() {
    const ctx = this.ctx;
    if (this._dirtyLayer) {
      this._prerenderTrack();
      this._dirtyLayer = false;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#101218";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.trackLayer, 0, 0);
    if (!this.track) return;

    const cars = this._carsNow();
    const byId = new Map(cars.map((c) => [c.id, c]));

    this._applyWorld(ctx);
    for (const car of cars) {
      if (this.showRays && car.kind === "quantum" && Array.isArray(car.rays)) {
        this._drawRays(ctx, car);
      }
    }
    for (const car of cars) this._drawCar(ctx, car);
    this._drawEffects(ctx, byId);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  _drawRays(ctx, car) {
    ctx.strokeStyle = "rgba(122,92,255,0.35)";
    ctx.lineWidth = 0.18;
    for (let i = 0; i < Math.min(car.rays.length, RAY_ANGLES.length); i++) {
      const a = car.theta + RAY_ANGLES[i];
      const d = Math.max(0, Math.min(1, car.rays[i])) * RAY_MAX_DIST;
      ctx.beginPath();
      ctx.moveTo(car.x, car.y);
      ctx.lineTo(car.x + Math.cos(a) * d, car.y + Math.sin(a) * d);
      ctx.stroke();
      ctx.fillStyle = "rgba(122,92,255,0.55)";
      ctx.beginPath();
      ctx.arc(car.x + Math.cos(a) * d, car.y + Math.sin(a) * d, 0.35, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  _drawCar(ctx, car) {
    const color = KIND_COLORS[car.kind] || "#c8c8c8";
    const L = 2.6; // car length in world units
    const W = 1.5;
    ctx.save();
    ctx.translate(car.x, car.y);
    ctx.rotate(car.theta);

    // rounded-triangle body pointing +x
    ctx.beginPath();
    ctx.moveTo(L * 0.62, 0);
    ctx.quadraticCurveTo(L * 0.62, W * 0.36, L * 0.1, W * 0.5);
    ctx.quadraticCurveTo(-L * 0.42, W * 0.56, -L * 0.5, W * 0.3);
    ctx.quadraticCurveTo(-L * 0.56, 0, -L * 0.5, -W * 0.3);
    ctx.quadraticCurveTo(-L * 0.42, -W * 0.56, L * 0.1, -W * 0.5);
    ctx.quadraticCurveTo(L * 0.62, -W * 0.36, L * 0.62, 0);
    ctx.closePath();
    ctx.fillStyle = car.off_track ? "#6b6f7a" : color;
    ctx.fill();
    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.lineWidth = 0.12;
    ctx.stroke();

    // cockpit dot
    ctx.beginPath();
    ctx.arc(L * 0.05, 0, 0.28, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.fill();
    ctx.restore();
  }

  _drawEffects(ctx, byId) {
    const now = performance.now();
    this.effects = this.effects.filter((fx) => {
      const car = byId.get(fx.carId);
      const age = (now - fx.start) / 1000;
      if (!car) return age < 1;
      if (fx.kind === "crash") {
        if (age > 0.6) return false;
        const k = 1 - age / 0.6;
        ctx.beginPath();
        ctx.arc(car.x, car.y, 1.5 + (1 - k) * 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(229,72,77,${0.45 * k})`;
        ctx.fill();
        return true;
      }
      // lap / clean_lap pulse: expanding ring in car color
      if (age > 0.9) return false;
      const k = age / 0.9;
      ctx.beginPath();
      ctx.arc(car.x, car.y, 1.5 + k * 6, 0, 2 * Math.PI);
      ctx.strokeStyle = KIND_COLORS[car.kind] || "#fff";
      ctx.globalAlpha = 1 - k;
      ctx.lineWidth = 0.4;
      ctx.stroke();
      ctx.globalAlpha = 1;
      return true;
    });
  }
}
