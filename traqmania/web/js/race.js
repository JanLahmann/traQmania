// Track + car renderer. Prerenders the track surface to an offscreen canvas,
// interpolates car states between websocket frames, and draws effects.

export const KIND_COLORS = {
  quantum: "#7a5cff",
  mlp: "#2fbf71",
  human: "#ff9f1c",
  hero: "#22d3ee", // expert-menu racing-line controller
  pro: "#5eead4", // expert-menu big DQN-trained MLP
};

// Evolution mode: one colour per training stage (cool -> hot as training
// progresses), assigned per unique car label in order of first appearance.
export const STAGE_COLORS = ["#5c6b8a", "#3e63dd", "#7a5cff", "#e5484d", "#ff9f1c"];

const TRAIL_MAX = 300; // points kept per car
const TRAIL_BREAK_DIST = 8; // world units; larger jumps break the polyline
const TRAIL_ALPHA = { train: 0.4, evolution: 0.26, attract: 0.22, race: 0.2 };

const V_MAX = 22.0; // [car].v_max — speed normalization for trails and bars
// Trail speed shading: below SPEED_COLD renders darkest, above SPEED_HOT
// brightest; the useful racing range on the bundled tracks sits in between.
const SPEED_COLD = 5.0;
const SPEED_HOT = 19.0;
const N_SPEED_BUCKETS = 5;
const BRAKE_DECEL = 9.0; // world units/s²; drag alone tops out at ~7.7

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

// Ray fan: n rays evenly spaced over [-60°, +60°], matching how the config
// profiles lay out [observation].ray_angles_deg for any qubit count.
function rayAngle(i, n) {
  return n > 1 ? -Math.PI / 3 + ((2 * Math.PI) / 3) * (i / (n - 1)) : 0;
}
const RAY_MAX_DIST = 30; // [observation].ray_max_dist — rays assumed normalized [0,1]

function themeColor(map, key, fallback) {
  if (typeof key === "string" && key.startsWith("#")) return key;
  return map[key] || fallback;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** CSS color t of the way from hex c1 to hex c2. */
function mixColor(c1, c2, t) {
  const a = hexToRgb(c1);
  const b = hexToRgb(c2);
  return `rgb(${Math.round(lerp(a[0], b[0], t))},${Math.round(
    lerp(a[1], b[1], t),
  )},${Math.round(lerp(a[2], b[2], t))})`;
}

/** Speed -> trail bucket index (0 = slowest/darkest). */
function speedBucket(v) {
  const t = Math.min(Math.max((v - SPEED_COLD) / (SPEED_HOT - SPEED_COLD), 0), 1);
  return Math.round(t * (N_SPEED_BUCKETS - 1));
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
    this.mode = "attract";
    this.trails = new Map(); // car id -> {car, pts: [[x,y,v]|null, ...]}
    this._stageColors = new Map(); // evolution label -> palette colour
    this._speedPalettes = new Map(); // car color -> per-bucket trail colors

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
    this.trails.clear();
    this._stageColors.clear();
    this._fitCamera();
    this._dirtyLayer = true;
  }

  pushState(msg) {
    this.prev = this.cur;
    this.cur = { msg, recv: performance.now() };
    this._updateTrails(msg.cars || []);
  }

  setMode(mode) {
    this.mode = mode;
  }

  /** Stable palette colour for an evolution stage label. */
  stageColor(label) {
    if (!this._stageColors.has(label)) {
      this._stageColors.set(label, STAGE_COLORS[this._stageColors.size % STAGE_COLORS.length]);
    }
    return this._stageColors.get(label);
  }

  /** 1-based race number for an evolution stage label (same first-appearance
   *  order as stageColor, so number and colour always agree with the legend). */
  stageNumber(label) {
    this.stageColor(label); // ensure the label is registered
    return [...this._stageColors.keys()].indexOf(label) + 1;
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

  /** Ring buffer of recent positions per (non-ghost) car, fed per ws frame. */
  _updateTrails(cars) {
    const seen = new Set();
    for (const car of cars) {
      seen.add(car.id);
      if (car.ghost) {
        this.trails.delete(car.id);
        continue;
      }
      let tr = this.trails.get(car.id);
      if (!tr) {
        tr = { car, pts: [] };
        this.trails.set(car.id, tr);
      }
      tr.car = car;
      const pts = tr.pts;
      const last = pts.length ? pts[pts.length - 1] : null;
      // respawn/teleport: break the polyline instead of drawing a chord
      if (last && Math.hypot(car.x - last[0], car.y - last[1]) > TRAIL_BREAK_DIST) {
        pts.push(null);
      }
      pts.push([car.x, car.y, car.v]);
      while (pts.length > TRAIL_MAX) pts.shift();
    }
    for (const id of [...this.trails.keys()]) {
      if (!seen.has(id)) this.trails.delete(id);
    }
  }

  _carColor(car) {
    if (this.mode === "evolution" && car.label && !car.ghost) return this.stageColor(car.label);
    return KIND_COLORS[car.kind] || "#c8c8c8";
  }

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
        dvdt: ((c.v - p.v) / span) * 1000,
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
    this._drawTrails(ctx);
    for (const car of cars) {
      if (this.showRays && car.kind === "quantum" && Array.isArray(car.rays)) {
        this._drawRays(ctx, car);
      }
    }
    for (const car of cars) this._drawCar(ctx, car);
    this._drawEffects(ctx, byId);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    this._drawLabels(ctx, cars);
  }

  /** Per-bucket trail colors for one car color: dark (slow) -> bright (fast). */
  _speedPalette(color) {
    let pal = this._speedPalettes.get(color);
    if (!pal) {
      pal = Array.from({ length: N_SPEED_BUCKETS }, (_, b) => {
        const t = b / (N_SPEED_BUCKETS - 1);
        return t < 0.5
          ? mixColor("#171c28", color, 0.25 + 1.5 * t) // slow: sunk toward the background
          : mixColor(color, "#ffffff", (t - 0.5) * 0.7); // fast: pushed toward white
      });
      this._speedPalettes.set(color, pal);
    }
    return pal;
  }

  /** Fading racing-line trails: one polyline per car, stroked in a few
   *  alpha chunks (oldest faintest). Within a chunk, segments are grouped by
   *  speed bucket — slow sections read dark and thin, fast ones bright and
   *  wide, so braking zones show up on the racing line. Ghost cars never
   *  leave trails. */
  _drawTrails(ctx) {
    if (!this.trails.size) return;
    const maxAlpha = TRAIL_ALPHA[this.mode] ?? 0.2;
    const chunks = 4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    for (const tr of this.trails.values()) {
      const pts = tr.pts;
      if (pts.length < 2) continue;
      const palette = this._speedPalette(this._carColor(tr.car));
      const per = Math.ceil(pts.length / chunks);
      for (let c = 0; c < chunks; c++) {
        const start = c * per;
        const end = Math.min(pts.length - 1, (c + 1) * per);
        if (start >= end) continue;
        const paths = new Array(N_SPEED_BUCKETS).fill(null);
        for (let i = start + 1; i <= end; i++) {
          const p0 = pts[i - 1];
          const p1 = pts[i];
          if (!p0 || !p1) continue; // respawn break
          const b = speedBucket(p1[2]);
          if (!paths[b]) paths[b] = new Path2D();
          paths[b].moveTo(p0[0], p0[1]);
          paths[b].lineTo(p1[0], p1[1]);
        }
        ctx.globalAlpha = maxAlpha * ((c + 1) / chunks);
        for (let b = 0; b < N_SPEED_BUCKETS; b++) {
          if (!paths[b]) continue;
          ctx.strokeStyle = palette[b];
          ctx.lineWidth = 0.3 + 0.09 * b;
          ctx.stroke(paths[b]);
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  /** Small speed bar below each car (fill = v / v_max; red under braking).
   *  Car descriptions live in the corner legend, not on the moving car; in
   *  evolution mode each car carries its race number in its stage colour. */
  _drawLabels(ctx, cars) {
    const { s, ox, oy } = this.transform;
    const dpr = window.devicePixelRatio || 1;
    if (this.mode === "evolution") {
      ctx.font = `700 ${12 * dpr}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      for (const car of cars) {
        if (!car.label || car.ghost) continue;
        ctx.fillStyle = this.stageColor(car.label);
        ctx.fillText(String(this.stageNumber(car.label)),
                     s * car.x + ox, -s * car.y + oy - s * 1.6);
      }
      ctx.textBaseline = "alphabetic";
    }
    for (const car of cars) {
      if (car.ghost) continue;
      const braking = car.v > 2 && car.dvdt < -BRAKE_DECEL;
      const w = 34 * dpr;
      const h = 5 * dpr;
      const r = h / 2;
      const sx = s * car.x + ox - w / 2;
      const sy = -s * car.y + oy + s * 1.9;
      const frac = Math.min(Math.max(car.v / V_MAX, 0), 1);
      // speedometer gauge: bordered rounded track + fill (red = braking)
      ctx.beginPath();
      ctx.roundRect(sx, sy, w, h, r);
      ctx.fillStyle = "rgba(16,18,24,0.75)";
      ctx.fill();
      ctx.strokeStyle = "rgba(230,233,239,0.5)";
      ctx.lineWidth = dpr;
      ctx.stroke();
      if (frac > 0.02) {
        ctx.beginPath();
        ctx.roundRect(sx + dpr, sy + dpr, (w - 2 * dpr) * frac, h - 2 * dpr, r);
        ctx.fillStyle = braking ? "#e5484d" : this._carColor(car);
        ctx.fill();
      }
    }
  }

  _drawRays(ctx, car) {
    ctx.strokeStyle = "rgba(122,92,255,0.35)";
    ctx.lineWidth = 0.18;
    for (let i = 0; i < car.rays.length; i++) {
      const a = car.theta + rayAngle(i, car.rays.length);
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
    const color = this._carColor(car);
    const ghost = car.ghost === true;
    const L = 2.6; // car length in world units
    const W = 1.5;
    ctx.save();
    if (ghost) ctx.globalAlpha = 0.35;
    ctx.translate(car.x, car.y);
    ctx.rotate(car.theta);

    // brake flare: red glow off the tail while decelerating hard (drag alone
    // can't exceed ~7.7 u/s², so this only fires on actual braking)
    if (!ghost && !car.off_track && car.v > 2 && car.dvdt < -BRAKE_DECEL) {
      const g = ctx.createRadialGradient(-L * 0.55, 0, 0.1, -L * 0.55, 0, 1.7);
      g.addColorStop(0, "rgba(229,72,77,0.85)");
      g.addColorStop(1, "rgba(229,72,77,0)");
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(-L * 0.55, 0, 1.7, 0, 2 * Math.PI);
      ctx.fill();
    }

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
    if (ghost) {
      // dashed white outline marks the ghost
      ctx.setLineDash([0.6, 0.4]);
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 0.16;
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.strokeStyle = "rgba(0,0,0,0.5)";
      ctx.lineWidth = 0.12;
      ctx.stroke();
    }

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
