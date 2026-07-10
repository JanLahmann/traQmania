// Minimal canvas charts: returns-per-episode for up to two agents
// (quantum vs mlp) plus epsilon on a secondary 0..1 axis, and a lap-time
// vs episode scatter fed from telemetry.lap_times / best_lap_s.

const SERIES_COLORS = { quantum: "#7a5cff", mlp: "#2fbf71" };
const EPS_COLOR = "rgba(160,168,186,0.7)";
const PAD = { l: 40, r: 34, t: 10, b: 22 };

export class TrainingChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = { quantum: [], mlp: [] }; // [{ep, ret}]
    this.epsilon = []; // [{ep, eps}]
    this._raf = 0;
    this.draw();
  }

  addPoint(agent, episode, meanReturn, epsilon) {
    if (this.series[agent]) this.series[agent].push({ ep: episode, ret: meanReturn });
    if (typeof epsilon === "number") this.epsilon.push({ ep: episode, eps: epsilon });
    if (!this._raf) {
      this._raf = requestAnimationFrame(() => {
        this._raf = 0;
        this.draw();
      });
    }
  }

  reset() {
    this.series = { quantum: [], mlp: [] };
    this.epsilon = [];
    this.draw();
  }

  _extent() {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const key of Object.keys(this.series)) {
      for (const p of this.series[key]) {
        if (p.ep < minX) minX = p.ep;
        if (p.ep > maxX) maxX = p.ep;
        if (p.ret < minY) minY = p.ret;
        if (p.ret > maxY) maxY = p.ret;
      }
    }
    if (!isFinite(minX)) return null;
    if (minX === maxX) maxX = minX + 1;
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    return { minX, maxX, minY, maxY };
  }

  draw() {
    const { ctx, canvas } = this;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px system-ui, sans-serif";

    const ext = this._extent();
    if (!ext) {
      ctx.fillStyle = "#8a91a0";
      ctx.textAlign = "center";
      ctx.fillText("waiting for telemetry…", W / 2, H / 2);
      return;
    }

    const px = (ep) => PAD.l + ((ep - ext.minX) / (ext.maxX - ext.minX)) * (W - PAD.l - PAD.r);
    const py = (v) => H - PAD.b - ((v - ext.minY) / (ext.maxY - ext.minY)) * (H - PAD.t - PAD.b);
    const pyEps = (v) => H - PAD.b - v * (H - PAD.t - PAD.b);

    // axes
    ctx.strokeStyle = "#2a3040";
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t);
    ctx.lineTo(PAD.l, H - PAD.b);
    ctx.lineTo(W - PAD.r, H - PAD.b);
    ctx.stroke();

    ctx.fillStyle = "#8a91a0";
    ctx.textAlign = "right";
    ctx.fillText(ext.maxY.toFixed(0), PAD.l - 4, PAD.t + 8);
    ctx.fillText(ext.minY.toFixed(0), PAD.l - 4, H - PAD.b);
    ctx.textAlign = "center";
    ctx.fillText(String(ext.minX), PAD.l, H - 8);
    ctx.fillText(String(ext.maxX), W - PAD.r, H - 8);
    ctx.textAlign = "left";
    ctx.fillText("ε", W - PAD.r + 4, PAD.t + 8);

    // epsilon (dashed, secondary 0..1 axis)
    if (this.epsilon.length > 1) {
      ctx.strokeStyle = EPS_COLOR;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      this.epsilon.forEach((p, i) => {
        if (i === 0) ctx.moveTo(px(p.ep), pyEps(p.eps));
        else ctx.lineTo(px(p.ep), pyEps(p.eps));
      });
      ctx.stroke();
      ctx.setLineDash([]);
      const last = this.epsilon[this.epsilon.length - 1];
      ctx.fillStyle = EPS_COLOR;
      ctx.fillText(last.eps.toFixed(2), W - PAD.r + 4, pyEps(last.eps) + 3);
    }

    // return series with last-value labels
    for (const [name, pts] of Object.entries(this.series)) {
      if (pts.length === 0) continue;
      ctx.strokeStyle = SERIES_COLORS[name];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(px(p.ep), py(p.ret));
        else ctx.lineTo(px(p.ep), py(p.ret));
      });
      ctx.stroke();
      ctx.lineWidth = 1;
      const last = pts[pts.length - 1];
      ctx.fillStyle = SERIES_COLORS[name];
      ctx.beginPath();
      ctx.arc(px(last.ep), py(last.ret), 2.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillText(last.ret.toFixed(1), Math.min(px(last.ep) + 5, W - 28), py(last.ret) - 4);
    }
  }
}

// -- SPSA loss chart ------------------------------------------------------------
// Single-series loss vs iteration line for hardware training sprints.

export class LossChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.points = []; // [{it, loss}]
    this._raf = 0;
    this.draw();
  }

  addPoint(iteration, loss) {
    this.points.push({ it: iteration, loss });
    if (!this._raf) {
      this._raf = requestAnimationFrame(() => {
        this._raf = 0;
        this.draw();
      });
    }
  }

  reset() {
    this.points = [];
    this.draw();
  }

  _extent() {
    if (this.points.length === 0) return null;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of this.points) {
      if (p.it < minX) minX = p.it;
      if (p.it > maxX) maxX = p.it;
      if (p.loss < minY) minY = p.loss;
      if (p.loss > maxY) maxY = p.loss;
    }
    if (minX === maxX) maxX = minX + 1;
    if (minY === maxY) {
      minY -= 0.5;
      maxY += 0.5;
    }
    return { minX, maxX, minY, maxY };
  }

  draw() {
    const { ctx, canvas } = this;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px system-ui, sans-serif";

    const ext = this._extent();
    if (!ext) {
      ctx.fillStyle = "#8a91a0";
      ctx.textAlign = "center";
      ctx.fillText("no sprint data yet…", W / 2, H / 2);
      return;
    }

    const px = (it) => PAD.l + ((it - ext.minX) / (ext.maxX - ext.minX)) * (W - PAD.l - PAD.r);
    const py = (v) => H - PAD.b - ((v - ext.minY) / (ext.maxY - ext.minY)) * (H - PAD.t - PAD.b);

    // axes
    ctx.strokeStyle = "#2a3040";
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t);
    ctx.lineTo(PAD.l, H - PAD.b);
    ctx.lineTo(W - PAD.r, H - PAD.b);
    ctx.stroke();

    ctx.fillStyle = "#8a91a0";
    ctx.textAlign = "right";
    ctx.fillText(ext.maxY.toFixed(2), PAD.l - 4, PAD.t + 8);
    ctx.fillText(ext.minY.toFixed(2), PAD.l - 4, H - PAD.b);
    ctx.textAlign = "center";
    ctx.fillText(String(ext.minX), PAD.l, H - 8);
    ctx.fillText(String(ext.maxX), W - PAD.r, H - 8);

    const color = SERIES_COLORS.quantum;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.points.forEach((p, i) => {
      if (i === 0) ctx.moveTo(px(p.it), py(p.loss));
      else ctx.lineTo(px(p.it), py(p.loss));
    });
    ctx.stroke();
    ctx.lineWidth = 1;
    const last = this.points[this.points.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px(last.it), py(last.loss), 2.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.textAlign = "left";
    ctx.fillText(last.loss.toFixed(3), Math.min(px(last.it) + 5, W - 30), py(last.loss) - 4);
  }
}

// -- lap-time chart -----------------------------------------------------------
// Scatter of lap time vs episode per agent. The y axis is inverted (best lap
// at the top — lower is better), with the overall best per agent as a dashed
// line + value.

export class LapChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.laps = { quantum: [], mlp: [] }; // [{ep, t}]
    this.best = { quantum: null, mlp: null };
    this._raf = 0;
    this.draw();
  }

  /** Replace one agent's data from a telemetry message (lap_times is the
   *  server-truncated "last <= 50 laps" list, so replacing is correct). */
  setAgentData(agent, lapTimes, bestLap) {
    if (!(agent in this.laps)) return;
    if (Array.isArray(lapTimes)) {
      this.laps[agent] = lapTimes
        .filter((p) => Array.isArray(p) && p.length >= 2)
        .map(([ep, t]) => ({ ep, t }));
    }
    if (typeof bestLap === "number" && isFinite(bestLap)) this.best[agent] = bestLap;
    if (!this._raf) {
      this._raf = requestAnimationFrame(() => {
        this._raf = 0;
        this.draw();
      });
    }
  }

  reset() {
    this.laps = { quantum: [], mlp: [] };
    this.best = { quantum: null, mlp: null };
    this.draw();
  }

  _extent() {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const key of Object.keys(this.laps)) {
      for (const p of this.laps[key]) {
        if (p.ep < minX) minX = p.ep;
        if (p.ep > maxX) maxX = p.ep;
        if (p.t < minY) minY = p.t;
        if (p.t > maxY) maxY = p.t;
      }
      const b = this.best[key];
      if (typeof b === "number") {
        if (b < minY) minY = b;
        if (b > maxY) maxY = b;
      }
    }
    if (!isFinite(minX) || !isFinite(minY)) return null;
    if (minX === maxX) maxX = minX + 1;
    if (minY === maxY) {
      minY -= 0.5;
      maxY += 0.5;
    }
    return { minX, maxX, minY, maxY };
  }

  draw() {
    const { ctx, canvas } = this;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.font = "10px system-ui, sans-serif";

    const ext = this._extent();
    if (!ext) {
      ctx.fillStyle = "#8a91a0";
      ctx.textAlign = "center";
      ctx.fillText("no laps yet…", W / 2, H / 2);
      return;
    }

    const px = (ep) => PAD.l + ((ep - ext.minX) / (ext.maxX - ext.minX)) * (W - PAD.l - PAD.r);
    // inverted: lowest (= best) lap time at the top
    const py = (t) => PAD.t + ((t - ext.minY) / (ext.maxY - ext.minY)) * (H - PAD.t - PAD.b);

    // axes
    ctx.strokeStyle = "#2a3040";
    ctx.beginPath();
    ctx.moveTo(PAD.l, PAD.t);
    ctx.lineTo(PAD.l, H - PAD.b);
    ctx.lineTo(W - PAD.r, H - PAD.b);
    ctx.stroke();

    ctx.fillStyle = "#8a91a0";
    ctx.textAlign = "right";
    ctx.fillText(`${ext.minY.toFixed(1)}s`, PAD.l - 4, PAD.t + 8);
    ctx.fillText(`${ext.maxY.toFixed(1)}s`, PAD.l - 4, H - PAD.b);
    ctx.textAlign = "center";
    ctx.fillText(String(ext.minX), PAD.l, H - 8);
    ctx.fillText(String(ext.maxX), W - PAD.r, H - 8);

    for (const [name, pts] of Object.entries(this.laps)) {
      const color = SERIES_COLORS[name];
      // best lap: dashed line + value
      const best = this.best[name];
      if (typeof best === "number") {
        const y = py(best);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.7;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(PAD.l, y);
        ctx.lineTo(W - PAD.r, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        ctx.textAlign = "left";
        ctx.fillText(`${best.toFixed(2)}s`, W - PAD.r + 2, y + 3);
      }
      if (pts.length === 0) continue;
      // faint connecting line + scatter dots
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      pts.forEach((p, i) => {
        if (i === 0) ctx.moveTo(px(p.ep), py(p.t));
        else ctx.lineTo(px(p.ep), py(p.t));
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = color;
      for (const p of pts) {
        ctx.beginPath();
        ctx.arc(px(p.ep), py(p.t), 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}
