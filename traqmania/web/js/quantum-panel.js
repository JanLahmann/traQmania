// Live quantum readout panel: one <Z_a> gauge per readout expectation in
// [-1, 1] plus a Q-value bar chart with argmax highlight. Gauge/bar counts
// follow the incoming `quantum` messages. UI updates throttled to <= 15 fps.

// Action 0 steers -1 (theta decreases = clockwise = RIGHT on screen); action 2 is +1 = left.
export const ACTION_LABELS = ["Right", "Straight", "Left", "Brake"];

const UPDATE_MS = 67; // ~15 fps
const DEFAULT_COUNT = 4; // shown until the first quantum message arrives

export class QuantumPanel {
  constructor({ gaugesEl, barsEl, actionEl }) {
    this.gaugesEl = gaugesEl;
    this.barsEl = barsEl;
    this.actionEl = actionEl;
    this.pending = null;
    this.gauges = [];
    this.bars = [];
    // Auto-zoom axis: trained drivers park every <Z> near +1, so a fixed
    // [-1, 1] axis pins all needles right. Track a rolling range instead
    // (fast to grow, slow to shrink) and show it under the gauges.
    this._rangeLo = -1;
    this._rangeHi = 1;
    this._rangeEl = document.createElement("p");
    this._rangeEl.className = "hint qgauge-range";

    this._buildGauges(DEFAULT_COUNT);
    this._buildBars(DEFAULT_COUNT);

    setInterval(() => this._apply(), UPDATE_MS);
  }

  _buildGauges(n) {
    this.gaugesEl.replaceChildren();
    this.gauges = [];
    for (let i = 0; i < n; i++) {
      const row = document.createElement("div");
      row.className = "qgauge";
      const label = document.createElement("span");
      label.className = "qgauge-label";
      label.textContent = `Z${i}`;
      if (i < 4) {
        // the first four qubits are the action readout the output head uses
        row.classList.add("qgauge-readout");
        label.title = "readout qubit — this ⟨Z⟩ becomes one of the four Q-values";
      }
      const track = document.createElement("div");
      track.className = "qgauge-track";
      const zero = document.createElement("div");
      zero.className = "qgauge-zero";
      const needle = document.createElement("div");
      needle.className = "qgauge-needle";
      track.append(zero, needle);
      const value = document.createElement("span");
      value.className = "qgauge-value";
      value.textContent = "0.00";
      row.append(label, track, value);
      this.gaugesEl.append(row);
      this.gauges.push({ needle, value, zero });
    }
  }

  _buildBars(n) {
    this.barsEl.replaceChildren();
    this.bars = [];
    for (let i = 0; i < n; i++) {
      const col = document.createElement("div");
      col.className = "qbar-col";
      const stack = document.createElement("div");
      stack.className = "qbar-stack";
      const bar = document.createElement("div");
      bar.className = "qbar";
      stack.append(bar);
      const val = document.createElement("span");
      val.className = "qbar-value";
      val.textContent = "—";
      const label = document.createElement("span");
      label.className = "qbar-label";
      label.textContent = ACTION_LABELS[i] ?? `A${i}`;
      col.append(val, stack, label);
      this.barsEl.append(col);
      this.bars.push({ col, bar, val });
    }
  }

  /** Feed a `quantum` protocol message; applied on the next UI tick. */
  update(msg) {
    this.pending = msg;
  }

  _apply() {
    const msg = this.pending;
    if (!msg) return;
    this.pending = null;

    const exps = msg.expectations || [];
    if (exps.length && exps.length !== this.gauges.length) this._buildGauges(exps.length);
    if (exps.length) {
      // grow the window immediately to include every value; shrink slowly so
      // the axis breathes instead of jittering
      const lo = Math.min(...exps);
      const hi = Math.max(...exps);
      const shrink = 0.01 * (this._rangeHi - this._rangeLo);
      this._rangeLo = Math.min(lo, this._rangeLo + shrink);
      this._rangeHi = Math.max(hi, this._rangeHi - shrink);
      const pad = 0.06 * (this._rangeHi - this._rangeLo) + 0.005;
      const axisLo = Math.max(-1, this._rangeLo - pad);
      const axisHi = Math.min(1, this._rangeHi + pad);
      const span = axisHi - axisLo || 1e-9;
      for (let i = 0; i < this.gauges.length; i++) {
        const v = Math.max(-1, Math.min(1, exps[i] ?? 0));
        const pct = Math.max(0, Math.min(1, (v - axisLo) / span)) * 100;
        this.gauges[i].needle.style.left = `${pct}%`;
        this.gauges[i].value.textContent = v.toFixed(2);
        const zero = this.gauges[i].zero;
        if (zero) {
          const zeroIn = axisLo <= 0 && 0 <= axisHi;
          zero.style.display = zeroIn ? "" : "none";
          if (zeroIn) zero.style.left = `${((0 - axisLo) / span) * 100}%`;
        }
      }
      const zoomed = axisHi - axisLo < 1.98;
      this._rangeEl.textContent = zoomed
        ? `axis auto-zoomed to ${axisLo.toFixed(2)} … ${axisHi.toFixed(2)} (full scale is −1 … +1)`
        : "axis: −1 … +1";
      if (this._rangeEl.parentElement !== this.gaugesEl.parentElement) {
        this.gaugesEl.after(this._rangeEl);
      }
    }

    const qs = msg.q_values || [];
    if (qs.length) {
      if (qs.length !== this.bars.length) this._buildBars(qs.length);
      const min = Math.min(...qs);
      const max = Math.max(...qs);
      const span = max - min + 1e-9; // normalize so within-update differences are visible
      for (let i = 0; i < this.bars.length; i++) {
        const q = qs[i] ?? min;
        const pct = 8 + ((q - min) / span) * 92; // keep a visible stub for the min
        this.bars[i].bar.style.height = `${pct}%`;
        this.bars[i].val.textContent = q.toFixed(2);
        this.bars[i].col.classList.toggle("argmax", i === msg.action);
      }
    }

    if (this.actionEl && Number.isInteger(msg.action)) {
      this.actionEl.textContent = ACTION_LABELS[msg.action] ?? `action ${msg.action}`;
    }
  }
}
