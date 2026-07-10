// Live quantum readout panel: 4 <Z_a> gauges in [-1, 1] plus a Q-value bar
// chart with argmax highlight. UI updates throttled to <= 15 fps.

export const ACTION_LABELS = ["Left", "Straight", "Right", "Brake"];

const UPDATE_MS = 67; // ~15 fps

export class QuantumPanel {
  constructor({ gaugesEl, barsEl, actionEl }) {
    this.actionEl = actionEl;
    this.pending = null;
    this.gauges = [];
    this.bars = [];

    gaugesEl.replaceChildren();
    barsEl.replaceChildren();

    for (let i = 0; i < 4; i++) {
      const row = document.createElement("div");
      row.className = "qgauge";
      const label = document.createElement("span");
      label.className = "qgauge-label";
      label.textContent = `Z${i}`;
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
      gaugesEl.append(row);
      this.gauges.push({ needle, value });
    }

    for (let i = 0; i < 4; i++) {
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
      label.textContent = ACTION_LABELS[i];
      col.append(val, stack, label);
      barsEl.append(col);
      this.bars.push({ col, bar, val });
    }

    setInterval(() => this._apply(), UPDATE_MS);
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
    for (let i = 0; i < this.gauges.length; i++) {
      const v = Math.max(-1, Math.min(1, exps[i] ?? 0));
      this.gauges[i].needle.style.left = `${((v + 1) / 2) * 100}%`;
      this.gauges[i].value.textContent = v.toFixed(2);
    }

    const qs = msg.q_values || [];
    if (qs.length) {
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
