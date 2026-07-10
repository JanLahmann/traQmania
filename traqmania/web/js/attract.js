// Attract-mode caption rotation + client-side idle timer that returns the
// exhibit to attract mode after `ui.attract_idle_seconds` without interaction.

// Rewritten by setCircuitSpec() for non-default circuit sizes: indexes 0, 1,
// 3 and 7 carry qubit / parameter / ray counts.
const CAPTIONS = [
  "A 4-qubit quantum circuit is driving this car.",
  "56 trainable parameters — a comparably tiny neural net drives the green car.",
  "Watch the panel on the right: live qubit measurements steer the car.",
  "Each action is one qubit: its ⟨Z⟩ expectation value becomes a Q-value.",
  "The circuit re-reads the car's sensors in every layer — data re-uploading.",
  "Purple = quantum agent, green = classical MLP. Same game, same rewards.",
  "Both agents learned by trial and error with double DQN.",
  "Three lidar rays, speed and heading — that's all the car can sense.",
  "Press Race to grab the wheel yourself (arrow keys or WASD).",
  "This runs a quantum simulator — the same circuit can run on real hardware.",
];

const ROTATE_MS = 6000;

export class AttractManager {
  constructor({ captionEl, onIdle }) {
    this.captionEl = captionEl;
    this.onIdle = onIdle;
    this.idleSeconds = 45;
    this.mode = "attract";
    this.captionIdx = 0;
    this.rotateId = null;
    this.idleId = null;

    const bump = () => this.notifyActivity();
    for (const ev of ["pointerdown", "pointermove", "keydown", "wheel", "touchstart"]) {
      window.addEventListener(ev, bump, { passive: true });
    }
  }

  setIdleSeconds(s) {
    this.idleSeconds = s;
    this._armIdle();
  }

  /** Rewrite the circuit-size captions from the welcome `circuit_spec`.
   *  No-op at the default 4 qubits, so the stock copy stays untouched. */
  setCircuitSpec(spec) {
    const n = spec && spec.n_qubits;
    if (!n || n === 4) return;
    CAPTIONS[0] = `A ${n}-qubit quantum circuit is driving this car.`;
    const total = spec.n_params && spec.n_params.total;
    if (total) {
      CAPTIONS[1] =
        `${total} trainable parameters — a comparably tiny neural net drives the green car.`;
    }
    CAPTIONS[3] =
      "The first four qubits are the actions: each ⟨Z⟩ expectation value becomes a Q-value.";
    const words = { 5: "Five", 7: "Seven", 9: "Nine" };
    CAPTIONS[7] = `${words[n - 1] || n - 1} lidar rays and speed — that's all the car can sense.`;
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === "attract") this._startCaptions();
    else this._stopCaptions();
    this._armIdle();
  }

  notifyActivity() {
    this._armIdle();
  }

  _armIdle() {
    if (this.idleId) clearTimeout(this.idleId);
    this.idleId = null;
    if (this.mode === "attract" || !this.idleSeconds) return;
    this.idleId = setTimeout(() => {
      if (this.mode !== "attract" && this.onIdle) this.onIdle();
    }, this.idleSeconds * 1000);
  }

  _startCaptions() {
    this.captionEl.hidden = false;
    this._showCaption();
    if (!this.rotateId) {
      this.rotateId = setInterval(() => {
        this.captionIdx = (this.captionIdx + 1) % CAPTIONS.length;
        this._showCaption();
      }, ROTATE_MS);
    }
  }

  _stopCaptions() {
    this.captionEl.hidden = true;
    if (this.rotateId) {
      clearInterval(this.rotateId);
      this.rotateId = null;
    }
  }

  _showCaption() {
    const el = this.captionEl;
    el.classList.remove("caption-in");
    el.textContent = CAPTIONS[this.captionIdx];
    // retrigger the fade-in animation
    void el.offsetWidth;
    el.classList.add("caption-in");
  }
}
