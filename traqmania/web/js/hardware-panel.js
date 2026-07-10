// Hardware tab: run the trained circuit on an IBM Quantum backend (real
// device, or a local noisy "fake" simulation of one). Sends C->S "hardware"
// commands and renders S->C "hardware_status" updates: phase pill, live
// counters, sprint loss chart and before/after eval-return comparison.

import { hardwareCmd } from "./net.js";
import { LossChart } from "./charts.js";

const BUSY_PHASES = new Set(["connecting", "transpiling", "running"]);

const PHASE_CLASS = {
  idle: "pill-idle",
  connecting: "pill-busy",
  transpiling: "pill-busy",
  running: "pill-busy",
  replay: "pill-ok",
  done: "pill-ok",
  error: "pill-err",
};

export function initHardwarePanel() {
  const $ = (sel) => document.querySelector(sel);
  const els = {
    backend: $("#hw-backend"),
    shots: $("#hw-shots"),
    iterations: $("#hw-iterations"),
    lap: $("#hw-lap"),
    sprint: $("#hw-sprint"),
    abort: $("#hw-abort"),
    phase: $("#hw-phase"),
    backendName: $("#hw-backend-name"),
    message: $("#hw-message"),
    counters: $("#hw-counters"),
    eval: $("#hw-eval"),
    replayCaption: $("#hw-replay-caption"),
  };
  const lossChart = new LossChart($("#hw-loss-chart"));

  const counters = new Map(); // label -> formatted value, insertion-ordered
  let evalBefore = null;
  let evalAfter = null;

  function intVal(el, fallback) {
    const v = parseInt(el.value, 10);
    return Number.isFinite(v) ? v : fallback;
  }

  function setBusy(busy) {
    els.lap.disabled = busy;
    els.sprint.disabled = busy;
    els.abort.disabled = !busy;
  }

  function renderCounters() {
    els.counters.innerHTML = [...counters.entries()]
      .map(
        ([label, value]) =>
          `<div class="stat-row"><span>${label}</span><span><b>${value}</b></span></div>`,
      )
      .join("");
  }

  function renderEval() {
    if (evalBefore === null && evalAfter === null) {
      els.eval.innerHTML = "";
      return;
    }
    const fmt = (v) => (typeof v === "number" ? v.toFixed(1) : "…");
    let delta = "";
    if (typeof evalBefore === "number" && typeof evalAfter === "number") {
      const d = evalAfter - evalBefore;
      const cls = d >= 0 ? "eval-up" : "eval-down";
      delta = ` <span class="${cls}">(${d >= 0 ? "+" : ""}${d.toFixed(1)})</span>`;
    }
    els.eval.innerHTML = `<div class="stat-row"><span>eval return</span>
      <span>before <b>${fmt(evalBefore)}</b> → after <b>${fmt(evalAfter)}</b>${delta}</span></div>`;
  }

  function resetRun() {
    counters.clear();
    renderCounters();
    els.message.textContent = "";
  }

  els.lap.addEventListener("click", () => {
    resetRun();
    hardwareCmd("lap", {
      backend: els.backend.value,
      shots: intVal(els.shots, 1024),
    });
    setBusy(true); // optimistic; hardware_status confirms
  });

  els.sprint.addEventListener("click", () => {
    resetRun();
    lossChart.reset();
    evalBefore = null;
    evalAfter = null;
    renderEval();
    hardwareCmd("sprint", {
      backend: els.backend.value,
      iterations: intVal(els.iterations, 10),
      shots: intVal(els.shots, 1024),
    });
    setBusy(true);
  });

  els.abort.addEventListener("click", () => hardwareCmd("abort"));

  /** Handle an S->C hardware_status message. */
  function handleStatus(msg) {
    const phase = msg.phase || "idle";
    els.phase.textContent = phase;
    els.phase.className = `pill ${PHASE_CLASS[phase] || "pill-idle"}`;
    if (typeof msg.backend_name === "string") els.backendName.textContent = msg.backend_name;
    if (typeof msg.message === "string") els.message.textContent = msg.message;
    els.message.classList.toggle("hw-error", phase === "error");
    setBusy(BUSY_PHASES.has(phase));
    els.replayCaption.hidden = phase !== "replay";

    if (typeof msg.decision === "number") counters.set("decision", String(msg.decision));
    if (typeof msg.seconds_per_decision === "number") {
      counters.set("s / decision", msg.seconds_per_decision.toFixed(2));
    }
    if (typeof msg.iteration === "number") counters.set("iteration", String(msg.iteration));
    if (typeof msg.loss === "number") {
      counters.set("loss", msg.loss.toFixed(4));
      if (typeof msg.iteration === "number") lossChart.addPoint(msg.iteration, msg.loss);
    }
    if (typeof msg.lap_time === "number") {
      counters.set("hardware lap", `${msg.lap_time.toFixed(2)}s`);
    }
    renderCounters();

    if (typeof msg.eval_return_before === "number") evalBefore = msg.eval_return_before;
    if (typeof msg.eval_return_after === "number") evalAfter = msg.eval_return_after;
    renderEval();
  }

  setBusy(false);
  renderCounters();
  return { handleStatus };
}
