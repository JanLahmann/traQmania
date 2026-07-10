// Boot + wiring: websocket handlers, mode/track UI, lap board, panels.

import * as net from "./net.js";
import { RaceRenderer, KIND_COLORS } from "./race.js";
import { initInput, setInputActive } from "./input.js";
import { QuantumPanel } from "./quantum-panel.js";
import { renderCircuit } from "./circuit.js";
import { TrainingChart } from "./charts.js";
import { AttractManager } from "./attract.js";
import { initExplain } from "./explain.js";

const $ = (sel) => document.querySelector(sel);

const state = {
  mode: "attract",
  tracks: [],
  trackName: null,
  bestLaps: new Map(), // car id -> best lap seconds
  lastState: null,
  training: false,
};

// -- components --------------------------------------------------------------

const renderer = new RaceRenderer($("#race-canvas"));
renderer.start();

const quantumPanel = new QuantumPanel({
  gaugesEl: $("#qgauges"),
  barsEl: $("#qbars"),
  actionEl: $("#qaction"),
});

const chart = new TrainingChart($("#training-chart"));

const attract = new AttractManager({
  captionEl: $("#attract-caption"),
  onIdle: () => net.setMode("attract"),
});

initExplain($("#panel-explain"));
initInput(() => attract.notifyActivity());

// -- helpers -----------------------------------------------------------------

const KIND_NAMES = { quantum: "Quantum", mlp: "MLP", human: "You" };

function fmtLap(t) {
  if (typeof t !== "number" || !isFinite(t)) return "—";
  return `${t.toFixed(2)}s`;
}

function toast(message, isError = true) {
  const el = $("#toast");
  el.textContent = message;
  el.classList.toggle("toast-error", isError);
  el.hidden = false;
  clearTimeout(toast._id);
  toast._id = setTimeout(() => {
    el.hidden = true;
  }, 4000);
}

function setStatus(text, cls) {
  const pill = $("#status-pill");
  pill.textContent = text;
  pill.className = `pill ${cls}`;
}

function applyMode(mode) {
  state.mode = mode;
  for (const btn of document.querySelectorAll(".mode-btn")) {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  }
  $("#race-controls").hidden = mode !== "race";
  setInputActive(mode === "race");
  attract.setMode(mode);
  if (mode === "train") selectTab("training");
  else if (mode === "attract") selectTab("quantum");
}

function selectTab(name) {
  for (const btn of document.querySelectorAll(".tab-btn")) {
    btn.classList.toggle("active", btn.dataset.tab === name);
  }
  for (const panel of document.querySelectorAll(".panel")) {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  }
}

function applyTrack(payload) {
  state.trackName = payload.name;
  state.bestLaps.clear();
  renderer.setTrack(payload);
  const sel = $("#track-select");
  if (sel.value !== payload.name) sel.value = payload.name;
}

// -- lap board ---------------------------------------------------------------

let lapboardAt = 0;

function renderLapboard(cars) {
  const now = performance.now();
  if (now - lapboardAt < 250) return;
  lapboardAt = now;
  const board = $("#lapboard");
  const rows = cars.map((car) => {
    const best = state.bestLaps.get(car.id);
    const color = KIND_COLORS[car.kind] || "#ccc";
    return `<div class="lap-row${car.off_track ? " off" : ""}">
      <span class="dot" style="background:${color}"></span>
      <span class="lap-kind">${KIND_NAMES[car.kind] || car.kind}</span>
      <span class="lap-cell">Lap <b>${car.lap}</b></span>
      <span class="lap-cell">Last <b>${fmtLap(car.last_lap_time)}</b></span>
      <span class="lap-cell">Best <b>${fmtLap(best)}</b></span>
    </div>`;
  });
  board.innerHTML = rows.join("");
}

// -- websocket handlers ------------------------------------------------------

net.on("_open", () => setStatus("connected", "pill-ok"));
net.on("_close", () => setStatus("reconnecting…", "pill-off"));

net.on("welcome", (msg) => {
  state.tracks = msg.tracks || [];
  const sel = $("#track-select");
  sel.replaceChildren(
    ...state.tracks.map((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      return opt;
    }),
  );
  if (msg.circuit_spec) {
    renderCircuit(msg.circuit_spec, $("#circuit-diagram"), $("#circuit-legend"));
  }
  if (msg.ui) {
    attract.setIdleSeconds(msg.ui.attract_idle_seconds || 45);
    document.body.classList.toggle("kiosk", Boolean(msg.ui.kiosk));
  }
  if (msg.track) applyTrack(msg.track);
  applyMode(msg.mode || "attract");
});

net.on("track", (msg) => applyTrack(msg.track));

net.on("state", (msg) => {
  state.lastState = msg;
  renderer.pushState(msg);
  renderLapboard(msg.cars || []);
  if (msg.mode && msg.mode !== state.mode) applyMode(msg.mode); // server-driven mode change
});

net.on("quantum", (msg) => quantumPanel.update(msg));

net.on("telemetry", (msg) => {
  chart.addPoint(msg.agent, msg.episode, msg.mean_return, msg.epsilon);
  updateTrainStats(msg);
});

net.on("event", (msg) => {
  switch (msg.kind) {
    case "lap":
    case "clean_lap":
      if (msg.car_id) {
        renderer.addEffect("lap", msg.car_id);
        if (typeof msg.lap_time === "number") {
          const best = state.bestLaps.get(msg.car_id);
          if (best === undefined || msg.lap_time < best) {
            state.bestLaps.set(msg.car_id, msg.lap_time);
          }
        }
      }
      break;
    case "crash":
      if (msg.car_id) renderer.addEffect("crash", msg.car_id);
      break;
    case "training_done":
      state.training = false;
      $("#train-start").disabled = false;
      toast(`Training done${msg.agent ? ` (${msg.agent})` : ""}`, false);
      break;
  }
});

net.on("error", (msg) => toast(msg.message || "server error"));

// -- UI wiring ---------------------------------------------------------------

const MODE_FOR_BUTTON = { attract: "attract", train: "train", race: "race" };

for (const btn of document.querySelectorAll(".mode-btn")) {
  btn.addEventListener("click", () => {
    const mode = MODE_FOR_BUTTON[btn.dataset.mode];
    net.setMode(mode);
    applyMode(mode); // optimistic; server `state.mode` confirms
  });
}

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => selectTab(btn.dataset.tab));
}

$("#track-select").addEventListener("change", (ev) => net.setTrack(ev.target.value));

$("#race-start").addEventListener("click", () => {
  net.raceCmd("start", $("#race-opponent").value, state.trackName || undefined);
});
$("#race-reset").addEventListener("click", () => {
  state.bestLaps.clear();
  net.raceCmd("reset", $("#race-opponent").value);
});

$("#train-start").addEventListener("click", () => {
  const agent = $("#train-agent").value;
  const episodes = parseInt($("#train-episodes").value, 10);
  chart.reset();
  state.training = true;
  $("#train-start").disabled = true;
  net.trainCmd("start", agent, {
    track: state.trackName || undefined,
    warm: $("#train-warm").checked,
    episodes: Number.isFinite(episodes) ? episodes : undefined,
  });
});
$("#train-stop").addEventListener("click", () => {
  net.trainCmd("stop", $("#train-agent").value);
  state.training = false;
  $("#train-start").disabled = false;
});

const trainStats = new Map(); // agent -> latest telemetry

function updateTrainStats(msg) {
  trainStats.set(msg.agent, msg);
  const rows = [...trainStats.values()].map(
    (m) => `<div class="stat-row">
      <span class="dot" style="background:${KIND_COLORS[m.agent] || "#ccc"}"></span>
      <span>${KIND_NAMES[m.agent] || m.agent}</span>
      <span>ep <b>${m.episode}</b></span>
      <span>ret <b>${m.mean_return.toFixed(1)}</b></span>
      <span>ε <b>${m.epsilon.toFixed(2)}</b></span>
      <span>loss <b>${m.loss == null ? "—" : m.loss.toFixed(3)}</b></span>
    </div>`,
  );
  $("#train-stats").innerHTML = rows.join("");
}

// -- boot --------------------------------------------------------------------

setStatus("connecting…", "pill-off");
applyMode("attract");
net.connect();
