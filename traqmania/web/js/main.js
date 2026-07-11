// Boot + wiring: websocket handlers, mode/track UI, lap board, panels.

import * as net from "./net.js";
import { RaceRenderer, KIND_COLORS } from "./race.js";
import { initInput, setInputActive } from "./input.js";
import { QuantumPanel } from "./quantum-panel.js";
import { renderCircuit } from "./circuit.js";
import { TrainingChart, LapChart } from "./charts.js";
import { AttractManager } from "./attract.js";
import { initExplain } from "./explain.js";
import { initHardwarePanel } from "./hardware-panel.js";

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
const lapChart = new LapChart($("#lap-chart"));

const attract = new AttractManager({
  captionEl: $("#attract-caption"),
  onIdle: () => net.setMode("attract"),
});

initExplain($("#panel-explain"));
const hardwarePanel = initHardwarePanel();
initInput(() => attract.notifyActivity(), {
  onGamepadChange: (connected) => {
    $("#gamepad-pill").hidden = !connected;
  },
});

// -- helpers -----------------------------------------------------------------

const KIND_NAMES = { quantum: "Quantum", mlp: "MLP", human: "You", hero: "Hero", pro: "Pro" };

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
  $("#driver-picker").hidden = mode !== "attract";
  setInputActive(mode === "race");
  attract.setMode(mode);
  renderer.setMode(mode);
  $("#evo-caption").hidden = mode !== "evolution";
  if (mode !== "evolution") {
    $("#evo-legend").hidden = true;
    evoLegendKey = "";
  }
  renderEpisodeOverlay();
  if (mode === "train") selectTab("training");
  else if (mode === "hardware") selectTab("hardware");
  else if (mode === "attract" || mode === "evolution") selectTab("quantum");
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
  // Generated tracks are named "random #<seed>"; they map onto the picker's
  // "random" entry, whose label shows the seed so the track is reproducible.
  const isRandom = payload.name.startsWith("random #");
  const randomOpt = sel.querySelector('option[value="random"]');
  if (randomOpt) randomOpt.textContent = isRandom ? `🎲 ${payload.name}` : "🎲 random";
  const value = isRandom ? "random" : payload.name;
  if (sel.value !== value) sel.value = value;
  $("#track-reroll").hidden = !isRandom;
  const seedInput = $("#track-seed");
  seedInput.hidden = !isRandom;
  $("#track-length").hidden = !isRandom;
  if (isRandom) {
    // show the active seed as the placeholder (copy it to save the track);
    // clear any typed value so the next 🎲 press rolls a fresh one
    const m = payload.name.match(/^random #(\d+)/);
    if (m) seedInput.placeholder = m[1];
    seedInput.value = "";
  }
}

// Expert mode (open the page with #expert) reveals the hidden hero driver:
// a model-based racing-line controller, the demo's "perfect drive" ceiling.
const EXPERT = window.location.hash.includes("expert");

/** Populate the Watch-mode driver picker from welcome.drivers/driver. */
function applyDrivers(drivers, current) {
  const sel = $("#driver-select");
  const label = (d) =>
    d === "auto" ? "auto (this track)"
    : d === "hero" ? "hero — racing line"
    : d === "pro" ? "pro — big classical DQN"
    : `${d}-trained`;
  sel.replaceChildren(
    ...(drivers || ["auto"])
      .filter((d) => EXPERT || (d !== "hero" && d !== "pro"))
      .map((d) => {
        const opt = document.createElement("option");
        opt.value = d;
        opt.textContent = label(d);
        return opt;
      }),
  );
  sel.value =
    current && (EXPERT || (current !== "hero" && current !== "pro"))
      ? current : "auto";
}

/** Ask for a generated track: typed seed (empty -> fresh roll) + length. */
function requestRandomTrack() {
  const raw = $("#track-seed").value.trim();
  const seed = /^\d+$/.test(raw) ? parseInt(raw, 10) : undefined;
  const length = $("#track-length").value;
  net.setTrack("random", seed, length === "medium" ? undefined : length);
}

// Circuit-size copy (q6/q8/q10 profiles or a live qubit switch): fix up the
// 4-qubit copy in the captions, the readout hint, the header dropdown and the
// Explain panel. At the default 4 qubits the authored text is reproduced, so
// switching back from a larger circuit restores it.
function applyCircuitSize(spec) {
  attract.setCircuitSpec(spec);
  const n = spec.n_qubits || 4;
  const sel = $("#qubit-select");
  if (sel && sel.value !== String(n)) sel.value = String(n);
  const hint = $("#qubit-hint");
  if (hint) {
    hint.innerHTML =
      n === 4
        ? "Pauli-Z expectation values &lt;Z<sub>a</sub>&gt; of the 4 qubits — one per action."
        : `Pauli-Z expectation values &lt;Z<sub>a</sub>&gt; of the first 4 of ${n} qubits` +
          " — one per action.";
  }
  initExplain($("#panel-explain"), spec);
}

// Observation feature names (welcome.obs_labels): what each encoded input is.
function applyObsLabels(labels) {
  const el = $("#obs-labels");
  if (!el) return;
  el.textContent = Array.isArray(labels) && labels.length ? `Inputs: ${labels.join(" · ")}` : "";
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
    if (car.ghost) {
      // ghosts stay off the board except for a dim "Ghost (best …)" entry
      const t = typeof best === "number" ? best : car.last_lap_time;
      // car.label carries the record's provenance ("best 14.2s · universal")
      const text =
        typeof t === "number" && isFinite(t)
          ? `Ghost (${car.label || `best ${fmtLap(t)}`})`
          : car.label
            ? `Ghost — ${car.label}`
            : "Ghost";
      return `<div class="lap-row ghost">
        <span class="dot ghost-dot"></span>
        <span class="lap-kind">${text}</span>
      </div>`;
    }
    const evo = state.mode === "evolution" && car.label;
    const color = evo ? renderer.stageColor(car.label) : KIND_COLORS[car.kind] || "#ccc";
    const name = evo ? car.label : KIND_NAMES[car.kind] || car.kind;
    return `<div class="lap-row${car.off_track ? " off" : ""}">
      <span class="dot" style="background:${color}"></span>
      <span class="lap-kind">${name}</span>
      <span class="lap-cell">Lap <b>${car.lap}</b></span>
      <span class="lap-cell">Last <b>${fmtLap(car.last_lap_time)}</b></span>
      <span class="lap-cell">Best <b>${fmtLap(best)}</b></span>
    </div>`;
  });
  board.innerHTML = rows.join("");
}

// -- evolution legend ----------------------------------------------------------

let evoLegendKey = "";

function updateEvoLegend(cars) {
  if (state.mode !== "evolution") return;
  const labeled = cars.filter((c) => c.label);
  const key = labeled.map((c) => `${c.label}${c.ghost ? "*" : ""}`).join("|");
  if (key === evoLegendKey) return;
  evoLegendKey = key;
  const el = $("#evo-legend");
  el.hidden = labeled.length === 0;
  el.innerHTML = labeled
    .map((c) => {
      if (c.ghost) {
        return `<div class="legend-row ghost"><span class="dot ghost-dot"></span>${c.label}</div>`;
      }
      const color = renderer.stageColor(c.label);
      return `<div class="legend-row"><span class="dot" style="background:${color}"></span>${c.label}</div>`;
    })
    .join("");
}

// -- episode counter + best-lap banner -----------------------------------------

const episodeByAgent = new Map(); // agent -> latest episode

function renderEpisodeOverlay() {
  const el = $("#episode-overlay");
  if (state.mode !== "train" || episodeByAgent.size === 0) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.innerHTML = [...episodeByAgent.entries()]
    .map(
      ([agent, ep]) =>
        `<div class="ep-line" style="color:${KIND_COLORS[agent] || "#e6e9ef"}">ep ${ep}</div>`,
    )
    .join("");
}

function showBestBanner(lapTime) {
  const el = $("#best-banner");
  el.textContent = `NEW BEST LAP ${lapTime.toFixed(2)}s`;
  el.hidden = false;
  el.classList.remove("banner-in");
  void el.offsetWidth; // retrigger the animation
  el.classList.add("banner-in");
  clearTimeout(showBestBanner._id);
  showBestBanner._id = setTimeout(() => {
    el.hidden = true;
  }, 1600);
}

// -- websocket handlers ------------------------------------------------------

net.on("_open", () => setStatus("connected", "pill-ok"));
net.on("_close", () => setStatus("reconnecting…", "pill-off"));

net.on("welcome", (msg) => {
  state.tracks = msg.tracks || [];
  const sel = $("#track-select");
  const randomOpt = document.createElement("option");
  randomOpt.value = "random";
  randomOpt.textContent = "🎲 random";
  sel.replaceChildren(
    ...state.tracks.map((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      return opt;
    }),
    randomOpt,
  );
  if (msg.circuit_spec) {
    renderCircuit(msg.circuit_spec, $("#circuit-diagram"), $("#circuit-legend"));
    applyCircuitSize(msg.circuit_spec);
  }
  applyObsLabels(msg.obs_labels);
  applyDrivers(msg.drivers, msg.driver);
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
  updateEvoLegend(msg.cars || []);
  if (msg.mode && msg.mode !== state.mode) applyMode(msg.mode); // server-driven mode change
});

net.on("quantum", (msg) => quantumPanel.update(msg));

net.on("hardware_status", (msg) => hardwarePanel.handleStatus(msg));

net.on("telemetry", (msg) => {
  chart.addPoint(msg.agent, msg.episode, msg.mean_return, msg.epsilon);
  if (msg.lap_times !== undefined || msg.best_lap_s != null) {
    lapChart.setAgentData(msg.agent, msg.lap_times, msg.best_lap_s);
  }
  if (typeof msg.episode === "number") {
    episodeByAgent.set(msg.agent, msg.episode);
    renderEpisodeOverlay();
  }
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
    case "new_best_lap":
      if (typeof msg.lap_time === "number") {
        showBestBanner(msg.lap_time);
        if (msg.car_id) {
          const best = state.bestLaps.get(msg.car_id);
          if (best === undefined || msg.lap_time < best) {
            state.bestLaps.set(msg.car_id, msg.lap_time);
          }
        }
      }
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

const MODE_FOR_BUTTON = {
  attract: "attract",
  train: "train",
  evolution: "evolution",
  race: "race",
  hardware: "hardware",
};

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

// Picking "random" (and each reroll click) asks the server for a generated
// track — a fresh roll, or a specific one when a seed is typed; the answering
// track payload carries the seed in its name. Changing length regenerates.
$("#track-select").addEventListener("change", (ev) => {
  if (ev.target.value === "random") requestRandomTrack();
  else net.setTrack(ev.target.value);
});
$("#track-reroll").addEventListener("click", requestRandomTrack);
$("#track-seed").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") requestRandomTrack();
});
$("#track-length").addEventListener("change", requestRandomTrack);

$("#qubit-select").addEventListener("change", (ev) => {
  net.setQubits(parseInt(ev.target.value, 10)); // server answers with a fresh welcome
});

$("#driver-select").addEventListener("change", (ev) => {
  net.setDriver(ev.target.value); // server rebuilds the attract car + re-welcomes
});

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
  lapChart.reset();
  episodeByAgent.clear();
  renderEpisodeOverlay();
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
