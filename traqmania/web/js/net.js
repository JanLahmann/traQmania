// WebSocket client for the traQmania server. Auto ws/wss from location,
// reconnects with exponential backoff (0.5 s -> 8 s), dispatches messages by
// their "type" field to registered handlers.

const BACKOFF_MIN = 500;
const BACKOFF_MAX = 8000;

const handlers = new Map(); // type -> Set<fn>
let ws = null;
let backoff = BACKOFF_MIN;
let closedByUser = false;

function emit(type, msg) {
  const set = handlers.get(type);
  if (!set) return;
  for (const fn of set) {
    try {
      fn(msg);
    } catch (err) {
      console.error(`handler for "${type}" failed`, err);
    }
  }
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function open() {
  ws = new WebSocket(wsUrl());

  ws.onopen = () => {
    backoff = BACKOFF_MIN;
    emit("_open", {});
    send("hello", {});
  };

  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      console.warn("non-JSON ws message ignored");
      return;
    }
    if (msg && typeof msg.type === "string") emit(msg.type, msg);
  };

  ws.onclose = () => {
    ws = null;
    emit("_close", {});
    if (closedByUser) return;
    setTimeout(open, backoff);
    backoff = Math.min(backoff * 2, BACKOFF_MAX);
  };

  ws.onerror = () => {
    if (ws && ws.readyState !== WebSocket.CLOSED) ws.close();
  };
}

/** Register a handler for a server message type (or "_open"/"_close"). */
export function on(type, fn) {
  if (!handlers.has(type)) handlers.set(type, new Set());
  handlers.get(type).add(fn);
  return () => handlers.get(type).delete(fn);
}

/** Connect (idempotent). */
export function connect() {
  closedByUser = false;
  if (!ws) open();
}

export function isConnected() {
  return ws !== null && ws.readyState === WebSocket.OPEN;
}

/** Send a protocol message; silently dropped while disconnected. */
export function send(type, payload = {}) {
  if (!isConnected()) return false;
  ws.send(JSON.stringify({ type, ...payload }));
  return true;
}

// -- typed helpers, one per C->S protocol message ---------------------------

/** `analog` (optional): { steer:[-1,1], throttle:[0,1], brake:[0,1] } — when
 *  present the server uses these instead of the keys bitmask (send keys:0). */
export const sendInput = (keys, analog) => send("input", analog ? { keys, ...analog } : { keys });
export const setMode = (mode) => send("set_mode", { mode });
/** `seed` / `length` (optional, track "random" only): `seed` reproduces a
 *  specific generated track, `length` picks short / medium / long. */
export const setTrack = (track, seed, length) => {
  const msg = { track };
  if (seed !== undefined) msg.seed = seed;
  if (length !== undefined) msg.length = length;
  send("set_track", msg);
};
export const setQubits = (n) => send("qubits", { n });
export const drawTrack = (points) => send("draw_track", { points });
export const setName = (name) => send("set_name", { name });
/** Pick which training's quantum weights drive the agent ("auto" = per-track). */
export const setDriver = (driver) => send("set_driver", { driver });

export function trainCmd(action, agent, opts = {}) {
  const msg = { action, agent };
  if (opts.track !== undefined) msg.track = opts.track;
  if (opts.warm !== undefined) msg.warm = opts.warm;
  if (opts.episodes !== undefined) msg.episodes = opts.episodes;
  return send("train", msg);
}

export function raceCmd(action, opponent, track) {
  const msg = { action, opponent };
  if (track !== undefined) msg.track = track;
  return send("race", msg);
}

/** action: "lap" | "sprint" | "abort"; opts: { backend, iterations, shots }. */
export function hardwareCmd(action, opts = {}) {
  const msg = { action };
  if (opts.backend !== undefined) msg.backend = opts.backend;
  if (opts.iterations !== undefined) msg.iterations = opts.iterations;
  if (opts.shots !== undefined) msg.shots = opts.shots;
  return send("hardware", msg);
}
