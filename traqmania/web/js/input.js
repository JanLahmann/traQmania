// Keyboard -> input bitmask (arrows + WASD), plus Gamepad API analog control.
// Keyboard sends on change plus a 10 Hz keepalive while any key is held.
// A connected gamepad is polled at ~30 Hz while race mode is active: left
// stick X -> steer (deadzone 0.15), right trigger -> throttle (A button too),
// left trigger -> brake. Analog messages go out at ~15 Hz plus immediately on
// significant change, with keys:0 so the server uses the analog fields.
// Only active in race mode; keyboard fallback is unchanged.

import { sendInput } from "./net.js";

export const KEY_THROTTLE = 1;
export const KEY_BRAKE = 2;
export const KEY_LEFT = 4;
export const KEY_RIGHT = 8;

const KEYMAP = {
  ArrowUp: KEY_THROTTLE,
  KeyW: KEY_THROTTLE,
  ArrowDown: KEY_BRAKE,
  KeyS: KEY_BRAKE,
  ArrowLeft: KEY_LEFT,
  KeyA: KEY_LEFT,
  ArrowRight: KEY_RIGHT,
  KeyD: KEY_RIGHT,
};

let mask = 0;
let active = false;
let keepaliveId = null;
let onActivity = null;

function push() {
  sendInput(mask);
  if (mask !== 0 && keepaliveId === null) {
    keepaliveId = setInterval(() => sendInput(mask), 100);
  } else if (mask === 0 && keepaliveId !== null) {
    clearInterval(keepaliveId);
    keepaliveId = null;
  }
}

function handleKey(ev, down) {
  const bit = KEYMAP[ev.code];
  if (bit === undefined) return;
  if (onActivity) onActivity();
  if (!active) return;
  ev.preventDefault();
  const next = down ? mask | bit : mask & ~bit;
  if (next !== mask) {
    mask = next;
    push();
  }
}

// -- gamepad -----------------------------------------------------------------

const PAD_DEADZONE = 0.15;
const PAD_POLL_MS = 33; // ~30 Hz poll
const PAD_SEND_MS = 66; // ~15 Hz steady send rate
const PAD_EPS = 0.05; // "significant change" threshold per channel

let padIndex = null;
let padPollId = null;
let padConnected = false;
let onGamepadChange = null;
let lastAnalog = { steer: 0, throttle: 0, brake: 0 };
let lastAnalogSentAt = 0;

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

/** Returns the first connected gamepad, or null. navigator.getGamepads is
 *  guarded — environments without the Gamepad API just never see a pad. */
function getPad() {
  if (typeof navigator === "undefined" || typeof navigator.getGamepads !== "function") {
    return null;
  }
  const pads = navigator.getGamepads() || [];
  if (padIndex !== null && pads[padIndex] && pads[padIndex].connected) {
    return pads[padIndex];
  }
  for (const p of pads) {
    if (p && p.connected) {
      padIndex = p.index;
      return p;
    }
  }
  padIndex = null;
  return null;
}

function readAnalog(pad) {
  let steer = pad.axes.length > 0 ? pad.axes[0] : 0; // left stick X
  if (Math.abs(steer) < PAD_DEADZONE) {
    steer = 0;
  } else {
    // rescale so the range past the deadzone maps smoothly onto [-1, 1]
    steer = (Math.sign(steer) * (Math.abs(steer) - PAD_DEADZONE)) / (1 - PAD_DEADZONE);
  }
  let throttle = pad.buttons[7] ? pad.buttons[7].value : 0; // right trigger
  if (pad.buttons[0] && pad.buttons[0].pressed) throttle = 1; // A button
  const brake = pad.buttons[6] ? pad.buttons[6].value : 0; // left trigger
  return {
    steer: clamp(steer, -1, 1),
    throttle: clamp(throttle, 0, 1),
    brake: clamp(brake, 0, 1),
  };
}

function pollPad() {
  const pad = getPad();
  if (!pad) {
    updatePadState(); // pad vanished without a disconnect event
    return;
  }
  const a = readAnalog(pad);
  const changed =
    Math.abs(a.steer - lastAnalog.steer) > PAD_EPS ||
    Math.abs(a.throttle - lastAnalog.throttle) > PAD_EPS ||
    Math.abs(a.brake - lastAnalog.brake) > PAD_EPS;
  if (changed && onActivity) onActivity();
  const now = performance.now();
  if (changed || now - lastAnalogSentAt >= PAD_SEND_MS) {
    if (sendInput(0, a)) {
      lastAnalog = a;
      lastAnalogSentAt = now;
    }
  }
}

function updatePadState() {
  const connected = getPad() !== null;
  if (connected !== padConnected) {
    padConnected = connected;
    if (onGamepadChange) onGamepadChange(connected);
  }
  const shouldPoll = connected && active;
  if (shouldPoll && padPollId === null) {
    padPollId = setInterval(pollPad, PAD_POLL_MS);
  } else if (!shouldPoll && padPollId !== null) {
    clearInterval(padPollId);
    padPollId = null;
    // release the analog controls so the car doesn't keep driving
    if (lastAnalog.steer !== 0 || lastAnalog.throttle !== 0 || lastAnalog.brake !== 0) {
      sendInput(0, { steer: 0, throttle: 0, brake: 0 });
    }
    lastAnalog = { steer: 0, throttle: 0, brake: 0 };
  }
}

/** Install listeners once. `activityCb` is called on any mapped keypress or
 *  significant gamepad input. `opts.onGamepadChange(connected)` fires when a
 *  controller connects/disconnects. */
export function initInput(activityCb, opts = {}) {
  onActivity = activityCb || null;
  onGamepadChange = opts.onGamepadChange || null;
  window.addEventListener("keydown", (ev) => handleKey(ev, true));
  window.addEventListener("keyup", (ev) => handleKey(ev, false));
  window.addEventListener("blur", () => {
    if (mask !== 0) {
      mask = 0;
      push();
    }
  });
  window.addEventListener("gamepadconnected", updatePadState);
  window.addEventListener("gamepaddisconnected", () => {
    padIndex = null;
    updatePadState();
  });
  updatePadState(); // pick up a pad that was connected before page load
}

/** Enable/disable input sending (race mode only). Releases keys on disable. */
export function setInputActive(enabled) {
  if (active === enabled) return;
  active = enabled;
  if (!enabled && mask !== 0) {
    mask = 0;
    push();
  }
  updatePadState();
}
