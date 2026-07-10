// Keyboard -> input bitmask. Arrows + WASD. Sends on change plus a 10 Hz
// keepalive while any key is held. Only active in race mode.

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

/** Install listeners once. `activityCb` is called on any mapped keypress. */
export function initInput(activityCb) {
  onActivity = activityCb || null;
  window.addEventListener("keydown", (ev) => handleKey(ev, true));
  window.addEventListener("keyup", (ev) => handleKey(ev, false));
  window.addEventListener("blur", () => {
    if (mask !== 0) {
      mask = 0;
      push();
    }
  });
}

/** Enable/disable input sending (race mode only). Releases keys on disable. */
export function setInputActive(enabled) {
  if (active === enabled) return;
  active = enabled;
  if (!enabled && mask !== 0) {
    mask = 0;
    push();
  }
}
