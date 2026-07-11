// Delayed hover tooltips: any element with a data-tip attribute gets a styled
// explanation bubble after a short hover dwell (so casual mouse travel stays
// quiet). One document-level delegate handles everything, including elements
// created later; pointerdown (tap/click) toggles the tip too, for touch
// kiosks where hover does not exist.

const SHOW_DELAY_MS = 550;

export function initTooltips() {
  const tip = document.createElement("div");
  tip.className = "ui-tip";
  tip.hidden = true;
  document.body.append(tip);

  let timer = null;
  let current = null;

  const hide = () => {
    clearTimeout(timer);
    timer = null;
    current = null;
    tip.hidden = true;
  };

  const show = (el) => {
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    let x = r.left + r.width / 2 - t.width / 2;
    x = Math.max(8, Math.min(x, window.innerWidth - t.width - 8));
    let y = r.bottom + 8;
    if (y + t.height > window.innerHeight - 8) y = r.top - t.height - 8;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  };

  document.addEventListener("pointerover", (ev) => {
    const el = ev.target.closest("[data-tip]");
    if (el === current) return;
    hide();
    if (!el) return;
    current = el;
    timer = setTimeout(() => show(el), SHOW_DELAY_MS);
  });

  document.addEventListener("pointerdown", (ev) => {
    const el = ev.target.closest("[data-tip]");
    if (el && tip.hidden && ev.pointerType !== "mouse") {
      current = el;
      show(el); // touch: tap shows immediately (no hover exists)
    } else {
      hide();
    }
  });

  window.addEventListener("scroll", hide, true);
}
