// Draw-a-track mode: a freehand overlay on the race stage. One stroke = one
// track: pointer down starts the line, release sends it to the server, which
// rescales/smooths/validates it (errors come back as the usual toast — draw
// again to adjust). Escape or ✕ cancels.

export function initDraw({ button, stage, getTransform, submit }) {
  let overlay = null;

  const close = () => {
    overlay?.remove();
    overlay = null;
    document.removeEventListener("keydown", onKey);
  };

  const onKey = (ev) => {
    if (ev.key === "Escape") close();
  };

  const open = () => {
    if (overlay) return;
    overlay = document.createElement("div");
    overlay.className = "draw-overlay";
    overlay.innerHTML =
      '<div class="draw-hint">Draw one closed loop — release to build the track ' +
      "(Esc cancels)</div>" +
      '<button type="button" class="draw-cancel" aria-label="Cancel drawing">✕</button>';
    const canvas = document.createElement("canvas");
    overlay.prepend(canvas);
    stage.append(overlay);
    overlay.querySelector(".draw-cancel").addEventListener("click", close);
    document.addEventListener("keydown", onKey);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = overlay.clientWidth * dpr;
    canvas.height = overlay.clientHeight * dpr;
    const ctx = canvas.getContext("2d");
    const pts = []; // CSS-pixel stroke points
    let drawing = false;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!pts.length) return;
      ctx.lineWidth = 3 * dpr;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.strokeStyle = "#7a5cff";
      ctx.beginPath();
      ctx.moveTo(pts[0][0] * dpr, pts[0][1] * dpr);
      for (const [x, y] of pts) ctx.lineTo(x * dpr, y * dpr);
      ctx.stroke();
      ctx.beginPath(); // start marker: close the loop near here
      ctx.arc(pts[0][0] * dpr, pts[0][1] * dpr, 6 * dpr, 0, 2 * Math.PI);
      ctx.fillStyle = "#2fbf71";
      ctx.fill();
    };

    canvas.addEventListener("pointerdown", (ev) => {
      drawing = true;
      pts.length = 0;
      canvas.setPointerCapture(ev.pointerId);
      pts.push([ev.offsetX, ev.offsetY]);
      render();
    });
    canvas.addEventListener("pointermove", (ev) => {
      if (!drawing) return;
      const [lx, ly] = pts[pts.length - 1];
      if (Math.hypot(ev.offsetX - lx, ev.offsetY - ly) < 4) return;
      pts.push([ev.offsetX, ev.offsetY]);
      render();
    });
    canvas.addEventListener("pointerup", () => {
      drawing = false;
      if (pts.length < 8) return; // a click, not a stroke: stay in draw mode
      const { s, ox, oy } = getTransform();
      const world = pts.map(([cx, cy]) => [
        (cx * dpr - ox) / s,
        -(cy * dpr - oy) / s,
      ]);
      submit(world);
      close();
    });
  };

  button.addEventListener("click", open);
}
