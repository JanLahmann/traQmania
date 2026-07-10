// Static SVG rendering of the circuit_spec from the welcome message.
// spec: {n_qubits, n_layers, gates:[{type:'ry_enc'|'ry'|'rz'|'cz', qubit|q0/q1, layer}]}

const WIRE_GAP = 44;
const COL_W = 34;
const LEFT_PAD = 46;
const TOP_PAD = 26;
const BOX_W = 26;
const BOX_H = 22;

const COLORS = {
  wire: "#3a4152",
  box: "#232836",
  boxStroke: "#4a5268",
  enc: "#3b2d73",
  encStroke: "#7a5cff",
  text: "#dfe3ea",
  encText: "#c9baff",
  cz: "#8fa0c9",
  meas: "#1e3a2f",
  measStroke: "#2fbf71",
};

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Greedy column packing: each gate goes in the first free column for the
 *  wires it spans (cz spans the full min..max qubit range). */
function layoutGates(spec) {
  const nextFree = new Array(spec.n_qubits).fill(0);
  const placed = [];
  for (const g of spec.gates) {
    let qs;
    if (g.type === "cz") {
      const lo = Math.min(g.q0, g.q1);
      const hi = Math.max(g.q0, g.q1);
      qs = [];
      for (let q = lo; q <= hi; q++) qs.push(q);
    } else {
      qs = [g.qubit];
    }
    const col = Math.max(...qs.map((q) => nextFree[q]));
    for (const q of qs) nextFree[q] = col + 1;
    placed.push({ gate: g, col, qs });
  }
  return { placed, nCols: Math.max(...nextFree) };
}

function wireY(q) {
  return TOP_PAD + q * WIRE_GAP;
}

function colX(c) {
  return LEFT_PAD + c * COL_W + COL_W / 2;
}

function gateBox(x, y, fill, stroke, label, sub, subColor) {
  const parts = [
    `<rect x="${x - BOX_W / 2}" y="${y - BOX_H / 2}" width="${BOX_W}" height="${BOX_H}"` +
      ` rx="4" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`,
    `<text x="${x}" y="${y + (sub ? -1 : 4)}" text-anchor="middle" font-size="9"` +
      ` fill="${subColor || COLORS.text}" font-weight="600">${esc(label)}</text>`,
  ];
  if (sub) {
    parts.push(
      `<text x="${x}" y="${y + 8.5}" text-anchor="middle" font-size="7"` +
        ` fill="${subColor || COLORS.text}">${esc(sub)}</text>`,
    );
  }
  return parts.join("");
}

/** Build the SVG markup for a circuit_spec. Pure function, no DOM needed. */
export function circuitSvg(spec) {
  const n = spec.n_qubits;
  const { placed, nCols } = layoutGates(spec);
  const width = LEFT_PAD + nCols * COL_W + 58;
  const height = TOP_PAD + (n - 1) * WIRE_GAP + 30;
  const out = [];

  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"` +
      ` viewBox="0 0 ${width} ${height}" role="img" aria-label="Quantum circuit diagram">`,
  );

  // wires + qubit labels + terminal <Z> measurement boxes
  for (let q = 0; q < n; q++) {
    const y = wireY(q);
    out.push(
      `<text x="8" y="${y + 3}" font-size="10" fill="${COLORS.text}">q${q}</text>`,
      `<line x1="${LEFT_PAD - 8}" y1="${y}" x2="${width - 44}" y2="${y}"` +
        ` stroke="${COLORS.wire}" stroke-width="1.2"/>`,
      gateBox(width - 26, y, COLORS.meas, COLORS.measStroke, `⟨Z${q}⟩`, null, "#9fe8c4"),
    );
  }

  // layer separators
  const colsPerLayer = nCols / spec.n_layers;
  for (let l = 1; l < spec.n_layers; l++) {
    const x = LEFT_PAD + l * colsPerLayer * COL_W - 2;
    out.push(
      `<line x1="${x}" y1="${TOP_PAD - 16}" x2="${x}" y2="${height - 12}"` +
        ` stroke="#2a3040" stroke-width="1" stroke-dasharray="3 4"/>`,
    );
  }

  for (const { gate, col } of placed) {
    const x = colX(col);
    if (gate.type === "cz") {
      const y0 = wireY(gate.q0);
      const y1 = wireY(gate.q1);
      out.push(
        `<line x1="${x}" y1="${Math.min(y0, y1)}" x2="${x}" y2="${Math.max(y0, y1)}"` +
          ` stroke="${COLORS.cz}" stroke-width="1.5"/>`,
        `<circle cx="${x}" cy="${y0}" r="3.4" fill="${COLORS.cz}"/>`,
        `<circle cx="${x}" cy="${y1}" r="3.4" fill="${COLORS.cz}"/>`,
      );
    } else if (gate.type === "ry_enc") {
      out.push(gateBox(x, wireY(gate.qubit), COLORS.enc, COLORS.encStroke, "RY", "λx", COLORS.encText));
    } else {
      out.push(
        gateBox(x, wireY(gate.qubit), COLORS.box, COLORS.boxStroke, gate.type.toUpperCase(), "θ"),
      );
    }
  }

  out.push("</svg>");
  return out.join("");
}

/** Render circuit + legend into the given containers (once, static). */
export function renderCircuit(spec, diagramEl, legendEl) {
  diagramEl.innerHTML = circuitSvg(spec);
  if (!legendEl) return;
  const nParams = spec.counts ? (spec.counts.ry || 0) + (spec.counts.rz || 0) : "";
  legendEl.innerHTML = `
    <ul class="circuit-legend">
      <li><span class="lg lg-enc"></span> Encoding gate RY(λ·x): writes an observation feature onto a qubit (re-uploaded every layer)</li>
      <li><span class="lg lg-var"></span> Trainable rotation RY/RZ(θ): the "weights" the agent learns</li>
      <li><span class="lg lg-cz"></span> CZ entangler ring: lets qubits influence each other</li>
      <li><span class="lg lg-meas"></span> ⟨Z⟩ readout: one expectation value per action</li>
    </ul>
    <p class="hint">${spec.n_qubits} qubits × ${spec.n_layers} data re-uploading layers${
      nParams ? ` — ${nParams} trainable rotation angles (plus encoding and output scalings)` : ""
    }.</p>`;
}
