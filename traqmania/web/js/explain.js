// Static explainer content for the Explain panel: sub-tabs with concise,
// accurate copy about the exhibit. The copy is templated on the circuit spec
// (welcome.circuit_spec) so q6/q8/q10 profiles state the right sizes; the
// defaults reproduce the 4-qubit text verbatim.

const RAY_WORDS = { 3: "three", 5: "five", 7: "seven", 9: "nine" };

const sections = ({ n_qubits: n = 4, n_layers: layers = 4, n_params: np = {} } = {}) => [
  {
    id: "what",
    title: "What is this?",
    html: `
      <p><strong>traQmania</strong> is a racing game where the driver is a
      <em>quantum circuit</em>. A tiny ${n}-qubit parameterized circuit reads the
      car's sensors — ${RAY_WORDS[n - 1] || n - 1} lidar rays, speed and heading —
      and its measurement
      results decide whether to steer right, go straight, steer left, or
      brake.</p>
      <p>A classical neural network (MLP) of similar size trains on exactly the
      same game, so you can compare the two approaches head to head — or race
      against either of them yourself.</p>`,
  },
  {
    id: "learn",
    title: "How does it learn?",
    html: `
      <p>Both agents learn with <strong>double DQN</strong>, a reinforcement
      learning algorithm. The agent tries actions, receives rewards (progress
      along the track, lap bonuses, penalties for going off track) and slowly
      learns a <em>Q-function</em>: an estimate of how much future reward each
      action is worth in the current situation.</p>
      <p>Exploration is <strong>ε-greedy</strong>: early in training the agent
      picks mostly random actions (ε near 1), and as ε decays it increasingly
      trusts its own Q-values. "Double" DQN means one network chooses the best
      next action while a periodically synced target network evaluates it —
      this reduces the over-optimism that plain DQN suffers from.</p>
      <p>Watch the Training tab: the learning curve shows the mean return per
      episode climbing as the agent figures out the track.</p>`,
  },
  {
    id: "circuit",
    title: "The quantum circuit",
    html: `
      <p>The Q-function of the quantum agent is a
      <strong>variational quantum circuit</strong> with ${n} qubits and ${layers} layers.
      Each layer first <em>encodes</em> the observation with RY(λ·x) rotations,
      then applies trainable RY/RZ rotations, then entangles neighbouring
      qubits with a ring of CZ gates.</p>
      <p>Re-encoding the input in every layer is called
      <strong>data re-uploading</strong> — it lets even a small circuit
      represent rich, non-linear functions of the input.</p>
      <p>The output is read as the <strong>⟨Z⟩ expectation value</strong> of
      ${n === 4 ? "each qubit" : "each of the first four qubits"}: four numbers
      in [-1, 1], scaled to become the four Q-values.
      The gauges in the Quantum tab show them live.</p>`,
  },
  {
    id: "compare",
    title: "Classical vs quantum",
    html: `
      <p>The classical baseline is a small <strong>MLP</strong> (multi-layer
      perceptron) trained with the same double DQN algorithm, the same rewards
      and the same observations. The quantum circuit has only about
      <strong>${np.total ?? 3 * layers * n + 8} trainable parameters</strong>;
      the MLP is kept comparably
      small.</p>
      <p>To be honest: on a toy task like this the quantum agent has
      <em>no proven advantage</em> — whether quantum models can beat classical
      ones on reinforcement learning problems is open research. What this
      exhibit shows is that a genuinely quantum model <em>can</em> learn a
      control task end to end, and lets you inspect every moving part while it
      does.</p>`,
  },
  {
    id: "try",
    title: "Try it",
    html: `
      <ul>
        <li><strong>Watch</strong> — attract mode: trained agents drive laps on
        their own.</li>
        <li><strong>Train</strong> — start a fresh (or warm-started) training
        run and watch the learning curve grow in the Training tab.</li>
        <li><strong>Race</strong> — drive yourself with the arrow keys or WASD
        (↑/W throttle, ↓/S brake, ←/→ steer) against the quantum or classical
        agent. A game controller works too: plug one in and the left stick
        steers while the triggers give analog throttle and brake.</li>
        <li><strong>Hardware</strong> — run the trained circuit on an IBM
        Quantum backend: either a local noisy simulation of a real device, or
        an actual quantum computer (queue times apply). Watch the hardware lap
        replay as a ghost next to a simulator car, or run a short SPSA
        training sprint directly on the backend.</li>
      </ul>
      <p>Everything else runs a fast <em>simulator</em> of the quantum circuit
      — that is standard practice for training, since today's real quantum
      hardware is too slow and too noisy for millions of training steps. The
      identical circuit executes on real quantum hardware in Hardware
      mode.</p>`,
  },
];

/** Build the explain panel (sub-tab nav + sections) inside `root`.
 *  `spec` is the welcome `circuit_spec` (optional: defaults to 4 qubits). */
export function initExplain(root, spec) {
  const SECTIONS = sections(spec);
  const nav = document.createElement("nav");
  nav.className = "explain-nav";
  const body = document.createElement("div");
  body.className = "explain-body";

  const show = (id) => {
    for (const btn of nav.querySelectorAll("button")) {
      btn.classList.toggle("active", btn.dataset.section === id);
    }
    const section = SECTIONS.find((s) => s.id === id);
    body.innerHTML = `<h2>${section.title}</h2>${section.html}`;
  };

  for (const s of SECTIONS) {
    const btn = document.createElement("button");
    btn.dataset.section = s.id;
    btn.textContent = s.title;
    btn.addEventListener("click", () => show(s.id));
    nav.append(btn);
  }

  root.replaceChildren(nav, body);
  show(SECTIONS[0].id);
}
