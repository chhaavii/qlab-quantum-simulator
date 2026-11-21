"use strict";

/* ---------------------------------------------------------------- helpers */
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const SUP = {
  "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
  5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹",
};
const sup = (text) => [...text].map((char) => SUP[char] ?? char).join("");

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude === 0) return (0).toFixed(digits);
  if (magnitude < 1e-3 || magnitude >= 1e5) {
    const [mantissa, exponent] = value.toExponential(2).split("e");
    return `${mantissa}×10${sup(String(Number(exponent)))}`;
  }
  return value.toFixed(digits);
}

const percent = (value, digits = 1) => `${(100 * value).toFixed(digits)}%`;

async function api(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  let data = {};
  try {
    data = await response.json();
  } catch {
    /* a non-JSON error page: fall through to the status message */
  }
  if (!response.ok) throw new Error(data.error || `request failed (${response.status})`);
  return data;
}

/* ------------------------------------------------------------- components */
function renderBars(container, items, options = {}) {
  container.innerHTML = "";
  if (!items.length) return;
  const max = Math.max(options.max ?? 0, ...items.map((item) => item.value), 1e-12);
  const fragment = document.createDocumentFragment();

  for (const item of items) {
    const column = document.createElement("div");
    column.className = "bar-col";
    if (item.title) column.title = item.title;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    // Keep a hairline for states that carry vanishingly little probability.
    const height = item.value > 1e-9 ? Math.max(2, (item.value / max) * 100) : 0;
    fill.style.height = `${height}%`;
    if (item.color) fill.style.background = item.color;
    track.appendChild(fill);

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = item.label;

    column.append(track, label);
    fragment.appendChild(column);
  }
  container.appendChild(fragment);
}

function renderStats(container, items) {
  container.innerHTML = "";
  for (const item of items) {
    const box = document.createElement("div");
    box.className = `stat ${item.tone ?? ""}`.trim();

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = item.label;

    const value = document.createElement("div");
    value.className = "value";
    value.textContent = item.value;

    box.append(label, value);

    if (item.sub) {
      const sub = document.createElement("div");
      sub.className = "sub";
      sub.textContent = item.sub;
      box.appendChild(sub);
    }
    if (typeof item.meter === "number") {
      const meter = document.createElement("div");
      meter.className = "meter";
      const bar = document.createElement("span");
      bar.style.width = `${Math.max(0, Math.min(1, item.meter)) * 100}%`;
      meter.appendChild(bar);
      box.appendChild(meter);
    }
    container.appendChild(box);
  }
}

function lineChart(points, options = {}) {
  const width = 360;
  const height = 200;
  const pad = { top: 18, right: 16, bottom: 34, left: 44 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  const xs = points.map((point) => point.x);
  const xMin = options.xMin ?? Math.min(...xs);
  const xMax = options.xMax ?? Math.max(...xs);
  const yMax = options.yMax ?? 1;

  const scaleX = (x) =>
    pad.left + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotWidth);
  const scaleY = (y) => pad.top + plotHeight - (y / yMax) * plotHeight;

  let markup = "";

  for (let step = 0; step <= 4; step += 1) {
    const value = (yMax * step) / 4;
    const y = scaleY(value);
    markup += `<line class="grid" x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(1)}"/>`;
    markup += `<text class="tick" x="${pad.left - 7}" y="${(y + 3).toFixed(1)}" text-anchor="end">${value.toFixed(2)}</text>`;
  }
  for (let step = 0; step <= 4; step += 1) {
    const value = xMin + ((xMax - xMin) * step) / 4;
    const x = scaleX(value);
    const text = options.xFormat ? options.xFormat(value) : value.toFixed(1);
    markup += `<text class="tick" x="${x.toFixed(1)}" y="${pad.top + plotHeight + 14}" text-anchor="middle">${text}</text>`;
  }

  markup += `<line class="axis" x1="${pad.left}" y1="${pad.top + plotHeight}" x2="${width - pad.right}" y2="${pad.top + plotHeight}"/>`;
  markup += `<line class="axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${pad.top + plotHeight}"/>`;

  if (options.marker) {
    const x = scaleX(options.marker.x);
    markup += `<line class="grid" x1="${x.toFixed(1)}" y1="${pad.top}" x2="${x.toFixed(1)}" y2="${pad.top + plotHeight}" stroke="rgba(167,139,250,0.5)"/>`;
  }

  const path = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${scaleX(point.x).toFixed(2)} ${scaleY(point.y).toFixed(2)}`)
    .join(" ");
  markup += `<path class="series" d="${path}"/>`;

  if (options.marker) {
    const x = scaleX(options.marker.x);
    const y = scaleY(options.marker.y);
    markup += `<circle class="marker" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4"/>`;
  }

  if (options.xLabel) {
    markup += `<text class="axis-label" x="${pad.left + plotWidth / 2}" y="${height - 3}" text-anchor="middle">${options.xLabel}</text>`;
  }
  if (options.yLabel) {
    const centre = pad.top + plotHeight / 2;
    markup += `<text class="axis-label" x="12" y="${centre}" text-anchor="middle" transform="rotate(-90 12 ${centre})">${options.yLabel}</text>`;
  }

  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${options.label ?? "chart"}">${markup}</svg>`;
}

const phaseColour = (phase) =>
  `hsl(${(190 + 90 * ((phase + Math.PI) / (2 * Math.PI))).toFixed(0)} 85% 62%)`;

/* ============================================================== circuit */
const circuit = { n: 2, gates: [] };
const TWO_QUBIT = new Set(["CNOT", "CZ", "SWAP"]);

function fillQubitSelect(select, n, fallback = 0) {
  let previous = Number(select.value);
  if (!Number.isInteger(previous) || previous < 0 || previous >= n) previous = fallback;
  select.innerHTML = "";
  for (let qubit = 0; qubit < n; qubit += 1) {
    const option = document.createElement("option");
    option.value = String(qubit);
    option.textContent = `q${qubit}`;
    select.appendChild(option);
  }
  select.value = String(Math.min(previous, n - 1));
}

function syncCircuitControls() {
  fillQubitSelect($("#g-target"), circuit.n);
  fillQubitSelect($("#g-control"), circuit.n);
  fillQubitSelect($("#g-ctarget"), circuit.n, Math.min(1, circuit.n - 1));
  const single = circuit.n < 2;
  $$("#two-qubit-slots select, #two-qubit-gates button").forEach((element) => {
    element.disabled = single;
  });
}

function describeGate(gate) {
  const [a, b] = gate.qubits;
  if (gate.qubits.length === 1) return `${gate.name} q${a}`;
  if (gate.name === "SWAP") return `SWAP q${a} ↔ q${b}`;
  return `${gate.name} q${a} → q${b}`;
}

function renderCircuitList() {
  const list = $("#gate-list");
  const count = circuit.gates.length;
  $("#gate-count").textContent = `${count} gate${count === 1 ? "" : "s"}`;
  list.innerHTML = "";

  if (!count) {
    const item = document.createElement("li");
    item.className = "empty";
    item.textContent = `No gates yet — the register is in |${"0".repeat(circuit.n)}⟩.`;
    list.appendChild(item);
    return;
  }

  circuit.gates.forEach((gate, index) => {
    const item = document.createElement("li");
    const text = document.createElement("span");
    text.textContent = `${index + 1}. ${describeGate(gate)}`;

    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.title = "remove this gate";
    remove.addEventListener("click", () => {
      circuit.gates.splice(index, 1);
      refreshCircuit();
    });

    item.append(text, remove);
    list.appendChild(item);
  });
}

function renderCircuitState(data) {
  renderBars(
    $("#amp-chart"),
    data.amplitudes.map((amplitude) => ({
      label: `|${amplitude.bits}⟩`,
      value: amplitude.prob,
      color: phaseColour(amplitude.phase),
      title: `|${amplitude.bits}⟩  =  ${formatNumber(amplitude.re, 4)} ${amplitude.im < 0 ? "−" : "+"} ${formatNumber(Math.abs(amplitude.im), 4)}i\nP = ${percent(amplitude.prob, 2)}`,
    })),
    { max: 1 },
  );

  const entries = Object.entries(data.counts);
  const largest = Math.max(...entries.map(([, shots]) => shots), 1);
  renderBars(
    $("#shot-chart"),
    entries.map(([bits, shots]) => ({
      label: `|${bits}⟩`,
      value: shots / largest,
      title: `${shots} of ${data.shots} shots`,
    })),
  );
  $("#shots-hint").textContent = `${data.shots} projective measurements of the whole ${data.qubits}-qubit register.`;
}

async function refreshCircuit() {
  renderCircuitList();
  const errorTarget = $("#circuit-error");
  try {
    const data = await api("/api/circuit", { qubits: circuit.n, gates: circuit.gates });
    errorTarget.textContent = "";
    renderCircuitState(data);
  } catch (error) {
    errorTarget.textContent = error.message;
  }
}

function addGate(name) {
  const qubits = TWO_QUBIT.has(name)
    ? [Number($("#g-control").value), Number($("#g-ctarget").value)]
    : [Number($("#g-target").value)];

  if (qubits.length === 2 && qubits[0] === qubits[1]) {
    $("#circuit-error").textContent = `${name} needs two different qubits`;
    return;
  }
  circuit.gates.push({ name, qubits });
  refreshCircuit();
}

function initCircuit() {
  syncCircuitControls();
  refreshCircuit();

  $("#q-count").addEventListener("change", (event) => {
    circuit.n = Number(event.target.value);
    circuit.gates = circuit.gates.filter((gate) => gate.qubits.every((q) => q < circuit.n));
    syncCircuitControls();
    refreshCircuit();
  });

  $$("#single-gates button").forEach((button) => {
    button.addEventListener("click", () => addGate(button.dataset.gate));
  });
  $$("#two-qubit-gates button").forEach((button) => {
    button.addEventListener("click", () => addGate(button.dataset.gate));
  });

  $("#undo-btn").addEventListener("click", () => {
    circuit.gates.pop();
    refreshCircuit();
  });
  $("#clear-btn").addEventListener("click", () => {
    circuit.gates = [];
    $("#circuit-error").textContent = "";
    refreshCircuit();
  });
}

/* =============================================================== grover */
function syncGroverMarked() {
  const n = Number($("#grover-n").value);
  const select = $("#grover-marked");
  const previous = Number(select.value);
  select.innerHTML = "";
  for (let index = 0; index < 2 ** n; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `|${index.toString(2).padStart(n, "0")}⟩  (${index})`;
    select.appendChild(option);
  }
  select.value = String(Number.isInteger(previous) && previous < 2 ** n ? previous : 0);
}

async function runGrover() {
  const button = $("#grover-run");
  const errorTarget = $("#grover-error");
  button.disabled = true;
  try {
    const data = await api("/api/grover", {
      qubits: Number($("#grover-n").value),
      marked: Number($("#grover-marked").value),
    });
    errorTarget.textContent = "";

    renderStats($("#grover-stats"), [
      {
        label: "Success probability",
        value: percent(data.success_probability, 1),
        sub: `${data.qubits} qubits, ${2 ** data.qubits} items`,
        tone: "good",
        meter: data.success_probability,
      },
      {
        label: "Iterations used",
        value: String(data.iterations),
        sub: "peak of the amplification curve",
      },
      {
        label: "Classical guessing",
        value: percent(data.classical_probability, 1),
        sub: "uniform sampling",
        tone: "warn",
        meter: data.classical_probability,
      },
    ]);

    $("#grover-curve").innerHTML = lineChart(
      data.curve.map((point) => ({ x: point.iterations, y: point.prob })),
      {
        marker: { x: data.iterations, y: data.success_probability },
        xFormat: (value) => String(Math.round(value)),
        xLabel: "iterations",
        yLabel: "success",
        label: "success probability against iteration count",
      },
    );

    const markedBits = data.marked.toString(2).padStart(data.qubits, "0");
    renderBars(
      $("#grover-bars"),
      data.probabilities.map((entry) => ({
        label: `|${entry.bits}⟩`,
        value: entry.prob,
        color:
          entry.bits === markedBits
            ? "linear-gradient(180deg, #a78bfa, rgba(167,139,250,0.5))"
            : undefined,
        title: `|${entry.bits}⟩  P = ${percent(entry.prob, 2)}${entry.bits === markedBits ? "  (marked)" : ""}`,
      })),
      { max: 1 },
    );
  } catch (error) {
    errorTarget.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function initGrover() {
  syncGroverMarked();
  $("#grover-n").addEventListener("change", () => {
    syncGroverMarked();
    runGrover();
  });
  $("#grover-marked").addEventListener("change", runGrover);
  $("#grover-run").addEventListener("click", runGrover);
  runGrover();
}

/* ========================================================= entanglement */
let entanglementToken = 0;

async function refreshEntanglement() {
  const token = ++entanglementToken;
  const thetaOverPi = Number($("#theta").value);
  $("#theta-out").textContent = thetaOverPi.toFixed(3);

  try {
    const data = await api("/api/entanglement", { theta: thetaOverPi * Math.PI });
    if (token !== entanglementToken) return;
    $("#ent-error").textContent = "";

    renderStats($("#ent-stats"), [
      {
        label: "Entropy of one half",
        value: data.entropy.toFixed(4),
        sub: "von Neumann, bits",
        tone: "accent",
        meter: data.entropy,
      },
      {
        label: "Concurrence",
        value: data.concurrence.toFixed(4),
        sub: "2·s₀·s₁, pure state",
        tone: "good",
        meter: data.concurrence,
      },
      {
        label: "Negativity",
        value: data.negativity.toFixed(4),
        sub: "(‖ρᵀᴮ‖₁ − 1)/2",
        meter: data.negativity / 0.5,
      },
    ]);

    renderBars(
      $("#ent-bars"),
      data.probabilities.map((entry) => ({
        label: `|${entry.bits}⟩`,
        value: entry.prob,
        title: `|${entry.bits}⟩  P = ${percent(entry.prob, 2)}`,
      })),
      { max: 1 },
    );
  } catch (error) {
    if (token === entanglementToken) $("#ent-error").textContent = error.message;
  }
}

function initEntanglement() {
  $("#theta").addEventListener("input", refreshEntanglement);
  refreshEntanglement();
}

/* ================================================================= bb84 */
function renderBitStrips(data) {
  const container = $("#bb84-bits-view");
  container.innerHTML = "";
  if (!data.alice_preview) {
    container.textContent = "no sifted bits this run";
    return;
  }

  const rows = [
    ["Alice", data.alice_preview],
    ["Bob", data.bob_preview],
  ];
  for (const [who, bits] of rows) {
    const row = document.createElement("div");
    row.className = "bit-row";
    const label = document.createElement("span");
    label.className = "who";
    label.textContent = who;
    const digits = document.createElement("div");
    digits.className = "bits";
    [...bits].forEach((bit, index) => {
      const cell = document.createElement("span");
      cell.textContent = bit;
      if (data.mismatch_preview.includes(index)) cell.classList.add("bad");
      digits.appendChild(cell);
    });
    row.append(label, digits);
    container.appendChild(row);
  }
}

async function runBB84() {
  const button = $("#bb84-run");
  const errorTarget = $("#bb84-error");
  button.disabled = true;
  try {
    const data = await api("/api/bb84", {
      n_bits: Number($("#bb84-bits").value),
      eavesdropper: $("#bb84-eve").checked,
    });
    errorTarget.textContent = "";

    const tampered = data.qber > data.threshold;
    renderStats($("#bb84-stats"), [
      {
        label: "Quantum bit error rate",
        value: percent(data.qber, 1),
        sub: `${data.errors} errors in ${data.sifted_length} sifted bits`,
        tone: tampered ? "warn" : "good",
        meter: data.qber / 0.25,
      },
      {
        label: "Sifted key length",
        value: String(data.sifted_length),
        sub: `from ${data.n_bits} raw bits`,
      },
      {
        label: "Decision",
        value: tampered ? "abort" : "keep key",
        sub: `threshold ${percent(data.threshold, 0)}`,
        tone: tampered ? "warn" : "good",
      },
    ]);

    renderBitStrips(data);
  } catch (error) {
    errorTarget.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function initBB84() {
  $("#bb84-run").addEventListener("click", runBB84);
  $("#bb84-eve").addEventListener("change", runBB84);
  runBB84();
}

/* ============================================================== gravity */
let gravityToken = 0;

function readGravity() {
  return {
    mass: 10 ** Number($("#g-mass").value),
    distance: Number($("#g-distance").value) * 1e-3,
    separation: Number($("#g-separation").value) * 1e-6,
    coherence_time: Number($("#g-coherence").value),
  };
}

function showGravityInputs(settings) {
  $("#g-mass-out").textContent = `${formatNumber(settings.mass, 2)} kg`;
  $("#g-distance-out").textContent = `${(settings.distance * 1e3).toFixed(2)} mm`;
  $("#g-separation-out").textContent = `${(settings.separation * 1e6).toFixed(0)} µm`;
  $("#g-coherence-out").textContent = `${settings.coherence_time.toFixed(2)} s`;
}

async function refreshGravity() {
  const token = ++gravityToken;
  const settings = readGravity();
  showGravityInputs(settings);

  try {
    const data = await api("/api/gravity", settings);
    if (token !== gravityToken) return;
    $("#gravity-error").textContent = "";

    renderStats($("#gravity-stats"), [
      {
        label: "Peak concurrence",
        value: data.best.concurrence.toFixed(4),
        sub: "at the optimal interaction time",
        tone: "good",
        meter: data.best.concurrence,
      },
      {
        label: "Optimal time t_opt",
        value: `${data.best.time.toExponential(2)} s`,
        sub: "2/k · arctan(k·T₂/2)",
      },
      {
        label: "Phase at t_opt",
        value: data.best.phase.toFixed(4),
        sub: `rate ${formatNumber(data.phase_rate, 3)} rad/s`,
        tone: "accent",
      },
    ]);

    $("#gravity-chart").innerHTML = lineChart(
      data.curve.map((point) => ({ x: point.t, y: point.concurrence })),
      {
        marker: { x: data.best.time, y: data.best.concurrence },
        xFormat: (value) => value.toFixed(2),
        xLabel: "time t (s)",
        yLabel: "concurrence",
        label: "concurrence against interaction time",
      },
    );

    $("#gravity-hint").textContent =
      `Curve plotted over 0 to ${data.span.toFixed(2)} s (3 × T₂). ` +
      `Exponential decay × |sin(Δφ/2)| peaks at the marked point.`;
  } catch (error) {
    if (token === gravityToken) $("#gravity-error").textContent = error.message;
  }
}

function initGravity() {
  $$("#gravity input[type=range]").forEach((slider) => {
    slider.addEventListener("input", refreshGravity);
  });
  refreshGravity();
}

/* ================================================================== boot */
initCircuit();
initGrover();
initEntanglement();
initBB84();
initGravity();
