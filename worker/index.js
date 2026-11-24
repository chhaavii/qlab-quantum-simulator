/* QLab on Cloudflare Workers.
 *
 * The site is the same single-page interface as web/static, and this Worker
 * answers the same five JSON endpoints as web/app.py.  It is a JavaScript port
 * of the qlab package (state vector, Grover, entanglement measures, BB84 and
 * the BMV gravity model), so the hosted demo needs no Python runtime.
 */

const G = 6.6743e-11;
const HBAR = 1.054571817e-34;
const MAX_CIRCUIT_QUBITS = 8;
const SHOTS = 1024;

/* --------------------------------------------------------------- complex */
const cmul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const cadd = (a, b) => [a[0] + b[0], a[1] + b[1]];

const SQ = 1 / Math.SQRT2;
const GATES = {
  H: [
    [[SQ, 0], [SQ, 0]],
    [[SQ, 0], [-SQ, 0]],
  ],
  X: [
    [[0, 0], [1, 0]],
    [[1, 0], [0, 0]],
  ],
  Y: [
    [[0, 0], [0, -1]],
    [[0, 1], [0, 0]],
  ],
  Z: [
    [[1, 0], [0, 0]],
    [[0, 0], [-1, 0]],
  ],
  S: [
    [[1, 0], [0, 0]],
    [[0, 0], [0, 1]],
  ],
  T: [
    [[1, 0], [0, 0]],
    [[0, 0], [Math.cos(Math.PI / 4), Math.sin(Math.PI / 4)]],
  ],
};

/* --------------------------------------------------------------- register */
class QReg {
  constructor(n) {
    this.n = n;
    this.dim = 1 << n;
    this.re = new Float64Array(this.dim);
    this.im = new Float64Array(this.dim);
    this.re[0] = 1;
  }

  mask(qubit) {
    return 1 << (this.n - 1 - qubit);
  }

  apply1(U, qubit) {
    const mask = this.mask(qubit);
    for (let i = 0; i < this.dim; i += 1) {
      if (i & mask) continue;
      const j = i | mask;
      const a = [this.re[i], this.im[i]];
      const b = [this.re[j], this.im[j]];
      const top = cadd(cmul(U[0][0], a), cmul(U[0][1], b));
      const bottom = cadd(cmul(U[1][0], a), cmul(U[1][1], b));
      this.re[i] = top[0];
      this.im[i] = top[1];
      this.re[j] = bottom[0];
      this.im[j] = bottom[1];
    }
  }

  controlled(U, controls, target) {
    const mask = this.mask(target);
    const controlMasks = controls.map((qubit) => this.mask(qubit));
    for (let i = 0; i < this.dim; i += 1) {
      if (i & mask) continue;
      let active = true;
      for (const bit of controlMasks) {
        if (!(i & bit)) {
          active = false;
          break;
        }
      }
      if (!active) continue;
      const j = i | mask;
      const a = [this.re[i], this.im[i]];
      const b = [this.re[j], this.im[j]];
      const top = cadd(cmul(U[0][0], a), cmul(U[0][1], b));
      const bottom = cadd(cmul(U[1][0], a), cmul(U[1][1], b));
      this.re[i] = top[0];
      this.im[i] = top[1];
      this.re[j] = bottom[0];
      this.im[j] = bottom[1];
    }
  }

  h(q) {
    this.apply1(GATES.H, q);
  }
  x(q) {
    this.apply1(GATES.X, q);
  }
  cnot(control, target) {
    this.controlled(GATES.X, [control], target);
  }
  cz(control, target) {
    this.controlled(GATES.Z, [control], target);
  }
  swap(a, b) {
    if (a === b) return;
    this.cnot(a, b);
    this.cnot(b, a);
    this.cnot(a, b);
  }

  probabilities() {
    const probs = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i += 1) {
      probs[i] = this.re[i] * this.re[i] + this.im[i] * this.im[i];
    }
    return probs;
  }

  norm() {
    let total = 0;
    for (let i = 0; i < this.dim; i += 1) {
      total += this.re[i] * this.re[i] + this.im[i] * this.im[i];
    }
    return Math.sqrt(total);
  }

  sample(shots) {
    const probs = this.probabilities();
    const cumulative = new Float64Array(this.dim);
    let running = 0;
    for (let i = 0; i < this.dim; i += 1) {
      running += probs[i];
      cumulative[i] = running;
    }
    const counts = {};
    for (let shot = 0; shot < shots; shot += 1) {
      const draw = Math.random() * running;
      let index = 0;
      while (index < this.dim - 1 && cumulative[index] < draw) index += 1;
      const key = index.toString(2).padStart(this.n, "0");
      counts[key] = (counts[key] ?? 0) + 1;
    }
    const ordered = {};
    for (const key of Object.keys(counts).sort()) ordered[key] = counts[key];
    return ordered;
  }
}

const bitsOf = (index, n) => index.toString(2).padStart(n, "0");

function amplitudesOf(reg, threshold = 1e-9) {
  const out = [];
  for (let index = 0; index < reg.dim; index += 1) {
    const re = reg.re[index];
    const im = reg.im[index];
    const prob = re * re + im * im;
    if (prob < threshold) continue;
    out.push({
      index,
      bits: bitsOf(index, reg.n),
      re,
      im,
      prob,
      phase: Math.atan2(im, re),
    });
  }
  return out;
}

function probabilitiesOf(reg) {
  const probs = reg.probabilities();
  const out = [];
  for (let index = 0; index < reg.dim; index += 1) {
    out.push({ bits: bitsOf(index, reg.n), prob: probs[index] });
  }
  return out;
}

/* ------------------------------------------------------------- validation */
function integer(payload, key, fallback, problem) {
  const raw = payload[key] ?? fallback;
  const value = Number(raw);
  if (!Number.isInteger(value)) throw new Error(problem);
  return value;
}

function number(payload, key, fallback) {
  const raw = payload[key] ?? fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`${key} must be a finite number`);
  return value;
}

function qubitsFor(spec, n, count, name) {
  const raw = spec.qubits ?? [];
  if (!Array.isArray(raw) || raw.length !== count) {
    throw new Error(`${name} acts on ${count} qubit(s)`);
  }
  const qubits = raw.map((q) => Number(q));
  if (qubits.some((q) => !Number.isInteger(q))) {
    throw new Error(`${name} needs integer qubit indices`);
  }
  if (new Set(qubits).size !== qubits.length) {
    throw new Error(`${name} needs distinct qubits`);
  }
  for (const q of qubits) {
    if (q < 0 || q >= n) throw new Error(`qubit ${q} is outside a ${n}-qubit register`);
  }
  return qubits;
}

/* ------------------------------------------------------------ /api/circuit */
function runCircuit(payload) {
  const n = integer(payload, "qubits", 2, "qubits must be an integer");
  if (n < 1 || n > MAX_CIRCUIT_QUBITS) {
    throw new Error(`a circuit can have 1 to ${MAX_CIRCUIT_QUBITS} qubits`);
  }
  const specs = payload.gates ?? [];
  if (!Array.isArray(specs)) throw new Error("gates must be a list");

  const reg = new QReg(n);
  for (const spec of specs) {
    if (spec === null || typeof spec !== "object" || Array.isArray(spec)) {
      throw new Error("each gate must be an object");
    }
    const name = String(spec.name ?? "").toUpperCase();
    if (name in GATES) {
      const [target] = qubitsFor(spec, n, 1, name);
      reg.apply1(GATES[name], target);
    } else if (name === "CNOT" || name === "CZ") {
      const [control, target] = qubitsFor(spec, n, 2, name);
      if (name === "CNOT") reg.cnot(control, target);
      else reg.cz(control, target);
    } else if (name === "SWAP") {
      const [a, b] = qubitsFor(spec, n, 2, name);
      reg.swap(a, b);
    } else {
      throw new Error(`unknown gate: ${JSON.stringify(spec.name)}`);
    }
  }

  return {
    qubits: n,
    gates: specs.length,
    amplitudes: amplitudesOf(reg),
    counts: reg.sample(SHOTS),
    shots: SHOTS,
    norm: reg.norm(),
  };
}

/* ------------------------------------------------------------- /api/grover */
function mark(reg, marked) {
  const n = reg.n;
  const zeros = [];
  for (let q = 0; q < n; q += 1) {
    if (!((marked >> (n - 1 - q)) & 1)) zeros.push(q);
  }
  for (const q of zeros) reg.x(q);
  const controls = [];
  for (let q = 0; q < n - 1; q += 1) controls.push(q);
  reg.controlled(GATES.Z, controls, n - 1);
  for (const q of zeros) reg.x(q);
}

function diffuse(reg) {
  for (let q = 0; q < reg.n; q += 1) reg.h(q);
  for (let q = 0; q < reg.n; q += 1) reg.x(q);
  const controls = [];
  for (let q = 0; q < reg.n - 1; q += 1) controls.push(q);
  reg.controlled(GATES.Z, controls, reg.n - 1);
  for (let q = 0; q < reg.n; q += 1) reg.x(q);
  for (let q = 0; q < reg.n; q += 1) reg.h(q);
}

function grover(n, marked, iterations) {
  const reg = new QReg(n);
  for (let q = 0; q < n; q += 1) reg.h(q);
  for (let step = 0; step < iterations; step += 1) {
    mark(reg, marked);
    diffuse(reg);
  }
  return { success: reg.probabilities()[marked], reg };
}

function bestIteration(n, marked) {
  const limit = Math.max(1, Math.ceil((Math.PI / 4) * Math.sqrt(2 ** n)) + 1);
  let best = { iterations: 1, success: 0 };
  for (let steps = 1; steps <= limit; steps += 1) {
    const success = grover(n, marked, steps).success;
    if (success > best.success) best = { iterations: steps, success };
  }
  return best;
}

function runGrover(payload) {
  const n = integer(payload, "qubits", 3, "qubits and marked must be integers");
  const marked = integer(payload, "marked", 0, "qubits and marked must be integers");
  if (n < 2 || n > 7) throw new Error("Grover searches 2 to 7 qubits here");
  if (marked < 0 || marked >= 2 ** n) {
    throw new Error(`the marked item must be between 0 and ${2 ** n - 1}`);
  }

  const best = bestIteration(n, marked);
  const result = grover(n, marked, best.iterations);

  const limit = Math.max(1, Math.ceil((Math.PI / 4) * Math.sqrt(2 ** n)) + 1);
  const curve = [];
  for (let steps = 1; steps <= limit + 1; steps += 1) {
    curve.push({ iterations: steps, prob: grover(n, marked, steps).success });
  }

  return {
    qubits: n,
    marked,
    iterations: best.iterations,
    success_probability: best.success,
    classical_probability: 1 / 2 ** n,
    probabilities: probabilitiesOf(result.reg),
    curve,
  };
}

/* -------------------------------------------------------- /api/entanglement */
function runEntanglement(payload) {
  let theta = number(payload, "theta", Math.PI / 4);
  theta = Math.min(Math.PI / 2, Math.max(0, theta));
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const p = c * c;
  const q = s * s;
  let entropy = 0;
  for (const weight of [p, q]) {
    if (weight > 1e-15) entropy -= weight * Math.log2(weight);
  }
  return {
    theta,
    theta_over_pi: theta / Math.PI,
    probabilities: [
      { bits: "00", prob: p },
      { bits: "11", prob: q },
    ],
    entropy: Math.max(0, entropy),
    concurrence: 2 * c * s,
    negativity: c * s,
    schmidt: [Math.max(c, s), Math.min(c, s)],
  };
}

/* ---------------------------------------------------------------- /api/bb84 */
function runBB84(payload) {
  const nBits = integer(payload, "n_bits", 1200, "n_bits must be an integer");
  if (nBits < 50 || nBits > 20000) {
    throw new Error("n_bits must be between 50 and 20000");
  }
  const eavesdropper = payload.eavesdropper === true || payload.eavesdropper === "true";
  const coin = () => (Math.random() < 0.5 ? 0 : 1);

  let errors = 0;
  let sifted = 0;
  const aliceKey = [];
  const bobKey = [];

  for (let i = 0; i < nBits; i += 1) {
    const aliceBit = coin();
    const aliceBasis = coin();
    const bobBasis = coin();

    let arrivingBit = aliceBit;
    let arrivingBasis = aliceBasis;
    if (eavesdropper) {
      const eveBasis = coin();
      const eveBit = eveBasis === aliceBasis ? aliceBit : coin();
      arrivingBit = eveBit;
      arrivingBasis = eveBasis;
    }
    const bobBit = bobBasis === arrivingBasis ? arrivingBit : coin();

    if (aliceBasis === bobBasis) {
      sifted += 1;
      if (aliceBit !== bobBit) errors += 1;
      aliceKey.push(aliceBit);
      bobKey.push(bobBit);
    }
  }

  const preview = 32;
  const alicePreview = aliceKey.slice(0, preview).join("");
  const bobPreview = bobKey.slice(0, preview).join("");
  const mismatch = [];
  for (let i = 0; i < alicePreview.length; i += 1) {
    if (alicePreview[i] !== bobPreview[i]) mismatch.push(i);
  }

  return {
    n_bits: nBits,
    eavesdropper,
    sifted_length: sifted,
    errors,
    qber: sifted ? errors / sifted : 0,
    threshold: 0.11,
    alice_preview: alicePreview,
    bob_preview: bobPreview,
    mismatch_preview: mismatch,
  };
}

/* ------------------------------------------------------------- /api/gravity */
function phaseDifference(mass, distance, separation, time) {
  const geometry =
    1 / distance + 1 / (distance + separation) - 2 / (distance + separation / 2);
  return ((G * mass ** 2 * time) / HBAR) * geometry;
}

const phaseRate = (mass, distance, separation) =>
  phaseDifference(mass, distance, separation, 1);

function bmvConcurrence(mass, distance, separation, time, coherence) {
  const dphi = phaseDifference(mass, distance, separation, time);
  return Math.exp(-time / coherence) * Math.abs(Math.sin(dphi / 2));
}

function runGravity(payload) {
  const mass = number(payload, "mass", 1e-12);
  const distance = number(payload, "distance", 1e-3);
  const separation = number(payload, "separation", 250e-6);
  const coherence = number(payload, "coherence_time", 0.2);
  if (mass <= 0 || distance <= 0 || separation <= 0 || coherence <= 0) {
    throw new Error("mass, distance, separation and T2 must be positive");
  }

  const rate = phaseRate(mass, distance, separation);
  const time = rate === 0 ? 0 : (2 / rate) * Math.atan((rate * coherence) / 2);
  const span = 3 * coherence;

  const curve = [];
  for (let i = 0; i <= 120; i += 1) {
    const t = (span * i) / 120;
    curve.push({
      t,
      concurrence: bmvConcurrence(mass, distance, separation, t, coherence),
      phase: phaseDifference(mass, distance, separation, t),
    });
  }

  return {
    mass,
    distance,
    separation,
    coherence_time: coherence,
    phase_rate: rate,
    best: {
      time,
      phase: phaseDifference(mass, distance, separation, time),
      concurrence: bmvConcurrence(mass, distance, separation, time, coherence),
    },
    curve,
    span,
  };
}

/* ------------------------------------------------------------------ routes */
const ROUTES = {
  "/api/circuit": runCircuit,
  "/api/grover": runGrover,
  "/api/entanglement": runEntanglement,
  "/api/bb84": runBB84,
  "/api/gravity": runGravity,
};

function queryPayload(search) {
  const payload = {};
  for (const [key, raw] of new URLSearchParams(search)) {
    try {
      payload[key] = JSON.parse(raw);
    } catch {
      payload[key] = raw;
    }
  }
  return payload;
}

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/api/")) {
      const handler = ROUTES[url.pathname];
      if (!handler) return json({ error: `no such endpoint: ${url.pathname}` }, 404);
      try {
        let payload = queryPayload(url.search);
        if (request.method === "POST") {
          const body = await request.text();
          const parsed = body ? JSON.parse(body) : {};
          if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
            throw new Error("expected a JSON object");
          }
          payload = parsed;
        }
        return json(handler(payload));
      } catch (error) {
        return json({ error: error.message || String(error) }, 400);
      }
    }

    // Everything that is not an API call is a static file.  Do not rewrite "/"
    // to "/index.html": the asset router already serves the index there and
    // redirects an explicit /index.html back to "/", which would loop.
    return env.ASSETS.fetch(request);
  },
};
