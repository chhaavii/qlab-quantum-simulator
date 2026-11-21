"""A web frontend for QLab.

Serves a single-page interface plus a JSON API on top of the simulator in
``qlab``.  Only the Python standard library is used, so running it needs
nothing beyond the packages already in ``requirements.txt``::

    python3 web/app.py            # then open http://localhost:8000

Every endpoint takes and returns JSON (POST):

    /api/circuit        build a circuit from a gate list, return its state
    /api/grover         Grover search, with a success-vs-iterations curve
    /api/entanglement   the three measures on cos(t)|00> + sin(t)|11>
    /api/bb84           a BB84 run, with or without intercept-resend
    /api/gravity        BMV concurrence against interaction time
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from qlab import algorithms, entanglement, gravity  # noqa: E402
from qlab import gates as matrices  # noqa: E402
from qlab.bb84 import run_bb84  # noqa: E402
from qlab.register import QuantumRegister  # noqa: E402

STATIC_DIR = os.path.join(HERE, "static")
MAX_CIRCUIT_QUBITS = 8
SHOTS = 1024

SINGLE_QUBIT = {
    "H": matrices.H,
    "X": matrices.X,
    "Y": matrices.Y,
    "Z": matrices.Z,
    "S": matrices.S,
    "T": matrices.T,
}


# --------------------------------------------------------------------- helpers
def _number(payload: dict, key: str, default: float) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _qubits(spec: dict, n: int, count: int, name: str) -> list[int]:
    try:
        qubits = [int(q) for q in spec.get("qubits", [])]
    except (TypeError, ValueError):
        raise ValueError(f"{name} needs integer qubit indices") from None
    if len(qubits) != count:
        raise ValueError(f"{name} acts on {count} qubit(s)")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"{name} needs distinct qubits")
    for qubit in qubits:
        if not 0 <= qubit < n:
            raise ValueError(f"qubit {qubit} is outside a {n}-qubit register")
    return qubits


def _amplitudes(reg: QuantumRegister, threshold: float = 1e-9) -> list[dict]:
    vector = reg.vector()
    out = []
    for index in range(reg.dim):
        amplitude = vector[index]
        probability = float(abs(amplitude) ** 2)
        if probability < threshold:
            continue
        out.append(
            {
                "index": index,
                "bits": format(index, f"0{reg.n}b"),
                "re": float(amplitude.real),
                "im": float(amplitude.imag),
                "prob": probability,
                "phase": float(np.angle(amplitude)),
            }
        )
    return out


def _probabilities(reg: QuantumRegister) -> list[dict]:
    vector = reg.vector()
    return [
        {
            "bits": format(index, f"0{reg.n}b"),
            "prob": float(abs(vector[index]) ** 2),
        }
        for index in range(reg.dim)
    ]


# --------------------------------------------------------------- /api/circuit
def run_circuit(payload: dict) -> dict:
    try:
        n = int(payload.get("qubits", 2))
    except (TypeError, ValueError):
        raise ValueError("qubits must be an integer") from None
    if not 1 <= n <= MAX_CIRCUIT_QUBITS:
        raise ValueError(f"a circuit can have 1 to {MAX_CIRCUIT_QUBITS} qubits")

    specs = payload.get("gates") or []
    if not isinstance(specs, list):
        raise ValueError("gates must be a list")

    reg = QuantumRegister(n)
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("each gate must be an object")
        name = str(spec.get("name", "")).upper()
        if name in SINGLE_QUBIT:
            (target,) = _qubits(spec, n, 1, name)
            reg.apply(SINGLE_QUBIT[name], target)
        elif name in ("CNOT", "CZ"):
            control, target = _qubits(spec, n, 2, name)
            (reg.cnot if name == "CNOT" else reg.cz)(control, target)
        elif name == "SWAP":
            a, b = _qubits(spec, n, 2, name)
            reg.swap(a, b)
        else:
            raise ValueError(f"unknown gate: {name or spec.get('name')!r}")

    rng = np.random.default_rng()
    return {
        "qubits": n,
        "gates": len(specs),
        "amplitudes": _amplitudes(reg),
        "counts": reg.sample(SHOTS, rng),
        "shots": SHOTS,
        "norm": float(np.linalg.norm(reg.vector())),
    }


# ---------------------------------------------------------------- /api/grover
def run_grover(payload: dict) -> dict:
    try:
        n = int(payload.get("qubits", 3))
        marked = int(payload.get("marked", 0))
    except (TypeError, ValueError):
        raise ValueError("qubits and marked must be integers") from None
    if not 2 <= n <= 7:
        raise ValueError("Grover searches 2 to 7 qubits here")
    if not 0 <= marked < 2**n:
        raise ValueError(f"the marked item must be between 0 and {2 ** n - 1}")

    best = algorithms.grover_best_iteration(n, marked)
    result = algorithms.grover(n, marked, iterations=best["iterations"])

    limit = max(1, int(np.ceil(np.pi / 4.0 * np.sqrt(2**n))) + 1)
    curve = []
    for steps in range(1, limit + 2):
        curve.append(
            {
                "iterations": steps,
                "prob": algorithms.grover(n, marked, iterations=steps)[
                    "success_probability"
                ],
            }
        )

    return {
        "qubits": n,
        "marked": marked,
        "iterations": best["iterations"],
        "success_probability": best["success_probability"],
        "classical_probability": 1.0 / 2**n,
        "probabilities": _probabilities_result(result),
        "curve": curve,
    }


def _probabilities_result(result: dict) -> list[dict]:
    probabilities = np.asarray(result["probabilities"], dtype=float)
    n = len(format(probabilities.size - 1, "b"))
    return [
        {"bits": format(index, f"0{n}b"), "prob": float(value)}
        for index, value in enumerate(probabilities)
    ]


# ---------------------------------------------------------- /api/entanglement
def run_entanglement(payload: dict) -> dict:
    theta = _number(payload, "theta", float(np.pi / 4))
    theta = float(np.clip(theta, 0.0, np.pi / 2))
    psi = np.array(
        [np.cos(theta), 0.0, 0.0, np.sin(theta)], dtype=complex
    )
    return {
        "theta": theta,
        "theta_over_pi": theta / np.pi,
        "probabilities": [
            {"bits": "00", "prob": float(np.cos(theta) ** 2)},
            {"bits": "11", "prob": float(np.sin(theta) ** 2)},
        ],
        # max() only guards against the -0.0 that log() can produce.
        "entropy": max(0.0, entanglement.von_neumann_entropy(psi)),
        "concurrence": entanglement.concurrence(psi),
        "negativity": entanglement.negativity(psi),
        "schmidt": [float(s) for s in entanglement.schmidt_coefficients(psi)],
    }


# ------------------------------------------------------------------ /api/bb84
def run_bb84_endpoint(payload: dict) -> dict:
    try:
        n_bits = int(payload.get("n_bits", 1200))
    except (TypeError, ValueError):
        raise ValueError("n_bits must be an integer") from None
    if not 50 <= n_bits <= 20000:
        raise ValueError("n_bits must be between 50 and 20000")

    rng = np.random.default_rng()
    result = run_bb84(n_bits, eavesdropper=bool(payload.get("eavesdropper")), rng=rng)
    preview = 32
    alice = "".join(str(int(b)) for b in result["alice_key"][:preview])
    bob = "".join(str(int(b)) for b in result["bob_key"][:preview])
    return {
        "n_bits": n_bits,
        "eavesdropper": bool(payload.get("eavesdropper")),
        "sifted_length": result["sifted_length"],
        "errors": result["errors"],
        "qber": result["qber"],
        "threshold": 0.11,
        "alice_preview": alice,
        "bob_preview": bob,
        "mismatch_preview": [
            i for i, (a, b) in enumerate(zip(alice, bob)) if a != b
        ],
    }


# --------------------------------------------------------------- /api/gravity
def run_gravity(payload: dict) -> dict:
    mass = _number(payload, "mass", 1e-12)
    distance = _number(payload, "distance", 1e-3)
    separation = _number(payload, "separation", 250e-6)
    coherence = _number(payload, "coherence_time", 0.2)
    if mass <= 0 or distance <= 0 or separation <= 0 or coherence <= 0:
        raise ValueError("mass, distance, separation and T2 must be positive")

    best = gravity.best_concurrence(mass, distance, separation, coherence)
    span = 3.0 * coherence
    curve = []
    for i in range(121):
        time = span * i / 120
        curve.append(
            {
                "t": time,
                "concurrence": gravity.concurrence(
                    mass, distance, separation, time, coherence
                ),
                "phase": gravity.phase_difference(mass, distance, separation, time),
            }
        )

    return {
        "mass": mass,
        "distance": distance,
        "separation": separation,
        "coherence_time": coherence,
        "phase_rate": gravity.phase_rate(mass, distance, separation),
        "best": {
            "time": best["time"],
            "phase": best["phase"],
            "concurrence": best["concurrence"],
        },
        "curve": curve,
        "span": span,
    }


ROUTES = {
    "/api/circuit": run_circuit,
    "/api/grover": run_grover,
    "/api/entanglement": run_entanglement,
    "/api/bb84": run_bb84_endpoint,
    "/api/gravity": run_gravity,
}


# ---------------------------------------------------------------------- server
class QLabHandler(BaseHTTPRequestHandler):
    server_version = "QLab/1.0"

    def log_message(self, fmt, *args):  # quieter, single-line access log
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._send_json({"error": "this endpoint expects POST"}, 405)
            return
        self._serve_static("/index.html" if path in ("", "/") else path)

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        handler = ROUTES.get(path)
        if handler is None:
            self._send_json({"error": f"no such endpoint: {path}"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            self._send_json(handler(payload))
        except json.JSONDecodeError:
            self._send_json({"error": "the request body is not valid JSON"}, 400)
        except Exception as exc:  # surface the problem to the interface
            self._send_json({"error": str(exc) or type(exc).__name__}, 400)

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str) -> None:
        relative = os.path.normpath(path.lstrip("/"))
        if relative.startswith(".."):
            self.send_error(403, "outside the static directory")
            return
        full = os.path.join(STATIC_DIR, relative)
        if not os.path.isfile(full):
            self.send_error(404, "not found")
            return
        content_type = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        with open(full, "rb") as handle:
            body = handle.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="serve the QLab web frontend")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)

    httpd = ThreadingHTTPServer((args.host, args.port), QLabHandler)
    print(f"QLab frontend listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
