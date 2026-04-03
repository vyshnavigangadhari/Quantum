# tsqs_tsp_increment1.py
# Hybrid-Enhanced Two-Step Quantum Search (TSQS) for TSP
# Includes: classical preprocessing, threshold oracle,
# automatic adaptive iteration tuning (always on),
# optional noise model, and result visualization.
# ==========================================================
# 1) Imports and setup
# ==========================================================
from qiskit import *
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
import itertools as it
import numpy as np
import math, time, matplotlib.pyplot as plt

# ----------------------------------------------------------
# Optional noise model import
# ----------------------------------------------------------
FakeVigo = None
try:
    from qiskit_ibm_runtime.fake_provider import FakeVigo
except ImportError:
    try:
        from qiskit.providers.fake_provider import FakeVigo
    except Exception:
        FakeVigo = None

# ==========================================================
# 2) Problem instance
# ==========================================================
n = 3  # choose n=3 or n=4
cost_matrix_3 = [
    [0, 2, 9],
    [1, 0, 6],
    [15, 7, 0]
]
cost_matrix_4 = [
    [0, 3, 7, 9],
    [4, 0, 8, 2],
    [6, 5, 0, 3],
    [10, 1, 4, 0]
]
C = cost_matrix_3 if n == 3 else cost_matrix_4
K = math.ceil(math.log2(n))
num_data = n * K

# ==========================================================
# 3) Classical preprocessing
# ==========================================================
def tour_cost(tour, C):
    return sum(C[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def nearest_neighbor_tour(C, start=0):
    n = len(C)
    tour, remaining = [start], set(range(n)) - {start}
    while remaining:
        last = tour[-1]
        next_city = min(remaining, key=lambda j: C[last][j])
        tour.append(next_city)
        remaining.remove(next_city)
    return tour

def two_opt_local_optimize(tour, C):
    n = len(tour)
    best, best_cost = tour[:], tour_cost(tour, C)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_tour = best[:]
                new_tour[i:j] = reversed(best[i:j])
                new_cost = tour_cost(new_tour, C)
                if new_cost < best_cost:
                    best, best_cost, improved = new_tour, new_cost, True
        if improved:
            break
    return best, best_cost

def reduce_symmetry(tours):
    reduced, seen = [], set()
    for t in tours:
        reps = []
        for _ in range(len(t)):
            reps.append(tuple(t))
            t = t[1:] + t[:1]
        rev = tuple(t[::-1])
        key = min(reps + [rev])
        if key not in seen:
            seen.add(key)
            reduced.append(list(key))
    return reduced

def preprocess_tours(C, n, top_k=6, alpha=1.05):
    base_tours = [nearest_neighbor_tour(C, start=i) for i in range(n)]
    optimized = [two_opt_local_optimize(t, C) for t in base_tours]
    optimized.sort(key=lambda x: x[1])
    best_cost = optimized[0][1]
    S = [t for t, c in optimized[:top_k]]
    S = reduce_symmetry(S)
    threshold = alpha * best_cost
    return S, threshold, best_cost

# ==========================================================
# 4) Helper functions
# ==========================================================
def int_to_bits(x, K): return [(x >> (K - 1 - i)) & 1 for i in range(K)]
def tour_to_bits(tour, n, K): return [b for city in tour for b in int_to_bits(city, K)]
def bits_to_tour(bits, n, K):
    tour = []
    for i in range(n):
        chunk, val = bits[i*K:(i+1)*K], 0
        for b in chunk: val = (val << 1) | b
        if val >= n: return None
        tour.append(val)
    return tour

def bitstring_to_bits(s): return [1 if ch == '1' else 0 for ch in s][::-1]
def decode_and_score(counts, n, K, C):
    agg = {}
    for bitstr, freq in counts.items():
        bits = bitstring_to_bits(bitstr)
        tour = bits_to_tour(bits, n, K)
        if tour is None or len(set(tour)) != n:
            continue
        c = tour_cost(tour, C)
        agg.setdefault(tuple(tour), [c, 0])
        agg[tuple(tour)][1] += freq
    return [(t, c, f) for t, (c, f) in agg.items()]

# ==========================================================
# 5) Oracles and diffusion
# ==========================================================
def append_validity_oracle(qc, data, work_valid, flag_valid):
    invalid = (1 << K) - 1 if n == (1 << K) else n
    needs_validity = (1 << K) != n
    if not needs_validity:
        qc.x(flag_valid)
        return
    for t in range(n):
        start = t * K
        code_bits = [(invalid >> (K - 1 - i)) & 1 for i in range(K)]
        for i, cb in enumerate(code_bits):
            if cb == 0:
                qc.x(data[start + i])
        qc.append(MCXGate(K), data[start:start + K] + [work_valid[t]])
        for i, cb in enumerate(code_bits):
            if cb == 0:
                qc.x(data[start + i])
    for t in range(n):
        qc.x(work_valid[t])
    qc.append(MCXGate(n), work_valid + [flag_valid])
    for t in range(n):
        qc.x(work_valid[t])

def append_R1(qc, data, anc):
    work_valid = [anc[i] for i in range(n)]
    flag_valid = anc[n]
    phase_qubit = anc[-1]
    append_validity_oracle(qc, data, work_valid, flag_valid)
    qc.cp(math.pi, flag_valid, phase_qubit)

def diffuser_D1(qc, data):
    N = len(data)
    qc.h(data)
    qc.x(data)
    last = data[-1]
    qc.h(last)
    qc.append(MCXGate(N - 1), data[:-1] + [last])
    qc.h(last)
    qc.x(data)
    qc.h(data)

def apply_G1(qc, data, anc, iters):
    for _ in range(iters):
        append_R1(qc, data, anc)
        diffuser_D1(qc, data)

def append_match_pattern(qc, data, pattern_bits, target):
    for i, bit in enumerate(pattern_bits):
        if bit == 0:
            qc.x(data[i])
    qc.append(MCXGate(len(data)), data[:] + [target])
    for i, bit in enumerate(pattern_bits):
        if bit == 0:
            qc.x(data[i])

def append_R2(qc, data, phase_anc, phases_by_tour, threshold=None):
    for tour, cost in phases_by_tour:
        if threshold is not None and cost > threshold:
            continue
        pat = tour_to_bits(tour, n, K)
        append_match_pattern(qc, data, pat, phase_anc)
        qc.p(math.pi, phase_anc)
        append_match_pattern(qc, data, pat, phase_anc)

def diffuser_D2(qc, data, anc, t1):
    apply_G1(qc, data, anc, t1)
    diffuser_D1(qc, data)
    apply_G1(qc, data, anc, t1)

# ==========================================================
# 6) Build hybrid TSQS circuit
# ==========================================================
def build_tsqs_circuit(n, K, C, t1, t2, S=None, tau=None):
    num_anc = n + 2
    qc = QuantumCircuit(num_data + num_anc, num_data)
    data = list(range(num_data))
    anc = list(range(num_data, num_data + num_anc))
    phase_R2 = anc[-1]

    qc.h(data)
    apply_G1(qc, data, anc, t1)
    tours = S if S is not None else list(it.permutations(range(n), n))
    phases_by_tour = [(t, tour_cost(t, C)) for t in tours]
    for _ in range(t2):
        append_R2(qc, data, phase_R2, phases_by_tour, threshold=tau)
        diffuser_D2(qc, data, anc, t1)
    qc.measure(data, range(num_data))
    return qc

# ==========================================================
# 7) Automatic Adaptive Iteration Tuning
# ==========================================================
def tune_iterations(n, K, C, S, tau, backend):
    print("\n[Adaptive Tuning] Searching optimal (t1, t2)...")
    space_size = 2 ** (n * K)
    subset_size = len(S)
    best_t1, best_t2, best_popt = 1, 1, -1
    for t1 in range(1, 8):
        for t2 in range(1, 4):
            qc = build_tsqs_circuit(n, K, C, t1, t2, S=S, tau=tau)
            tqc = transpile(qc, backend=backend, optimization_level=0)
            result = backend.run(tqc, shots=256).result()
            decoded = decode_and_score(result.get_counts(), n, K, C)
            if not decoded:
                continue
            min_cost = min(c for _, c, _ in decoded)
            opt_mass = sum(f for _t, c, f in decoded if c == min_cost)
            prob_opt = opt_mass / 256
            if prob_opt > best_popt:
                best_t1, best_t2, best_popt = t1, t2, prob_opt
    print(f"[Adaptive Tuning] Best (t1, t2) = ({best_t1}, {best_t2}), P_opt={best_popt:.3f}")
    return best_t1, best_t2

# ==========================================================
# 8) Main execution
# ==========================================================
S, tau, cmin = preprocess_tours(C, n, top_k=6, alpha=1.05)
backend = AerSimulator(seed_simulator=42)
if FakeVigo:
    noise_model = NoiseModel.from_backend(FakeVigo())
    backend.set_options(noise_model=noise_model)

t1, t2 = tune_iterations(n, K, C, S, tau, backend)
print(f"[Hybrid] n={n}, K={K}, |S|={len(S)}, τ={tau:.2f}, t1={t1}, t2={t2}")

start = time.time()
qc = build_tsqs_circuit(n, K, C, t1, t2, S=S, tau=tau)
tqc = transpile(qc, backend=backend, optimization_level=0)
result = backend.run(tqc, shots=2048).result()
counts = result.get_counts()
print(f"[Runtime] {time.time() - start:.2f}s")

# ==========================================================
# 9) Results and visualization
# ==========================================================
decoded = decode_and_score(counts, n, K, C)
if decoded:
    rows = [{"tour": tuple(x+1 for x in t), "cost": c, "freq": f, "prob": round(f/2048, 3)}
            for t, c, f in decoded]
    rows = sorted(rows, key=lambda r: (r["cost"], -r["freq"]))
    min_cost = min(r["cost"] for r in rows)
    print("\n=== Tours Summary (Hybrid TSQS) ===")
    for r in rows:
        tag = " <-- OPTIMAL" if r["cost"] == min_cost else ""
        print(f"tour={r['tour']} cost={r['cost']} freq={r['freq']} p={r['prob']}{tag}")

    # Plot results
    tours = ["-".join(map(str, r["tour"])) for r in rows]
    freqs = [r["freq"] for r in rows]
    probs = [r["prob"] for r in rows]
    colors = ["#2ecc71" if r["cost"] == min_cost else "#3498db" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(tours, freqs, color=colors)
    axes[0].set_title("Tour Frequency Distribution (Hybrid TSQS)")
    axes[0].set_xlabel("Tours")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis="x", rotation=60)

    axes[1].bar(tours, probs, color=colors)
    axes[1].set_title("Tour Probability Distribution (Hybrid TSQS)")
    axes[1].set_xlabel("Tours")
    axes[1].set_ylabel("Probability")
    axes[1].tick_params(axis="x", rotation=60)

    plt.tight_layout()
    plt.show()

print(f"\nCircuit width: {tqc.num_qubits}")
print(f"Logical depth: {tqc.depth()}")
print(f"Physical depth: {tqc.decompose(reps=5).depth()}")
