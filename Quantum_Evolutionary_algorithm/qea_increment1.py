# qea_increment1.py
# Quantum Evolution (QAOA-style) for TSP
# Output format aligned with tsqs_increment2.py for easy comparison.
# ==========================================================
# 1) Imports and setup
# ==========================================================
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
import itertools as it
import numpy as np
import time, math, matplotlib.pyplot as plt
from scipy.optimize import minimize


# ==========================================================
# 2) Problem instance (match TSQS)
# ==========================================================
n = 4
cost_matrix_4 = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

cost_matrix_3 = [
    [0, 2, 9],
    [1, 0, 6],
    [15, 7, 0]
]
C = cost_matrix_4

# ==========================================================
# 3) Helper functions
# ==========================================================
def tour_cost(tour, C):
    return sum(C[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))

def all_tours(n):
    return list(it.permutations(range(n), n))

def bitstring_to_index(bitstr):
    return int(bitstr, 2)

# ==========================================================
# 4) Build cost/mixer Hamiltonians (QAOA)
# ==========================================================
tours = all_tours(n)
costs = np.array([tour_cost(t, C) for t in tours])

num_qubits = int(math.ceil(math.log2(len(tours))))  # 2^num_qubits >= n!
max_cost = int(np.max(costs))
min_cost = int(np.min(costs))
penalty_cost = max_cost + (max_cost - min_cost)

paulis = []
coeffs = []
dim = 2 ** num_qubits
for i in range(dim):
    cost = costs[i] if i < len(costs) else penalty_cost
    bitstring = format(i, f"0{num_qubits}b")
    z_string = "".join(["Z" if b == "1" else "I" for b in bitstring])
    paulis.append(z_string)
    coeffs.append(cost)

H_cost = SparsePauliOp(paulis, coeffs)
H_mixer = SparsePauliOp(
    ["".join("X" if j == i else "I" for j in range(num_qubits)) for i in range(num_qubits)],
    [1] * num_qubits,
)

backend = AerSimulator(seed_simulator=42)

# ==========================================================
# 5) QAOA circuit
# ==========================================================
def build_qaoa(params, p=1):
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    gammas = params[:p]
    betas = params[p:]

    for layer in range(p):
        qc.append(PauliEvolutionGate(H_cost, time=gammas[layer]), range(num_qubits))
        qc.append(PauliEvolutionGate(H_mixer, time=betas[layer]), range(num_qubits))

    qc.measure_all()
    return qc

# ==========================================================
# 6) Adaptive parameter tuning
# ==========================================================
def objective(params, p=1, shots=256):
    qc = build_qaoa(params, p=p)
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    exp_value = 0.0
    for bitstr, freq in counts.items():
        idx = bitstring_to_index(bitstr)
        cost = costs[idx] if idx < len(costs) else penalty_cost
        exp_value += (freq / shots) * cost
    return exp_value

print("\n[Adaptive Tuning] Searching optimal (gamma, beta)...")
p_layers = 1
initial_params = [0.5] * (2 * p_layers)
start_tune = time.time()
opt = minimize(lambda x: objective(x, p=p_layers, shots=256), initial_params, method="COBYLA")
gamma_opt, beta_opt = opt.x[:p_layers][0], opt.x[p_layers:][0]
print(f"[Adaptive Tuning] Best (gamma, beta) = ({gamma_opt:.3f}, {beta_opt:.3f})")
print(f"[Tuning Runtime] {time.time() - start_tune:.2f}s")

# ==========================================================
# 7) Main execution
# ==========================================================
start = time.time()
qc = build_qaoa(opt.x, p=p_layers)
tqc = transpile(qc, backend)
result = backend.run(tqc, shots=2048).result()
counts = result.get_counts()
print(f"[Runtime] {time.time() - start:.2f}s")

# ==========================================================
# 8) Results and visualization
# ==========================================================
rows = []
for i, tour in enumerate(tours):
    bit = format(i, f"0{num_qubits}b")
    freq = counts.get(bit, 0)
    prob = round(freq / 2048, 3)
    rows.append({
        "tour": tuple(x + 1 for x in tour),
        "cost": int(costs[i]),
        "freq": freq,
        "prob": prob
    })

rows = sorted(rows, key=lambda r: (r["cost"], -r["freq"]))
min_cost = min(r["cost"] for r in rows)

print("\n=== Tours Summary (Quantum Evolution - QAOA) ===")
for r in rows:
    tag = " <-- OPTIMAL" if r["cost"] == min_cost else ""
    print(f"tour={r['tour']} cost={r['cost']} freq={r['freq']} p={r['prob']}{tag}")

tours_labels = ["-".join(map(str, r["tour"])) for r in rows]
freqs = [r["freq"] for r in rows]
probs = [r["prob"] for r in rows]
colors = ["#2ecc71" if r["cost"] == min_cost else "#3498db" for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(tours_labels, freqs, color=colors)
axes[0].set_title("Tour Frequency Distribution (Quantum Evolution)")
axes[0].set_xlabel("Tours")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis="x", rotation=60)

axes[1].bar(tours_labels, probs, color=colors)
axes[1].set_title("Tour Probability Distribution (Quantum Evolution)")
axes[1].set_xlabel("Tours")
axes[1].set_ylabel("Probability")
axes[1].tick_params(axis="x", rotation=60)

plt.tight_layout()
plt.show()

print(f"\nCircuit width: {tqc.num_qubits}")
print(f"Logical depth: {tqc.depth()}")
print(f"Physical depth: {tqc.decompose(reps=5).depth()}")
