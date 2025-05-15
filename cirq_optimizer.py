import cirq
import networkx as nx

# -----------------------------
# Gate Merging Optimization
# -----------------------------

def merge_func(op1, op2):
    if cirq.num_qubits(op1) == 1 and cirq.num_qubits(op2) == 1:
        try:
            return cirq.merge_single_qubit_gates_into_phxz([op1, op2])
        except Exception:
            return None
    return None

def merge_gates(circuit):
    print("\n[Optimizer] Applying gate merging...")
    optimized_circuit = cirq.merge_operations(circuit, merge_func)
    return optimized_circuit


# -----------------------------
# Depth Reduction Heuristics
# -----------------------------

def reduce_depth(circuit):
    print("[Optimizer] Applying depth reduction...")
    # Strategy: use circuit optimizers like eject_phased_paulis and drop empty moments
    circuit = cirq.eject_phased_paulis(circuit)
    circuit = cirq.drop_empty_moments(circuit)
    return circuit


# -----------------------------
# Noise-Aware Routing
# -----------------------------

def apply_noise_aware_routing(circuit, noise_model=None, device=None):
    print("[Optimizer] Applying noise-aware routing...")
    if noise_model:
        # Apply noise-aware moment alignment if supported
        circuit = cirq.optimize_for_target_gateset(circuit, context=cirq.TransformerContext(deep=True))
    if device:
        circuit = cirq.route_circuit(circuit, device)
    return circuit


# -----------------------------
# Combined Optimization Pipeline
# -----------------------------

def optimize_circuit(circuit, noise_model=None, device=None):
    print("\n[Optimizer] Starting optimization pipeline...")
    # 1. Merge gates
    optimized_circuit = merge_gates(circuit)
    # 2. Reduce circuit depth
    optimized_circuit = reduce_depth(optimized_circuit)
    # 3. Apply noise-aware routing
    optimized_circuit = apply_noise_aware_routing(optimized_circuit, noise_model=noise_model, device=device)
    print("[Optimizer] Optimization complete.")
    return optimized_circuit
