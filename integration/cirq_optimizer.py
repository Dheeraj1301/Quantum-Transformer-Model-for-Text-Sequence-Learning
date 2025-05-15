import cirq
import json

def optimize_circuit(original_circuit_json):
    circuit = cirq.read_json(json_text=original_circuit_json)
    # Example optimization: merge single qubit gates
    opt_circuit = cirq.merge_single_qubit_gates_to_phxz(circuit)
    return opt_circuit

if __name__ == "__main__":
    with open('data/raw/circuit_0.json', 'r') as f:
        circuit_json = f.read()
    optimized = optimize_circuit(circuit_json)
    print("Optimized circuit:\n", optimized)
