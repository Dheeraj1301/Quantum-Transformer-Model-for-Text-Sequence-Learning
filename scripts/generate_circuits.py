
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cirq
from cirq import depolarize
from cirq.devices import GridQubit
from cirq import ConstantQubitNoiseModel
from cirq_optimizer import optimize_circuit


def create_sample_circuit() -> cirq.Circuit:
    """
    Creates a sample quantum circuit for demonstration purposes.
    
    Returns:
        A sample quantum circuit.
    """
    qubits = [GridQubit(0, i) for i in range(2)]
    circuit = cirq.Circuit()
    circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.H(qubits[1])])
    return circuit

def main():
    # Create a sample circuit
    circuit = create_sample_circuit()
    print("Original Circuit:")
    print(circuit)

    # Define a simple depolarizing noise model
    noise_model = ConstantQubitNoiseModel(depolarize(p=0.01))

    # Optimize the circuit
    optimized_circuit = optimize_circuit(circuit, noise_model=noise_model)
    print("\nOptimized Circuit:")
    print(optimized_circuit)

if __name__ == "__main__":
    main()
