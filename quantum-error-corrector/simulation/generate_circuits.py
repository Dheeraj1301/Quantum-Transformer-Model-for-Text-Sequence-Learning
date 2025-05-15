import cirq
import random
import os
import json

def generate_random_circuit(n_qubits=6, depth=10):
    qubits = [cirq.GridQubit(i, 0) for i in range(n_qubits)]
    circuit = cirq.Circuit()
    for _ in range(depth):
        gate = random.choice([cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.CNOT])
        if gate == cirq.CNOT:
            q1, q2 = random.sample(qubits, 2)
            circuit.append(gate(q1, q2))
        else:
            q = random.choice(qubits)
            circuit.append(gate(q))
    return circuit

def save_circuit(circuit, idx, save_path):
    os.makedirs(save_path, exist_ok=True)
    with open(f'{save_path}/circuit_{idx}.json', 'w') as f:
        json.dump(cirq.to_json(circuit), f)

def generate_dataset(num_circuits=1000, save_path='data/raw'):
    for i in range(num_circuits):
        c = generate_random_circuit()
        save_circuit(c, i, save_path)

if __name__ == '__main__':
    generate_dataset()
