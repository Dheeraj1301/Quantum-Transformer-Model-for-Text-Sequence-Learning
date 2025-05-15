import cirq
import json
import os
from cirq.google import XmonSimulator

def load_circuit(path):
    with open(path, 'r') as f:
        return cirq.read_json(f)

def simulate(circuit):
    sim = XmonSimulator()
    result = sim.simulate(circuit)
    return result.final_state_vector

def process_all(path='data/raw', save_path='data/processed'):
    os.makedirs(save_path, exist_ok=True)
    metadata = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            circuit = load_circuit(os.path.join(path, file))
            vector = simulate(circuit)
            metadata.append({
                'filename': file,
                'gate_count': len(circuit),
                'depth': circuit.depth(),
                'fidelity': float(abs(vector[0])**2)  # dummy metric
            })
    with open(f'{save_path}/circuit_metadata.csv', 'w') as f:
        f.write("filename,gate_count,depth,fidelity\n")
        for row in metadata:
            f.write(f"{row['filename']},{row['gate_count']},{row['depth']},{row['fidelity']}\n")

if __name__ == '__main__':
    process_all()
