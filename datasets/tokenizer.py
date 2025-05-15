import json
import torch



class GateTokenizer:
    
    def __init__(self):
        self.vocab = {
            "X": 0, "Y": 1, "Z": 2, "H": 3, "CNOT": 4, "<PAD>": 5
        }

    def encode(self, data):
        circuit_data = data.get("moments") or [[gate] for gate in data.get("circuit", [])]

        tokens = []
        for moment in circuit_data:
            for op in moment:
                if isinstance(op, dict):  # Old format
                    gate = op.get("gate", {}).get("cirq_type", "")
                elif isinstance(op, list):  # Simple format
                    gate = op[0]
                else:
                    gate = "UNK"

                tokens.append(self.vocab.get(gate, 0))
        return tokens


    def decode(self, token_list):
        rev_vocab = {v: k for k, v in self.vocab.items()}

        # Convert tensor or int to list
        if isinstance(token_list, torch.Tensor):
            token_list = token_list.tolist()
        elif isinstance(token_list, int):
            token_list = [token_list]

        return [rev_vocab.get(t, "<UNK>") for t in token_list]
