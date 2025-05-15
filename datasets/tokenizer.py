import json

class GateTokenizer:
    def __init__(self):
        self.vocab = {
            "X": 0, "Y": 1, "Z": 2, "H": 3, "CNOT": 4, "<PAD>": 5
        }

    def encode(self, circuit_json):
        data = json.loads(circuit_json)
        tokens = []
        for moment in data['moments']:
            for op in moment['operations']:
                gate = op['gate']['cirq_type']
                tokens.append(self.vocab.get(gate, self.vocab['<PAD>']))
        return tokens

    def decode(self, token_list):
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return [rev_vocab.get(t, "<UNK>") for t in token_list]
