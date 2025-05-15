def compute_fidelity(pred_tokens, target_tokens):
    """ Dummy fidelity: fraction of matching tokens """
    matches = sum(p == t for p, t in zip(pred_tokens, target_tokens))
    return matches / max(len(target_tokens), 1)

def compute_gate_depth(tokens):
    """ Dummy gate depth: count of unique non-pad tokens """
    return len(set(t for t in tokens if t != 0))  # Assuming 0 is <PAD>
