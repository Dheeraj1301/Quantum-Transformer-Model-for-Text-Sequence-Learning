import torch
from model.transformer_model import  TransformerModel
from datasets.tokenizer import GateTokenizer

def evaluate_model(model_path="checkpoints/tqco_model.pt"):
    model =  TransformerModel(vocab_size=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tokenizer = GateTokenizer()
    input_sequence = [0, 1, 3, 4, 2]  # Sample tokenized input
    input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        output = model(input_tensor)
        predicted_tokens = torch.argmax(output, dim=-1).squeeze().tolist()

    

# after
    if isinstance(predicted_tokens, int):
        predicted_tokens = [predicted_tokens]
    decoded = tokenizer.decode(predicted_tokens)
    print("Predicted:", predicted_tokens)
    print("Decoded:", decoded)
    return predicted_tokens, decoded
if __name__ == '__main__':
    evaluate_model()
