import torch
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from utils.vocab import Vocab, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
import pickle

# Tell PyTorch it's safe to unpickle Vocab objects
torch.serialization.add_safe_globals([Vocab])


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/recipe_model.pth"
MAX_LEN = 50

# Load model and vocab
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
input_vocab = checkpoint["input_vocab"]
target_vocab = checkpoint["target_vocab"]

encoder = Encoder(len(input_vocab)).to(DEVICE)
decoder = Decoder(len(target_vocab)).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate_response(query):
    tokens = input_vocab.numericalize(query)
    tokens = tokens[:MAX_LEN]
    tokens = tokens + [input_vocab.token2idx[PAD_TOKEN]] * (MAX_LEN - len(tokens))
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden, cell = encoder(input_tensor)

        outputs = []
        input_token = torch.tensor([target_vocab.token2idx[SOS_TOKEN]], dtype=torch.long).to(DEVICE)

        for _ in range(MAX_LEN):
            output, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            predicted_token = top1.item()

            if predicted_token == target_vocab.token2idx[EOS_TOKEN]:
                break

            outputs.append(predicted_token)
            input_token = top1

    return " ".join(target_vocab.denumericalize(outputs))


if __name__ == "__main__":
    while True:
        query = input("\n👤 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = generate_response(query)
        print(f"🤖 RecipeBot: {response}")
