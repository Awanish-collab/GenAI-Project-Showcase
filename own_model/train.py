import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import RecipeDataset
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from utils.vocab import PAD_TOKEN
import os
from tqdm import tqdm

# Configs
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EMBED_SIZE = 128
HIDDEN_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/recipe_dataset.csv"
SAVE_PATH = "model/recipe_model.pth"

# Load dataset
dataset = RecipeDataset(DATA_PATH)
input_vocab = dataset.input_vocab
target_vocab = dataset.target_vocab

pad_idx = target_vocab.token2idx[PAD_TOKEN]
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
encoder = Encoder(len(input_vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
decoder = Decoder(len(target_vocab), EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        src = batch["input_ids"].to(DEVICE)             # (B, src_len)
        trg = batch["target_ids"].to(DEVICE)            # (B, trg_len)

        optimizer.zero_grad()

        output = model(src, trg)                        # (B, trg_len, vocab_size)
        output = output[:, 1:].reshape(-1, output.shape[-1])  # exclude first token
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# Save model and vocab
os.makedirs("model", exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "input_vocab": input_vocab,
    "target_vocab": target_vocab
}, SAVE_PATH)

print("✅ Model training complete and saved.")
