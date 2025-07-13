import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#https://medium.com/@RobuRishabh/learning-word-embeddings-with-cbow-and-skip-gram-b834bde18de4
#https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/
set_seed(42)
# Load DataFrame
model_name = "skipgram" #"cbow"
# Example hyperparameters
batch_size = 128
embedding_dim = 256
learning_rate = 0.001
epochs = 100
window_size = 2

category = "sports"  # business
suffix_category = f"{category}_"
data_file = f"Data/preprocessed_{category}_articles.pkl"
vocab_file = f"Data/vocab_{category}_dict.pkl"

df = pd.read_pickle(data_file)
UNK_TOKEN = "<UNK>"

# Load vocabulary
with open(vocab_file, "rb") as f:
    vocab = pickle.load(f)


def generate_cbow_data(tokens_list, window_size=2):
    data = []
    for tokens in tokens_list:
        for i in range(window_size, len(tokens) - window_size):
            context = tokens[i - window_size:i] + tokens[i + 1:i + window_size + 1]
            target = tokens[i]
            data.append((context, target))
    return data


def generate_skipgram_data(tokens_list, window_size=2):
    data = []
    for tokens in tokens_list:
        for i in range(len(tokens)):
            center = tokens[i]
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if j != i:
                    data.append((center, tokens[j]))
    return data


class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def get_embedding(self, input_ids):
        return self.embeddings(input_ids)


class CBOWModel(WordEmbeddingModel):
    # adapted from https://github.com/FraLotito/pytorch-continuous-bag-of-words/blob/master/cbow.py
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        # context_idxs: (batch_size, context_window*2)
        embeds = self.embeddings(context_idxs)  # shape: (batch_size, context_size, emb_dim)
        context_embed = embeds.mean(dim=1)  # average over context
        out = self.linear(context_embed)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


class SkipGramModel(WordEmbeddingModel):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_idxs):
        # center_idxs: (batch_size)
        embeds = self.embeddings(center_idxs)  # shape: (batch_size, emb_dim)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# Custom Dataset class for CBOW
class CBOWDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idxs = torch.tensor([self.vocab.get(w, self.vocab[UNK_TOKEN]) for w in context], dtype=torch.long)
        target_idx = torch.tensor(self.vocab.get(target, self.vocab[UNK_TOKEN]), dtype=torch.long)
        return context_idxs, target_idx

class SkipGramDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        center_idx = torch.tensor(self.vocab.get(center, self.vocab[UNK_TOKEN]), dtype=torch.long)
        context_idx = torch.tensor(self.vocab.get(context, self.vocab[UNK_TOKEN]), dtype=torch.long)
        return center_idx, context_idx


# Prepare data and Model
if model_name == "cbow":
    model_data = generate_cbow_data(df['lemmas'].tolist(), window_size=window_size)
    dataset = CBOWDataset(model_data, vocab)
    model = CBOWModel(len(vocab), embedding_dim)
else:
    model_data = generate_skipgram_data(df['lemmas'].tolist(), window_size=window_size)
    dataset = SkipGramDataset(model_data, vocab)
    model = SkipGramModel(len(vocab), embedding_dim)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loss, and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_history = []

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for context_idxs, target_idx in dataloader:
        optimizer.zero_grad()
        # forward
        output = model(context_idxs)  # output shape: (batch_size, vocab_size)
        loss = criterion(output, target_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)  # log the average loss
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
# Save the trained model
import os
export_file_name = f"{model_name}_model_{batch_size}b_{learning_rate}l_{epochs}e_{window_size}w.pth"
save_dir = f"models/{category}/{model_name}/{export_file_name}"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), f"{save_dir}/{export_file_name}")
loss_df = pd.DataFrame({'epoch': range(1, epochs + 1), 'loss': loss_history})
loss_df.to_csv(f"{save_dir}/training_loss_log_{export_file_name}.csv", index=False)

plt.plot(range(1, epochs + 1), loss_history, marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/loss_plot_{export_file_name}.png")
plt.show()

print(f"Model saved to {save_dir}")