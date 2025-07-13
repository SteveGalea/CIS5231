import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
UNK_TOKEN = "<UNK>"

filepath = "models/sports/cbow/"
filename = "cbow_model_256b_0.01l_25e_2w.pth/cbow_model_256b_0.01l_25e_2w.pth"
# Load vocabulary
with open("Data/vocab_sports_dict.pkl", "rb") as f:
    vocab = pickle.load(f)


class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(WordEmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def get_embedding(self, input_ids):
        return self.embeddings(input_ids)
# Example CBOW model class (adapt if yours is different)
class CBOWModel(WordEmbeddingModel):
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

# Re-create the model with same dimensions
vocab_size = len(vocab)  # use the actual vocab size
embedding_dim = 256      # or the dimension you used in training
model = CBOWModel(vocab_size, embedding_dim)

# Load saved state dict
model.load_state_dict(torch.load(filepath+filename))
print(model.eval())

# Get embedding weights: shape (vocab_size, embedding_dim)
embeddings = model.embeddings.weight.data.cpu().numpy()

from sklearn.manifold import TSNE
import numpy as np

# Reduce dimensionality to 2D

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

embeddings_2d = tsne.fit_transform(embeddings)
import matplotlib.pyplot as plt

# Invert vocab dictionary to get word from index
idx_to_word = {idx: word for word, idx in vocab.items()}

plt.figure(figsize=(14, 10))

# Plot only top N words to avoid clutter
N = 500
for i in range(N):
    x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
    word = idx_to_word.get(i, "<UNK>")
    plt.scatter(x, y, s=5)
    plt.text(x+0.002, y+0.002, word, fontsize=8)

# Zoom into a specific region (customize these values)
plt.xlim(-0.8, 0.8)   # adjust x-axis range
plt.ylim(-0.8, 0.8)   # adjust y-axis range

plt.title("t-SNE Visualization of Word Embeddings")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{filepath}top_{N}_tsne_word_embeddings_{filename}.png")
plt.show()


# # interactive
# # Assume you have 2D embeddings and labels
# df = pd.DataFrame({
#     "x": embeddings_2d[:, 0],
#     "y": embeddings_2d[:, 1],
#     "word": [idx_to_word.get(i, "<UNK>") for i in range(len(embeddings_2d))]
# })
# import plotly.express as px
# fig = px.scatter(df, x="x", y="y", text="word", title="2D Word Embeddings")
# fig.update_traces(textposition='top center', marker=dict(size=5))
# fig.show()