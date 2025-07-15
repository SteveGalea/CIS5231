import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models.consts import UNK_TOKEN
from models.cbow import CBOWModel
from models.skipgram import SkipGramModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
torch.set_num_threads(torch.get_num_threads())
torch.backends.mkldnn.enabled = True
device = torch.device("cpu")
lr = "0.001"
epochs = "100"
window = "2"
batch_size = 256
model_name = "skipgram" #"cbow"
filepath = f"models/sports/{model_name}/{model_name}_model_{batch_size}b_{lr}l_{epochs}e_{window}w_files/"
filename = f"{model_name}_model_{batch_size}b_{lr}l_{epochs}e_{window}w.pth"

embedding_dim = 256      # or the dimension you used in training
# Load vocabulary

with open("Data/vocab_sports_dict.pkl", "rb") as f:
    vocab = pickle.load(f)


# Re-create the model with same dimensions

# Prepare data and Model
if model_name == "cbow":
    model = CBOWModel(len(vocab), embedding_dim).to(device)
else:
    model = SkipGramModel(len(vocab), embedding_dim).to(device)

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

# Plot only top N words to avoid
n = 30
N = 10000
for i in range(N):
    x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
    word = idx_to_word.get(i, UNK_TOKEN)
    plt.scatter(x, y, s=5)

# Randomly select n unique indices to label
label_indices = random.sample(range(N), n)

for i in label_indices:
    x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
    word = idx_to_word.get(i, UNK_TOKEN)
    plt.text(x + 0.002, y + 0.002, word, fontsize=8)

# Zoom into a specific region (customize these values)
# plt.xlim(-1, 10)   # adjust x-axis range
# plt.ylim(-1, 10)   # adjust y-axis range

plt.title("t-SNE Visualization of Word Embeddings")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{filepath}top_{N}_tsne_word_embeddings_{filename}.png")
#plt.show()

from sklearn.cluster import KMeans
num_clusters = 30  # adjust as you like
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Find the closest word index to each cluster center
center_indices = []
for center in kmeans.cluster_centers_:
    # Compute distances from center to all embeddings
    distances = np.linalg.norm(embeddings - center, axis=1)
    closest_idx = np.argmin(distances)
    center_indices.append(closest_idx)

# Plot only the cluster centers in 2D t-SNE space
plt.figure(figsize=(14, 10))

for i in center_indices:
    x, y = embeddings_2d[i, 0], embeddings_2d[i, 1]
    word = idx_to_word.get(i, UNK_TOKEN)
    plt.scatter(x, y, s=50, marker='x', color='red')
    plt.text(x + 0.002, y + 0.002, word, fontsize=12, fontweight='bold', color='black')

plt.title(f"t-SNE Visualization of {num_clusters} Cluster Center Words")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{filepath}cluster_centers_{num_clusters}_tsne_{filename}.png")
plt.show()