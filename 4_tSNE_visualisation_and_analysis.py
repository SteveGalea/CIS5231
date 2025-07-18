import torch
import random
import pickle
from models.consts import UNK_TOKEN
from models.cbow import CBOWModel
from models.skipgram import SkipGramModel

from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
batch_size = 128
model_name = "cbow"  # "cbow"
filepath = f"models/sports/{model_name}/{model_name}_model_{batch_size}b_{lr}l_{epochs}e_{window}w_files/"
filename = f"{model_name}_model_{batch_size}b_{lr}l_{epochs}e_{window}w.pth"

embedding_dim = 256  # or the dimension you used in training
# Load vocabulary

with open("Data/vocab_sports_dict.pkl", "rb") as f:
    vocab = pickle.load(f)

# Prepare data and Model
if model_name == "cbow":
    model = CBOWModel(len(vocab), embedding_dim).to(device)
else:
    model = SkipGramModel(len(vocab), embedding_dim).to(device)

# Load saved state dict
model.load_state_dict(torch.load(filepath + filename, map_location=torch.device('cpu')))
print(model.eval())

# Get embedding weights: shape (vocab_size, embedding_dim)
embeddings = model.embeddings.weight.data.cpu().numpy()

# Reduce dimensionality to 2D

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

embeddings_2d = tsne.fit_transform(embeddings)

# Invert vocab dictionary to get word from index
idx_to_word = {idx: word for word, idx in vocab.items()}

plt.figure(figsize=(14, 14))

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
plt.xlim(-8, 8)   # adjust x-axis range
plt.ylim(-8, 8)   # adjust y-axis range

plt.title("t-SNE Visualization of Word Embeddings")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{filepath}top_{N}_tsne_word_embeddings_{filename}.png")

# Define semantic groups
categories = {
    'Gender': ["man", "woman", "boy", "girl", "male", "female", "men", "women", "womens"],
    'Royalty': ["king", "queen", "man", "woman", "prince", "princess"],
    'Business': ["businessman", "man", "business", "businessmenin"],
    'Sports': ["foot","ball", "football", "soccer", "american", "cricket", "india", "basketball","america","states", "olympics", "major", "tournament", "handball"],
    'Country': ["america", "american", "british", "english", "england", "united", "states", "ozil", "mueller", "germany", "germans", "turkey", "bonnucci", "armbonucci","italy"],
    'Colors': ["yellow", "gold", "red", "green", "blue", "angry"],
    'Position': ["silver", "second", "gold", "first", "third", "bronze", "win", "lose"]
}

for key, word_list in categories.items():
    words_subset = [word for word in word_list if word in vocab]
    indices = [vocab[word] for word in words_subset]
    coords = embeddings_2d[indices]
    plt.figure(figsize=(10, 8))
    for i, word in enumerate(words_subset):
        x, y = coords[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.02, y + 0.02, word, fontsize=10)

    plt.title("t-SNE Plot of Selected Word Embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filepath}{key}_subset_words_tsne_{filename}.png")

data = []
for i in range(len(embeddings_2d)):
    word = idx_to_word.get(i, UNK_TOKEN)
    x, y = embeddings_2d[i]
    data.append((word, x, y))

# Create DataFrame
df = pd.DataFrame(data, columns=['word', 'x', 'y'])

# Save to CSV
csv_path = f"{filepath}tsne_embeddings_{filename}.csv"
df.to_csv(csv_path, index=False)