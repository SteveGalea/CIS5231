import torch
import torch.nn as nn
from torch.utils.data import Dataset

from models.consts import UNK_TOKEN
from .word_embedding import WordEmbeddingModel
import torch.nn.functional as F

# Chat GPT suggested:
torch.set_num_threads(torch.get_num_threads())
torch.backends.mkldnn.enabled = True

# Boilerplate code from ChatGPT, but later adapted to incorporate code from:
# https://leshem-ido.medium.com/skip-gram-word2vec-algorithm-explained-85cd67a45ffa
class SkipGramModel(WordEmbeddingModel):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_idxs):
        # center_idxs: (batch_size)
        # shape: (batch_size, emb_dim)
        embeds = self.embeddings(center_idxs)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Boilerplate code from ChatGPT, but later adapted to incorporate code from
# https://leshem-ido.medium.com/skip-gram-word2vec-algorithm-explained-85cd67a45ffa
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

#https://leshem-ido.medium.com/skip-gram-word2vec-algorithm-explained-85cd67a45ffa
def generate_skipgram_data(tokens_list, window_size=2):
    data = []
    for tokens in tokens_list:
        for i in range(len(tokens)):
            center = tokens[i]
            for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
                if j != i:
                    data.append((center, tokens[j]))
    return data
