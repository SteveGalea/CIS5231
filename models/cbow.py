import torch
import torch.nn as nn
from .word_embedding import WordEmbeddingModel
from torch.utils.data import Dataset
from .consts import UNK_TOKEN
import torch.nn.functional as F

torch.set_num_threads(torch.get_num_threads())
torch.backends.mkldnn.enabled = True  # For optimized CPU kernels


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


def generate_cbow_data(tokens_list, window_size=2):
    data = []
    for tokens in tokens_list:
        for i in range(window_size, len(tokens) - window_size):
            context = tokens[i - window_size:i] + tokens[i + 1:i + window_size + 1]
            target = tokens[i]
            data.append((context, target))
    return data
