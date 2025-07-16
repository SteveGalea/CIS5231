import torch.nn as nn
import random
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.cbow import CBOWModel, CBOWDataset, generate_cbow_data
from models.skipgram import SkipGramModel, SkipGramDataset, generate_skipgram_data


torch.set_num_threads(torch.get_num_threads())  # Use all available CPU threads
torch.backends.mkldnn.enabled = True  # Enable MKLDNN (optimized CPU kernels)
if __name__ == "__main__":
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)
    #https://medium.com/@RobuRishabh/learning-word-embeddings-with-cbow-and-skip-gram-b834bde18de4
    #https://www.geeksforgeeks.org/nlp/word-embeddings-in-nlp/
    embedding_dim = 256
    device = torch.device("cpu")
    # Load DataFrame
    model_name = "cbow"
    #  hyperparameters
    batch_size = 256
    learning_rate = 0.01
    epochs = 50
    window_size = 4

    category = "sports"  # business
    suffix_category = f"{category}_"
    data_file = f"Data/preprocessed_{category}_articles.pkl"
    vocab_file = f"Data/vocab_{category}_dict.pkl"

    df = pd.read_pickle(data_file)

    export_file_name = f"{model_name}_model_{batch_size}b_{learning_rate}l_{epochs}e_{window_size}w.pth"
    export_folder_name = f"{model_name}_model_{batch_size}b_{learning_rate}l_{epochs}e_{window_size}w_files"
    save_dir = f"models/{category}/{model_name}/{export_folder_name}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    # Load vocabulary
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)

    # Prepare data and Model
    if model_name == "cbow":
        model_data = generate_cbow_data(df['lemmas'].tolist(), window_size=window_size)
        dataset = CBOWDataset(model_data, vocab)
        model = CBOWModel(len(vocab), embedding_dim).to(device)
    else:
        model_data = generate_skipgram_data(df['lemmas'].tolist(), window_size=window_size)
        dataset = SkipGramDataset(model_data, vocab)
        model = SkipGramModel(len(vocab), embedding_dim).to(device)

    #model = torch.compile(model)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4) # use of minibatches

    # Loss, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for context_idxs, target_idx in dataloader:
            context_idxs = context_idxs.to(device)
            target_idx = target_idx.to(device)
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
        if (epoch + 1) % 10 == 0:
            os.makedirs(f"{save_dir}/checkpoints/checkpoint_{epoch}e", exist_ok=True),
            torch.save(model.state_dict(), f"{save_dir}/checkpoints/checkpoint_{epoch}e/checkpoint_{epoch}e_{export_file_name}")
            loss_df = pd.DataFrame({'epoch': range(1, epoch + 2), 'loss': loss_history})
            loss_df.to_csv(f"{save_dir}/checkpoints/checkpoint_{epoch}e/training_loss_log_checkpoint_{epoch}e_{export_file_name}.csv", index=False)

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