import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataset, pad_idx, n_epochs=10, lr=1e-3, batch_size=32, device=None):
    """
    Entraîne un modèle de type séquence (ex : Transformer ou RNN).

    Args:
        model (nn.Module): Le modèle PyTorch à entraîner.
        dataset (Dataset): Dataset retournant des triplets (src, tgt_in, tgt_out).
        pad_idx (int): Index du token <pad> à ignorer dans la perte.
        n_epochs (int): Nombre total d’époques d’entraînement.
        lr (float): Taux d’apprentissage.
        batch_size (int): Taille des batches.
        device (torch.device): Appareil cible (CPU ou GPU).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0

        for src, tgt_in, tgt_out in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

            optimizer.zero_grad()
            output = model(src, tgt_in)  # [batch_size, seq_len, vocab_size]

            # Mise en forme pour CrossEntropy : [batch_size * seq_len, vocab_size] vs [batch_size * seq_len]
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}/{n_epochs}] Average Loss: {avg_loss:.4f}")