import torch
import torch.nn as nn

class Seq2SeqModel(nn.Module):
    """
    Modèle Seq2Seq simple avec un encodeur et un décodeur LSTM,
    utilisé pour des tâches de complétion de texte ou de traduction.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, pad_idx=0):
        """
        Initialise le modèle Seq2Seq.

        Args:
            vocab_size (int): Taille du vocabulaire.
            embed_dim (int): Dimension des embeddings.
            hidden_dim (int): Dimension des états cachés des LSTMs.
            pad_idx (int): Index du token de padding pour l'embedding.
        """
        super(Seq2SeqModel, self).__init__()
        
        # Couche d'embedding partagée pour l'encodeur et le décodeur
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Encodeur LSTM
        self.encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Décodeur LSTM
        self.decoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Couche linéaire de projection vers l'espace du vocabulaire
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        """
        Propagation avant du modèle.

        Args:
            src (Tensor): Séquences source (prompts), taille (batch_size, src_seq_len)
            tgt (Tensor): Séquences cibles (completions), taille (batch_size, tgt_seq_len)

        Returns:
            Tensor: Logits de sortie, taille (batch_size, tgt_seq_len, vocab_size)
        """
        # Embedding des séquences source et cible
        embedded_src = self.embedding(src)  # (batch, src_len, embed_dim)
        embedded_tgt = self.embedding(tgt)  # (batch, tgt_len, embed_dim)

        # Encodage
        _, (hidden, cell) = self.encoder(embedded_src)  # on récupère hidden/cell final

        # Décodage (teacher forcing avec la vraie cible)
        outputs, _ = self.decoder(embedded_tgt, (hidden, cell))  # (batch, tgt_len, hidden_dim)

        # Projection linéaire vers le vocabulaire
        logits = self.fc_out(outputs)  # (batch, tgt_len, vocab_size)

        return logits
