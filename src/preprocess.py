import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

def tokenize(text):
    """
    Tokenise une chaîne de texte en une liste de mots.

    Args:
        text (str): Texte brut à tokeniser.

    Returns:
        list[str]: Liste de tokens.
    """
    return text.lower().strip().split()

def build_vocab(texts, min_freq=1):
    """
    Construit un vocabulaire à partir d'une liste de textes.

    Args:
        texts (list[str]): Liste de textes.
        min_freq (int): Fréquence minimale pour inclure un mot dans le vocabulaire.

    Returns:
        dict: Dictionnaire {mot: index} avec tokens spéciaux :
              <pad>, <sos>, <eos>, <unk>
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3
    }
    idx = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def encode(text, vocab):
    """
    Encode un texte en indices d'après le vocabulaire.

    Args:
        text (str): Texte brut.
        vocab (dict): Dictionnaire {mot: index}.

    Returns:
        list[int]: Liste d'indices correspondant au texte.
    """
    return [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)]

def decode(indices, idx2word):
    """
    Décode une séquence d'indices en texte lisible.

    Args:
        indices (list[int]): Indices à convertir.
        idx2word (dict): Dictionnaire {index: mot}.

    Returns:
        str: Texte reconstruit sans les tokens spéciaux.
    """
    tokens = [idx2word[idx] for idx in indices if idx in idx2word and idx2word[idx] not in ("<pad>", "<sos>", "<eos>")]
    return " ".join(tokens)

def prepare_input(text, vocab, device):
    """
    Prépare un texte sous forme de tenseur pour le modèle.

    Ajoute les tokens <sos> et <eos> et le convertit en tenseur 2D.

    Args:
        text (str): Texte brut.
        vocab (dict): Vocabulaire utilisé pour l'encodage.
        device (torch.device): Appareil (CPU ou GPU).

    Returns:
        torch.Tensor: Tenseur (1, seq_len)
    """
    tokens = [vocab["<sos>"]] + encode(text, vocab) + [vocab["<eos>"]]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

def filter_prompts(user_input, prompt_list):
    """
    Filtre les prompts qui commencent par le texte saisi (insensible à la casse).

    Args:
        user_input (str): Entrée utilisateur.
        prompt_list (list[str]): Liste des prompts existants.

    Returns:
        list[str]: Prompts correspondants à l'entrée utilisateur.
    """
    user_input = user_input.lower().strip()
    return [p for p in prompt_list if p.lower().startswith(user_input)]
