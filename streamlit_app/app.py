import streamlit as st
import torch
import json
import os
from src.preprocess import tokenize, build_vocab, encode, decode, prepare_input, filter_prompts
from src.model import Seq2SeqModel
from src.infer import generate_completion

# Chargement des données pour suggestions dynamiques
with open("../data/prompts_completions.jsonl", "r", encoding="utf-8") as f:
    prompts_data = [json.loads(line)["prompt"] for line in f]

# Chemins vers le vocabulaire et le modèle sauvegardé
vocab_path = "../models/vocab.json"
model_path = "../models/seq2seq_model.pth"

@st.cache_data
def load_resources():
    """
    Charge le vocabulaire, crée le dictionnaire inverse idx2word, 
    initialise le modèle Seq2Seq et charge ses poids sauvegardés.

    Retourne :
        vocab (dict): mapping mot -> index
        idx2word (dict): mapping index -> mot
        model (Seq2SeqModel): modèle chargé en mémoire
        device (torch.device): appareil d'exécution (GPU ou CPU)
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idx2word = {idx: word for word, idx in vocab.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqModel(len(vocab), 128, 256, vocab["<pad>"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    return vocab, idx2word, model, device

# Chargement initial des ressources
vocab, idx2word, model, device = load_resources()

# Titre de l'application
st.title("🔮 Complétion intelligente de phrases administratives")

# Champ texte pour la saisie utilisateur
user_input = st.text_input("Saisissez une demande...")

# Affichage des suggestions dynamiques basées sur la saisie
suggestions = filter_prompts(user_input, prompts_data)[:5]
if suggestions:
    st.markdown("**Suggestions :**")
    for suggestion in suggestions:
        if st.button(suggestion):
            user_input = suggestion
            st.session_state["user_input"] = suggestion

# Bouton pour lancer la génération de complétion
if st.button("Compléter la demande") and user_input:
    result = generate_completion(model, user_input, vocab, idx2word, device)
    st.markdown("### ✍️ Complétion proposée :")
    st.write(result)
