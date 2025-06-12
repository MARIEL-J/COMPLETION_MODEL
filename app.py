import streamlit as st
import torch
import json
from src.preprocess import tokenize, build_vocab, encode, decode, prepare_input, filter_prompts
from src.model import Seq2SeqModel
from src.infer import generate_response

# Chargement des données pour suggestions dynamiques
with open("data/prompts_completions.jsonl", "r", encoding="utf-8") as f:
    prompts_data = [json.loads(line)["prompt"] for line in f]

# Chemins vers le vocabulaire et le modèle sauvegardé
vocab_path = "models/vocab.json"
model_path = "models/seq2seq_model.pth"

@st.cache_data
def load_resources():
    """
    Charge le vocabulaire, initialise le modèle Seq2Seq et charge ses poids.
    Vérifie la présence des tokens spéciaux nécessaires.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    vocab = {k: int(v) for k, v in vocab.items()}
    idx2word = {v: k for k, v in vocab.items()}

    required_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
    for token in required_tokens:
        if token not in vocab:
            st.error(f"Le token spécial '{token}' est manquant dans le vocabulaire.")
            st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqModel(vocab_size=len(vocab), embed_dim=128, hidden_dim=256, pad_idx=vocab["<pad>"])
    model = model.to(device)

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        st.stop()

    return vocab, idx2word, model, device

# Chargement initial des ressources
vocab, idx2word, model, device = load_resources()

# Titre principal
st.title("🔮 Complétion intelligente de phrases administratives")

with st.expander("ℹ️ À propos de l'application"):
    st.markdown("""
    Cette application utilise un modèle de type **Seq2Seq** entraîné sur un jeu de paires `prompt → complétion`
    pour générer automatiquement des phrases administratives complètes à partir de débuts de phrase.
    
                
    💡 Commencez à taper votre demande, cliquez sur une suggestion ou lancez la complétion automatique.
    """)

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

if "auto_complete" not in st.session_state:
    st.session_state["auto_complete"] = False

# Champ texte pour la saisie
user_input = st.text_input("Saisissez une demande incomplète...", st.session_state["user_input"])

# Suggestions dynamiques
suggestions = filter_prompts(user_input, prompts_data)[:5]
if suggestions:
    st.markdown("**💡 Suggestions tirées du jeu d'entraînement :**")
    for i, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_btn_{i}"):
            st.session_state["user_input"] = suggestion
            st.session_state["auto_complete"] = True
            st.rerun()

# Complétion manuelle
if st.button("✨ Compléter la demande") and user_input:
    st.session_state["user_input"] = user_input
    st.session_state["auto_complete"] = True
    st.rerun()

# Génération automatique après clic sur suggestion ou bouton
if st.session_state.get("auto_complete") and st.session_state["user_input"]:
    try:
        result = generate_response(
            st.session_state["user_input"], model, vocab, idx2word, device=device
        )
        st.markdown("### ✍️ Complétion proposée :")
        st.success(result)

        with st.expander("🔍 Détails internes (tokens encodés)"):
            tokens = tokenize(st.session_state["user_input"])
            indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
            st.write("Tokens :", tokens)
            st.write("Indices :", indices)

    except Exception as e:
        st.error(f"Une erreur est survenue pendant la génération : {e}")
    finally:
        st.session_state["auto_complete"] = False  # Réinitialiser

