import torch
from src.preprocess import tokenize, prepare_input, decode
import torch.nn.functional as F
import re
import random

def generate_response(prompt, model, vocab, idx2word, max_len=50, temperature=1.0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Préparer l'entrée
    src_tensor = prepare_input(prompt, vocab, device)

    with torch.no_grad():
        embedded_src = model.embedding(src_tensor)
        _, (hidden, cell) = model.encoder(embedded_src)

    # Génération avec sampling
    generated_indices = [vocab["<sos>"]]

    for _ in range(max_len):
        current_input = torch.tensor([[generated_indices[-1]]], dtype=torch.long).to(device)
        with torch.no_grad():
            embedded = model.embedding(current_input)
            output, (hidden, cell) = model.decoder(embedded, (hidden, cell))
            logits = model.fc_out(output.squeeze(1)) / temperature
            probs = F.softmax(logits, dim=1)
            predicted_idx = torch.multinomial(probs, num_samples=1).item()

        if predicted_idx == vocab["<eos>"]:
            break

        generated_indices.append(predicted_idx)

    # Texte brut généré
    raw_output = decode(generated_indices[1:], idx2word)

    # Nettoyage du texte
    raw_output_cleaned = raw_output.replace("\n", " ").replace("  ", " ").strip().lower()

    # Champs attendus
    fields = {
        "description": "📌 Description",
        "durée estimée": "⏱️ Durée estimée",
        "qui peut faire la demande": "👤 Qui peut faire la demande",
        "institution en charge": "🏛️ Institution en charge",
        "démarches": "📝 Démarches"
    }

    # Titre à partir du prompt
    title = prompt.replace("Je veux faire une demande de :", "").strip().capitalize()
    structured_output = f"🔹 Demande de : {title}\n\n"

    # Extraction de chaque champ avec regex
    for key, label in fields.items():
        pattern = re.compile(f"{key}\\s*:\\s*(.*?)(?=(description|durée estimée|qui peut faire la demande|institution en charge|démarches|$))", re.IGNORECASE)
        match = pattern.search(raw_output_cleaned)
        if match:
            content = match.group(1).strip(" .:")
            structured_output += f"{label} : {content}\n"

    return structured_output



def interactive_loop(model, vocab, idx2word):
    """
    Lance une boucle interactive pour tester le modèle avec des entrées utilisateur.

    Args:
        model (nn.Module): Le modèle entraîné.
        vocab (dict): Dictionnaire mot -> index.
        idx2word (dict): Dictionnaire index -> mot.
    """
    print("Entrez un prompt (tapez 'exit' pour quitter):")
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ("exit", "quit"):
            break
        response = generate_response(user_input, model, vocab, idx2word)
        print(f"Réponse générée : {response}")
