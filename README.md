# 📄 COMPLETION_MODEL_PROJET

Un projet complet de génération de complétions textuelles administratives à l’aide d’un modèle **seq2seq** personnalisé (basé sur Flan-T5 ou similaire). Il inclut la préparation des données, l’entraînement du modèle, l’inférence, et une application d’interface utilisateur simple  codé avec le framework streamlit.

## 🚀 Objectif

Automatiser la **génération** et la **reformulation structurée** de textes administratifs à partir d’une demande utilisateur, selon une structure normalisée (ex. : description, durée estimée, institution en charge, démarches...).

## 🗂️ Structure du projet

```bash
COMPLETION_MODEL_PROJET/
├── data/                          # Données d'entraînement et prompts de test
│   └── prompts_completions.jsonl
├── models/                        # Modèles entraînés et vocabulaire
│   ├── seq2seq_model.pth
│   └── vocab.json
├── notebooks/                     # Notebook Jupyter pour nettoyage, exploration, preprocessing et construction du modèle
│   └── data_preprocessing.ipynb
├── src/                           # Code source principal
│   ├── infer.py                  # Inférence : génération & reformulation
│   ├── model.py                  # Architecture du modèle seq2seq
│   ├── preprocess.py             # Prétraitement des données
│   └── train.py                  # Script d'entraînement
├── app.py                         # Interface utilisateur
├── README.md                      # Documentation du projet
└── requirements.txt               # Dépendances Python
```

## ⚙️ Installation

1. **Cloner le dépôt**  
```bash
git clone https://github.com/MARIEL-J/COMPLETION_MODEL_PROJET.git
cd COMPLETION_MODEL_PROJET
```

2. **Créer un environnement virtuel (optionnel mais recommandé)**  
```bash
python -m venv venv
source venv/bin/activate  # Sous Linux/macOS
venv\Scripts\activate     # Sous Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 🧠 Entraînement du modèle

```bash
python src/train.py
```

Paramètres personnalisables (dans le script ou via argparse) : nombre d’épochs, batch size, taux d’apprentissage, etc.

## ✨ Génération + Reformulation

Lancer le script d’inférence à partir d’un prompt :

```bash
python src/infer.py --prompt "Je veux faire une demande de : carte d'identité"
```

Le modèle :
1. Génère un texte brut.
2. Le reformule dans un format clair et structuré.

## 🖥️ Interface (optionnelle)

```bash
python app.py
```

Lance une interface pour directement tester l'application avec entrer d'un prompt.

## 📊 Données

Les données utilisées pour l'entraînement sont au format JSONL :

```json
{"prompt": "Je veux faire une demande de : acte de naissance", "completion": "description : ..."}
```

Un notebook de préparation est fourni dans `notebooks/data_preprocessing.ipynb`.

## ✅ Format de sortie attendu

Exemple de reformulation :

```
🔹 Demande de : Acte de naissance

📌 Description : Permet d’obtenir un justificatif officiel de naissance.

⏱️ Durée estimée : 48h

👤 Qui peut faire la demande : Tout citoyen béninois.

🏛️ Institution en charge : Mairie de la commune de naissance.

📝 Démarches : Remplir le formulaire en ligne et fournir une pièce d’identité.
```

## 📦 Modèle utilisé

- Architecture seq2seq avec attention.
