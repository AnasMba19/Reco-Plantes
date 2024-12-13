import streamlit as st
from models.main import load_model, preprocess_image, predict_image, class_names

# Fonction pour définir un fond en dégradé avec un style global
def set_custom_background_and_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #2ecc71, #f1c40f); /* Dégradé vert -> jaune */
            color: #FFFFFF; /* Texte blanc */
            padding: 20px;
        }
        h1, h2, h3, p {
            color: #FFFFFF; /* Texte blanc */
            background-color: rgba(0, 0, 0, 0.5); /* Fond noir translucide */
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        }
        .stButton > button {
            background-color: #4CAF50; /* Vert bouton */
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, background-color 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #45a049; /* Vert bouton survol */
            transform: scale(1.05);
        }
        .stMarkdown {
            border: 2px solid #3CB371; /* Bordure vert clair */
            padding: 15px;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Appeler la fonction pour définir le style
set_custom_background_and_style()

# Titre principal
st.title("🌿 Reconnaissance de Maladies des Plantes")

# Description de l'application
st.markdown(
    """
    Bienvenue ! Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes.

    **Instructions :**
    1. Téléchargez une image en cliquant sur le bouton "Browse Files".
    2. Choisissez un modèle dans le menu déroulant.
    3. Obtenez des résultats instantanément ! 🌟
    """
)

# Séparateur visuel
st.divider()

# Ajouter un bouton pour réinitialiser l'application
if st.button("🔄 Réinitialiser"):
    st.experimental_rerun()

# Sélection du modèle
model_choice = st.selectbox("Choisissez un modèle :", ["Model 1", "Model 2", "Model 3", "Model 4"])

# Mapping des modèles
models = {
    "Model 1": "models/Anas_Essai_1_MOB_L2.keras",
    "Model 2": "models/Anas_Essai_1_MOB_Repeat.keras",
    "Model 3": "models/leila_best_model_cnn_TEM3.keras",
    "Model 4": "models/model_cnn_4_best.keras",
}
model_path = models[model_choice]

# Chargement du modèle
model = load_model(model_path)

# Téléchargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    input_shape = model.input_shape[1:3]
    image_array = preprocess_image(uploaded_file, target_size=input_shape)
    predicted_class, confidence = predict_image(model, image_array)
    st.success(f"Résultat : {predicted_class} ({confidence:.2f}%)")

    # Ajouter une section pour afficher des statistiques
    st.markdown("### Statistiques")
    st.write(f"Confiance de la prédiction : {confidence:.2f}%")
