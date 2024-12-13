import streamlit as st
from models.main import load_model, preprocess_image, predict_image, class_names

# Fonction pour définir un style global avec un fond en dégradé
def set_custom_background_and_style():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        .stApp {
            background: linear-gradient(120deg, #28a745, #ffc107); /* Dégradé vert -> jaune */
            color: #FFFFFF; /* Texte blanc */
            font-family: 'Roboto', sans-serif; /* Police élégante */
            padding: 20px;
        }
        h1, h2, h3, p {
            color: #FFFFFF;
            background-color: #006400; /* Fond vert foncé */
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
            background-color: #45a049;
            transform: scale(1.05);
        }
        .instructions {
            margin-bottom: 20px;
            line-height: 1.6;
        }
        footer {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Appeler la fonction pour appliquer le style
set_custom_background_and_style()

# Titre principal
st.title("🌿 Reconnaissance de Maladies des Plantes")

# Description de l'application
st.markdown(
    """
    <div class="instructions">
    Bienvenue ! Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes.

    **Instructions :**
    <ul>
        <li>📂 Téléchargez une image en cliquant sur le bouton **Browse Files**.</li>
        <li>🛠️ Sélectionnez un modèle dans le menu déroulant.</li>
        <li>✨ Consultez les résultats instantanément !</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
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

# Description dynamique des modèles
model_descriptions = {
    "Model 1": "Modèle basé sur MobileNet avec régularisation L2.",
    "Model 2": "Modèle MobileNet avec augmentation des données.",
    "Model 3": "Modèle CNN développé par Leila.",
    "Model 4": "Modèle CNN avec optimisation avancée.",
}
st.info(f"ℹ️ **Description du modèle choisi :** {model_descriptions[model_choice]}")

# Chargement du modèle
model = load_model(model_path)

# Téléchargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png"])
if uploaded_file is not None:
    with st.spinner("Analyse en cours..."):
        st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
        input_shape = model.input_shape[1:3]
        image_array = preprocess_image(uploaded_file, target_size=input_shape)
        predicted_class, confidence = predict_image(model, image_array)
        st.success(f"✅ **Résultat :** {predicted_class} ({confidence:.2f}%)")

    # Ajouter une section pour afficher des statistiques
    st.markdown("### Statistiques")
    st.write(f"Confiance de la prédiction : {confidence:.2f}%")
else:
    st.warning("⚠️ Veuillez télécharger une image valide.")

# Footer avec crédits
st.markdown(
    """
    <footer>
        © 2024 Reconnaissance des Maladies des Plantes | Développé par Anas Mba19
        <br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">GitHub</a> | 
        <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </footer>
    """,
    unsafe_allow_html=True
)
