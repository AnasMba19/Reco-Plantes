import streamlit as st
from models.main import load_model, preprocess_image, predict_image, class_names

# Optionnel : si tu veux des animations Lottie, assure-toi d'installer le package
try:
    from streamlit_lottie import st_lottie
    import requests
    # On va charger une petite animation Lottie, par exemple une feuille dans le vent
    # Tu peux trouver d’autres animations sur https://lottiefiles.com/
    lottie_url = "https://assets5.lottiefiles.com/packages/lf20_tljjahia.json"
    lottie_json = requests.get(lottie_url).json()
except ImportError:
    lottie_json = None

def set_custom_background_and_style():
    # Ex : Ajouter une image de fond subtile.
    # Remplace l’URL par une image libre de droits, par exemple.
    background_url = "https://images.unsplash.com/photo-1441906363162-903afd0d3d52?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        .stApp {{
            background: url("{background_url}") no-repeat center center fixed;
            background-size: cover;
            color: #FFFFFF; 
            font-family: 'Roboto', sans-serif;
            padding: 20px;
        }}
        h1, h2, h3, p {{
            color: #FFFFFF;
            background-color: rgba(0,100,0, 0.8);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        }}
        .stButton > button {{
            background-color: #4CAF50; 
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, background-color 0.2s ease;
        }}
        .stButton > button:hover {{
            background-color: #45a049;
            transform: scale(1.05);
        }}
        .instructions {{
            margin-bottom: 20px;
            line-height: 1.6;
        }}
        footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #FFFFFF;
        }}
        /* Carte style */
        .result-card {{
            background-color: rgba(0,100,0,0.8);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_background_and_style()

# Barre latérale
st.sidebar.title("Paramètres")
if st.sidebar.button("🔄 Réinitialiser"):
    st.experimental_rerun()

model_choice = st.sidebar.selectbox("Choisissez un modèle :", ["Model 1", "Model 2", "Model 3", "Model 4"])

models = {
    "Model 1": "models/Anas_Essai_1_MOB_L2.keras",
    "Model 2": "models/Anas_Essai_1_MOB_Repeat.keras",
    "Model 3": "models/leila_best_model_cnn_TEM3.keras",
    "Model 4": "models/model_cnn_4_best.keras",
}
model_path = models[model_choice]

model_descriptions = {
    "Model 1": "Modèle basé sur MobileNet avec régularisation L2.",
    "Model 2": "Modèle MobileNet avec augmentation des données.",
    "Model 3": "Modèle CNN développé par Leila.",
    "Model 4": "Modèle CNN avec optimisation avancée.",
}

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Titre principal
st.title("🌿 Reconnaissance de Maladies des Plantes")

# Création des onglets
tab_instructions, tab_analyse = st.tabs(["📖 Instructions", "🔬 Analyse"])

with tab_instructions:
    st.markdown(
        """
        <div class="instructions">
        Bienvenue ! Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes.

        **Instructions :**
        <ul>
            <li>📂 Téléchargez une image via la barre latérale.</li>
            <li>🛠️ Sélectionnez un modèle dans la barre latérale.</li>
            <li>✨ Allez dans l'onglet "Analyse" pour voir le résultat instantanément !</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(f"ℹ️ **Description du modèle choisi :** {model_descriptions[model_choice]}")
    # Si lottie_json est dispo, on affiche une animation
    if lottie_json:
        st_lottie(lottie_json, height=200)

with tab_analyse:
    # Chargement du modèle uniquement si on a un fichier
    if uploaded_file is not None:
        with st.spinner("Analyse en cours..."):
            model = load_model(model_path)
            st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
            input_shape = model.input_shape[1:3]
            image_array = preprocess_image(uploaded_file, target_size=input_shape)
            predicted_class, confidence = predict_image(model, image_array)

        # Affichage du résultat dans une "carte"
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"✅ **Résultat :** {predicted_class}")
        st.markdown("**Confiance :**")
        st.progress(confidence / 100.0)  # Barre de progression pour le pourcentage
        st.write(f"{confidence:.2f}%")

        # Lien "En savoir plus" (à personnaliser)
        st.markdown("[En savoir plus sur cette maladie](https://www.google.com)")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Veuillez télécharger une image valide dans la barre latérale.")

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
