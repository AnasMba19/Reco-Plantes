import streamlit as st
from models.main import load_model, preprocess_image, predict_image, class_names

try:
    from streamlit_lottie import st_lottie
    import requests
    # Animation lottie
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_pkscqlmk.json"
    lottie_json = requests.get(lottie_url).json()
except ImportError:
    lottie_json = None

def set_custom_style():
    # URL du background depuis GitHub (vérifiez bien l'URL raw)
    background_url = "https://raw.githubusercontent.com/AnasMba19/Reco-Plantes/main/assets/background.jpg"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        .stApp {{
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.25)), 
                        url("{background_url}") no-repeat center center fixed !important;
            background-size: cover !important;
        }}

        /* Rendre le fond transparent sur le container principal pour voir le background */
        [data-testid="stAppViewContainer"] > .main {{
            background: none !important;
            padding: 40px;
        }}

        .content-block {{
            background-color: rgba(34, 139, 34, 0.85);
            color: #FFFFFF;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}

        h1, h2, h3, p {{
            margin: 0 0 10px 0;
        }}

        .stButton > button {{
            background-color: #4CAF50; 
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, background-color 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: #45a049;
            transform: scale(1.03);
        }}

        .banner {{
            display: flex;
            align-items: center;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .banner img {{
            max-height: 80px;
            border-radius: 8px;
        }}

        footer {{
            text-align: center;
            margin-top: 40px;
            font-size: 14px;
            color: #f0f0f0;
        }}

        footer a {{
            color: #f0f0f0;
            text-decoration: none;
            font-weight: bold;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Appliquer le style en premier
set_custom_style()

# Titre principal avec logo
st.markdown(
    """
    <div class="banner">
        <img src="https://raw.githubusercontent.com/AnasMba19/Reco-Plantes/main/assets/logo_recoplantes.jpg" alt="Logo RecoPlantes">
        <h1>Reconnaissance de Maladies des Plantes</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Instructions
st.markdown(
    '<div class="content-block">'
    '<p>Bienvenue ! Cette application utilise des modèles d\'apprentissage profond pour détecter les maladies des plantes.</p>'
    '<p><strong>Instructions :</strong></p>'
    '<ul>'
    '<li>📂 Téléchargez une image via la barre latérale.</li>'
    '<li>🛠️ Sélectionnez un modèle dans la barre latérale.</li>'
    '<li>✨ Rendez-vous dans l\'onglet "Analyser" pour voir le résultat instantanément !</li>'
    '</ul>'
    '</div>',
    unsafe_allow_html=True
)

# Barre latérale
st.sidebar.title("Paramètres")
if st.sidebar.button("🔄 Réinitialiser"):
    st.experimental_rerun()

model_choice = st.sidebar.selectbox("Choisissez un modèle :", ["Model 1", "Model 2", "Model 3", "Model 4"])

# Modèles disponibles
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
st.info(f"ℹ️ **Description du modèle choisi :** {model_descriptions[model_choice]}")

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    model = load_model(model_path)
    input_shape = model.input_shape[1:3]
    image_array = preprocess_image(uploaded_file, target_size=input_shape)
    predicted_class, confidence = predict_image(model, image_array)
    st.success(f"✅ Résultat : {predicted_class} ({confidence:.2f}%)")
else:
    st.warning("⚠️ Veuillez télécharger une image valide.")

# Footer
st.markdown(
    """
    <footer>
        © 2024 Reconnaissance des Maladies des Plantes | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI
        <br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">GitHub</a> | 
        <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </footer>
    """,
    unsafe_allow_html=True
)
