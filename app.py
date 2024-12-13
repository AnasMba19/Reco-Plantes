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

        /* Personnalisation de la barre latérale */
        [data-testid="stSidebar"] {{
            background: linear-gradient(135deg, #2e8b57, #76c7c0); /* Dégradé vert assorti */
            color: white;
            border-right: 3px solid #006400; /* Bordure élégante */
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.3);
            padding: 20px;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            background-color: #3cb371;
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.2s;
            padding: 12px;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: #4CAF50;
            transform: scale(1.05);
        }}

        [data-testid="stSidebar"] .stSelectbox {{
            background-color: #1e5631;
            color: white;
            border-radius: 5px;
            padding: 8px;
            border: 1px solid #006400;
            font-weight: bold;
        }}

        [data-testid="stSidebar"] .stSelectbox:hover {{
            background-color: #3cb371;
        }}

        [data-testid="stSidebar"] .stFileUploader {{
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
            font-weight: bold;
        }}

        /* Ajout d'espacements et alignements */
        [data-testid="stSidebar"] .stFileUploader, [data-testid="stSidebar"] .stSelectbox {{
            margin-top: 10px;
            margin-bottom: 20px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Appliquer le style en premier
set_custom_style()

# Barre latérale avec titre et options
st.sidebar.title("Reco-Plantes")
if st.sidebar.button("🔄 Réinitialiser"):
    st.experimental_rerun()

model_choice = st.sidebar.selectbox(
    "Choisissez un modèle :", 
    ["ResNet50", "MobileNetV2"]
)

# Modèles disponibles
models = {
    "ResNet50": "models/resnet50_model.keras",
    "MobileNetV2": "models/mobilenetv2_model.keras",
}
model_path = models[model_choice]

model_descriptions = {
    "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
    "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
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
