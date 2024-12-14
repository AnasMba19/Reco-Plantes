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
    # URL du background depuis GitHub
    background_url = "https://raw.githubusercontent.com/AnasMba19/Reco-Plantes/main/assets/background.jpg"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        /* Global style for the app */
        .stApp {{
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(120deg, rgba(46,139,87, 0.9), rgba(255,255,255, 0.9)), 
                        url("{background_url}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Sidebar style */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #006400, #2e8b57);
            color: white;
            border-right: 3px solid #004d00;
            padding: 20px;
        }}

        [data-testid="stSidebar"] h1, h2, h3 {{
            color: white;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 15px;
        }}

        [data-testid="stSidebar"] .stButton > button {{
            background: #4CAF50;
            border: none;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, background-color 0.2s ease;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background: #45a049;
            transform: scale(1.05);
        }}

        /* Title and Header */
        h1.title {{
            color: #2e8b57;
            font-size: 48px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            margin-top: 10px;
            margin-bottom: 20px;
        }}

        h2.subtitle {{
            color: #006400;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }}

        .content-block {{
            background: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}

        footer {{
            background: #004d00;
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}

        footer a {{
            color: #80FF80;
            text-decoration: none;
        }}

        footer a:hover {{
            text-decoration: underline;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Appliquer le style personnalisé
set_custom_style()

# Barre latérale
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
st.sidebar.info(f"ℹ️ **Modèle choisi :** {model_descriptions[model_choice]}")

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Titre principal
st.markdown('<h1 class="title">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

# Instructions
st.markdown(
    """
    <div class="content-block">
        <h2 class="subtitle">Bienvenue dans l'application !</h2>
        <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
        <p><strong>Comment utiliser :</strong></p>
        <ul>
            <li>1️⃣ Téléchargez une image via la barre latérale.</li>
            <li>2️⃣ Sélectionnez un modèle dans le menu latéral.</li>
            <li>3️⃣ Le résultat s'affichera automatiquement après analyse.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Analyse et résultats
if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    model = load_model(model_path)
    input_shape = model.input_shape[1:3]
    image_array = preprocess_image(uploaded_file, target_size=input_shape)
    predicted_class, confidence = predict_image(model, image_array)
    st.markdown(
        f"""
        <div class="content-block">
            <h2 class="subtitle">Résultat de l'Analyse</h2>
            <p>✅ Classe prédite : <strong>{predicted_class}</strong></p>
            <p>📊 Confiance : <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Veuillez télécharger une image valide.")

# Animation Lottie
if lottie_json:
    st_lottie(lottie_json, height=200)

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
