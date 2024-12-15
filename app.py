import os
import streamlit as st
import base64
from models.main import load_model, preprocess_image, predict_image, class_names
from PIL import Image
import time

# Cacher les ressources coûteuses
@st.cache_resource
def load_cached_model(path):
    return load_model(path)

def set_custom_style():
    # URL du background depuis GitHub
    background_url = "https://raw.githubusercontent.com/AnasMba19/Reco-Plantes/main/assets/background.jpg"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Montserrat:wght@600&display=swap');

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

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {{
            color: white;
            font-weight: bold;
        }}

        /* Result styles */
        .result-success {{
            background-color: #d4edda;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .result-warning {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .result-error {{
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }}

        /* Footer styles */
        footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #F5F5DC;
            background-color: #8B4513;
            padding: 10px;
            border-radius: 10px;
        }}
        footer a {{
            text-decoration: none;
            color: #F5F5DC;
            margin: 0 5px;
        }}
        footer a:hover {{
            color: #FFD700;
            text-decoration: underline;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Vérification des fichiers uploadés
def validate_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image.verify()
        return True
    except Exception:
        return False

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        st.error(f"⚠️ Erreur lors de l'encodage de l'image : {e}")
        return None

# Appliquer les styles personnalisés
set_custom_style()

# Sidebar
st.sidebar.title("Reco-Plantes")

model_choice = st.sidebar.selectbox(
    "Choisissez un modèle :",
    ["ResNet50 🖼️", "MobileNetV2 ⚡"]
)

# Modèles
models = {
    "ResNet50": "models/resnet50_model.keras",
    "MobileNetV2": "models/mobilenetv2_model.keras",
}

# Normaliser le choix du modèle pour correspondre aux clés du dictionnaire
normalized_model_choice = model_choice.split()[0]  # Extrait "ResNet50" ou "MobileNetV2"
model_path = models[normalized_model_choice]
model_descriptions = {
    "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
    "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
}

# Description du modèle dans la sidebar
st.sidebar.markdown(
    f"""
    <div style="color:black; font-size:16px;">
        ℹ️ {model_descriptions[normalized_model_choice]}
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Titre principal
st.markdown('<h1 style="color:#004d00; text-align:center;">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

if uploaded_file:
    if validate_image(uploaded_file):
        st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
        with st.spinner("Analyse en cours... Veuillez patienter"):
            model = load_cached_model(model_path)
            input_shape = model.input_shape[1:3]
            try:
                image_array = preprocess_image(uploaded_file, target_size=input_shape)
                predicted_class, confidence = predict_image(model, image_array)
            except Exception as e:
                st.error(f"⚠️ Erreur lors de l'analyse de l'image : {e}")
                st.stop()

        # Afficher les résultats avec style dynamique
        if confidence >= 80:
            result_style = "result-success"
        elif confidence >= 50:
            result_style = "result-warning"
        else:
            result_style = "result-error"

        st.markdown(
            f"""
            <div class="{result_style}">
                <h2>Résultat de l'Analyse</h2>
                <p>✅ Classe prédite : <strong>{predicted_class}</strong></p>
                <p>📊 Confiance : <strong>{confidence:.2f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Le fichier téléchargé n'est pas une image valide.")
else:
    st.warning("Veuillez télécharger une image pour commencer.")

# Footer
st.markdown(
    """
    <footer>
        &copy; 2024 Reconnaissance des Maladies des Plantes | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI<br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width: 20px; vertical-align: middle;"> GitHub
        </a> |
        <a href="https://streamlit.io" target="_blank">
            <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit" style="width: 20px; vertical-align: middle;"> Streamlit
        </a>
    </footer>
    """,
    unsafe_allow_html=True
)
