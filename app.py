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

        /* Hover effects for buttons */
        button:hover {{
            transform: scale(1.05);
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }}

        /* Animated image */
        .animated-image {{
            animation: rotateZoom 3s infinite ease-in-out;
            display: block;
            margin: 0 auto;
            width: 250px;
        }}

        @keyframes rotateZoom {{
            0% {{
                transform: scale(1) rotate(0deg);
            }}
            50% {{
                transform: scale(1.1) rotate(20deg);
            }}
            100% {{
                transform: scale(1) rotate(0deg);
            }}
        }}

        /* Instructions custom list */
        .custom-list li {{
            list-style: none;
            margin: 10px 0;
            display: flex;
            align-items: center;
        }}
        .custom-list li::before {{
            content: '\2713';
            color: #2e8b57;
            font-weight: bold;
            font-size: 20px;
            margin-right: 10px;
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

# Instructions interactives
def display_instructions():
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.8); padding: 20px; border-radius: 10px;">
            <h2>Bienvenue dans l'application !</h2>
            <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes.</p>
            <p><strong>Comment utiliser :</strong></p>
            <ul class="custom-list">
                <li>1. Téléchargez une image via la barre latérale.</li>
                <li>2. Sélectionnez un modèle dans le menu latéral.</li>
                <li>3. Le résultat s'affichera automatiquement après analyse.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

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

# Normaliser le choix du modèle pour correspondre aux clés du dictionnaire
models = {
    "ResNet50": "models/resnet50_model.keras",
    "MobileNetV2": "models/mobilenetv2_model.keras",
}
normalized_model_choice = model_choice.split()[0]
model_path = models[normalized_model_choice]

# Instructions affichées en haut
display_instructions()

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Image animée
image_path = "assets/images/imagecss.png"
if os.path.exists(image_path):
    image_base64 = get_image_base64(image_path)
    if image_base64:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="{image_base64}" alt="Plant Animation" class="animated-image">
            </div>
            """,
            unsafe_allow_html=True
        )

if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
else:
    st.warning("Veuillez télécharger une image pour commencer.")

# Footer
st.markdown(
    """
    <footer>
        &copy; 2024 Reconnaissance des Maladies des Plantes | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI<br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">GitHub</a> |
        <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </footer>
    """,
    unsafe_allow_html=True
)
