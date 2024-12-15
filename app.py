import os
import streamlit as st
import base64
from models.main import load_model, preprocess_image, predict_image, class_names

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
            forced-color-adjust: none;
        }}

        /* Sidebar style */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #006400, #2e8b57);
            color: white;
            border-right: 3px solid #004d00;
            padding: 20px;
        }}

        [data-testid="stSidebar"] h1 {{
            color: white;
            font-weight: bold;
            font-size: 22px; /* Increased size */
        }}

        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {{
            color: white;
            font-weight: bold;
            font-size: 18px;
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
            width: 250px; /* Increased size */
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

        /* Enlever les puces de la liste */
        ul {{
            list-style-type: none; /* Enlève les points */
            padding-left: 0; /* Ajuste le retrait à gauche */
        }}

        /* Result styles */
        .result-success {{
            background-color: #d4edda;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .result-warning {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .result-error {{
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}

        /* Backdrop filter for content block */
        .content-block {{
            backdrop-filter: blur(5px);
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}

        .stWarning {{
            margin-top: 20px;
            background-color: #D2B48C; /* Marron clair */
            color: black;
            padding: 10px;
            border-radius: 10px;
            font-weight: bold;
            text-align: center;
        }}

        /* Footer styles */
        footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #F5F5DC; /* Blanc cassé */
            background-color: #8B4513; /* Marron */
            padding: 10px;
            border-radius: 10px;
        }}
        footer a {{
            text-decoration: none;
            color: #F5F5DC; /* Blanc cassé pour les liens */
        }}
        footer a:hover {{
            color: #FFD700; /* Golden yellow for hover effect */
            text-decoration: underline;
        }}

        /* Responsive design */
        @media (max-width: 768px) {{
            .stApp {{
                font-size: 14px;
            }}
        }}

        /* Animation for the title */
        .title {{
            animation: fadeIn 2s ease-in-out;
            color: #004d00; /* Dark green color */
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        </style>
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

# Apply custom styles
set_custom_style()

# Sidebar
st.sidebar.title("Reco-Plantes")

model_choice = st.sidebar.selectbox(
    "Choisissez un modèle :",
    ["ResNet50 🖼️", "MobileNetV2 ⚡"]
)

# Models
models = {
    "ResNet50": "models/resnet50_model.keras",
    "MobileNetV2": "models/mobilenetv2_model.keras",
}

# Normalize the model choice to match the dictionary keys
normalized_model_choice = model_choice.split()[0]  # Extract "ResNet50" or "MobileNetV2"
model_path = models[normalized_model_choice]

model_descriptions = {
    "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
    "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
}

# Use Markdown for description to fix error
st.sidebar.markdown(
    f"""
    <div style="color:black; font-size:16px;">
        ℹ️ {model_descriptions[normalized_model_choice]}
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Main title
st.markdown('<h1 class="title">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

# Instructions
st.markdown(
    """
    <div class="content-block" style="background-color: #2e8b57; color: white;">
        <h2 class="subtitle">Bienvenue dans l'application !</h2>
        <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
        <p><strong>Comment utiliser :</strong></p>
        <ul>
            <li>1 Téléchargez une image via la barre latérale.</li>
            <li>2 Sélectionnez un modèle dans le menu latéral.</li>
            <li>3 Le résultat s'affichera automatiquement après analyse.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Warning message
st.markdown(
    """
    <div class="stWarning">
        ⚠️ Veuillez télécharger une image valide.
    </div>
    """,
    unsafe_allow_html=True
)

# Image animation
image_path = "assets/images/imagecss.png"
if os.path.exists(image_path):
    image_base64 = get_image_base64(image_path)
    if image_base64:
        st.markdown(
            f"""
            <div>
                <img src="{image_base64}" alt="Plant Animation" class="animated-image">
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    """
    <footer>
        &copy; 2024 Reconnaissance des Maladies des Plantes | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI
        <br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">
            GitHub
        </a> |
        <a href="https://streamlit.io" target="_blank">
            Streamlit
        </a>
    </footer>
    """,
    unsafe_allow_html=True
)
