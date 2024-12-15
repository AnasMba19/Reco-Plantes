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
            font-size: 22px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }}

        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {{
            color: #f0f0f0;
            font-weight: bold;
            font-size: 18px;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        }}

        /* Hover effects for buttons */
        button:hover {{
            transform: scale(1.1);
            transition: transform 0.3s ease-in-out;
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

        /* Custom list icons */
        .custom-list li {{
            list-style: none;
            margin: 10px 0;
            display: flex;
            align-items: center;
            transition: all 0.3s ease-in-out;
        }}
        .custom-list li:hover {{
            transform: translateX(10px);
            color: #004d00;
        }}
        .custom-list li::before {{
            content: '\2713';
            color: #2e8b57;
            font-weight: bold;
            font-size: 20px;
            margin-right: 10px;
        }}
        .custom-list li span.number {{
            color: #ffffff;
            background: #006400;
            border-radius: 50%;
            padding: 5px 10px;
            margin-right: 10px;
            display: inline-block;
            width: 25px;
            text-align: center;
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

        /* Footer styles */
        footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 16px;
            color: #F5F5DC;
            background-color: #8B4513;
            padding: 20px 10px;
            border-radius: 10px;
            letter-spacing: 0.5px;
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

        /* Espacement des blocs */
        .content-block, .result-block {{
            margin-bottom: 30px;
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
normalized_model_choice = model_choice.split()[0]
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

# Titre principal avec animation
st.markdown('<h1 class="title">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

# Instructions avec liste personnalisée
st.markdown(
    """
    <div class="content-block" style="background-color: #2e8b57; color: white;">
        <h2 class="subtitle">Bienvenue dans l'application !</h2>
        <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
        <p><strong>Comment utiliser :</strong></p>
        <ul class="custom-list">
            <li>
                <span class="number">1</span>
                Téléchargez une image via la barre latérale.
            </li>
            <li>
                <span class="number">2</span>
                Sélectionnez un modèle dans le menu latéral.
            </li>
            <li>
                <span class="number">3</span>
                Le résultat s'affichera automatiquement après analyse.
            </li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Analyse et résultats
if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    with st.spinner("Analyse en cours... Veuillez patienter"):
        model = load_model(model_path)
        input_shape = model.input_shape[1:3]
        image_array = preprocess_image(uploaded_file, target_size=input_shape)
        predicted_class, confidence = predict_image(model, image_array)

    if confidence >= 80:
        result_style = "result-success"
    elif confidence >= 50:
        result_style = "result-warning"
    else:
        result_style = "result-error"

    st.markdown(
        f"""
        <div class="result-block {result_style}">
            <h2 class="subtitle">Résultat de l'Analyse</h2>
            <p>✅ Classe prédite : <strong>{predicted_class}</strong></p>
            <p>📊 Confiance : <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="stWarning">
            ⚠️ Veuillez télécharger une image valide.
        </div>
        """,
        unsafe_allow_html=True
    )

# Image animée
image_path = "assets/images/imagecss.png"

if not os.path.exists(image_path):
    st.error(f"⚠️ L'image '{image_path}' est introuvable. Vérifiez le chemin ou le dossier.")
else:
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

# Footer avec icônes
st.markdown(
    """
    <footer>
        &copy; 2024 Reconnaissance des Maladies des Plantes | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI<br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub"> GitHub
        </a> |
        <a href="https://streamlit.io" target="_blank">
            <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit"> Streamlit
        </a>
    </footer>
    """,
    unsafe_allow_html=True
)
