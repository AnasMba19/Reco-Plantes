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

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label {{
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
            animation: scaleAnimation 2s infinite alternate;
            display: block;
            margin: 0 auto;
        }}

        @keyframes scaleAnimation {{
            from {{ transform: scale(1); }}
            to {{ transform: scale(1.1); }}
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

        /* Footer styles */
        footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #555;
        }}
        footer a {{
            text-decoration: none;
            color: #555;
        }}
        footer a:hover {{
            color: #2e8b57;
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
model_path = models[model_choice]

model_descriptions = {
    "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
    "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
}
st.sidebar.info(f"ℹ️ **Modèle choisi :** {model_descriptions[model_choice]}")

uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Main title
st.markdown('<h1 class="title">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

# Instructions
st.markdown(
    """
    <div class="content-block">
        <h2 class="subtitle">Bienvenue dans l'application !</h2>
        <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
        <p><strong>Comment utiliser :</strong></p>
        <ul>
            <li><span class='icon-number'>1</span> Téléchargez une image via la barre latérale.</li>
            <li><span class='icon-number'>2</span> Sélectionnez un modèle dans le menu latéral.</li>
            <li><span class='icon-number'>3</span> Le résultat s'affichera automatiquement après analyse.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Analysis and results
if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    with st.spinner("Analyse en cours... Veuillez patienter"):
        model = load_model(model_path)
        input_shape = model.input_shape[1:3]
        image_array = preprocess_image(uploaded_file, target_size=input_shape)
        predicted_class, confidence = predict_image(model, image_array)

    result_style = "result-success" if confidence >= 80 else "result-warning" if confidence >= 50 else "result-error"
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

# Animated image
image_path = "assets/images/imagecss.png"

if not os.path.exists(image_path):
    st.error(f"⚠️ L'image '{image_path}' est introuvable. Vérifiez le chemin ou le dossier.")
else:
    image_base64 = get_image_base64(image_path)
    if image_base64:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 20px;">
                <img src="{image_base64}" alt="Plant Animation" class="animated-image" style="width: 200px;">
            </div>
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    """
    <footer>
        © 2024 Reconnaissance des Maladies des Plantes | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI
        <br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width:20px; vertical-align:middle; margin-right:5px;">GitHub
        </a> |
        <a href="https://streamlit.io" target="_blank">
            <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit" style="width:20px; vertical-align:middle; margin-right:5px;">Streamlit
        </a>
    </footer>
    """,
    unsafe_allow_html=True
)
