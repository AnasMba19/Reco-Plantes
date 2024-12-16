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

        /* Fond marron pour le bloc regroupé */
        .sidebar-block {{
            background-color: #8B4513; /* Marron */
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
            color: #FFFFFF;
        }}
        .sidebar-block label, .sidebar-block p {{
            color: #FFFFFF; /* Texte blanc */
            margin-bottom: 10px;
        }}
        .sidebar-block select, .sidebar-block input {{
            width: 100%;
            padding: 5px;
            border-radius: 5px;
            margin-top: 5px;
            border: 1px solid #F5F5DC; /* Blanc cassé */
        }}

        /* Hover effects for buttons */
        button:hover {{
            transform: scale(1.05);
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }}

        /* Footer styles */
        footer {{
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #F5F5DC;
            background-color: #8B4513; /* Marron */
            padding: 10px;
            border-radius: 10px;
        }}
        footer a {{
            text-decoration: none;
            color: #F5F5DC;
        }}
        footer a:hover {{
            color: #FFD700;
            text-decoration: underline;
        }}
        footer img {{
            width: 20px;
            vertical-align: middle;
            margin-right: 5px;
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

# Bloc regroupé avec fond marron
st.sidebar.markdown(
    """
    <div class="sidebar-block">
        <label for="model-select">Choisissez un modèle :</label>
        <select id="model-select">
            <option>ResNet50 🖼️</option>
            <option>MobileNetV2 ⚡</option>
        </select>
        <p style="margin-top: 15px;">📥 <strong>Téléchargez une image</strong></p>
        <p>Drag and drop file here<br>Limit 200MB per file • JPG, PNG, JPEG</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Téléchargement de fichier
uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

# Titre principal
st.markdown('<h1 class="title">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

# Instructions
st.markdown(
    """
    <div class="content-block" style="background-color: #2e8b57; color: white;">
        <h2 class="subtitle">Bienvenue dans l'application !</h2>
        <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
        <p><strong>Comment utiliser :</strong></p>
        <ul>
            <li>1. Téléchargez une image via la barre latérale.</li>
            <li>2. Sélectionnez un modèle dans le menu latéral.</li>
            <li>3. Le résultat s'affichera automatiquement après analyse.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# Analyse et résultats
if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    with st.spinner("Analyse en cours... Veuillez patienter"):
        model = load_model(models[model_choice.split()[0]])
        input_shape = model.input_shape[1:3]
        image_array = preprocess_image(uploaded_file, target_size=input_shape)
        predicted_class, confidence = predict_image(model, image_array)

    # Affichage des résultats
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

# Footer
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
