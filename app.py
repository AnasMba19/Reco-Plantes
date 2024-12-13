import streamlit as st
from models.main import load_model, preprocess_image, predict_image, class_names

try:
    from streamlit_lottie import st_lottie
    import requests
    # Animation lottie thématique (feuilles qui poussent)
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_pkscqlmk.json"
    lottie_json = requests.get(lottie_url).json()
except ImportError:
    lottie_json = None

def set_custom_background_and_style():
    # Image de fond cohérente avec le thème "plantes"
    background_url = "https://images.unsplash.com/photo-1516651022221-3c83666ae09c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        .stApp {{
            background: url("{background_url}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Roboto', sans-serif;
        }}

        /* Couche semi-transparente pour améliorer la lisibilité du texte */
        /* On va encapsuler les blocs dans des containers personnalisés */
        .content-block {{
            background-color: rgba(0, 100, 0, 0.8);
            color: #FFFFFF;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }}

        h1, h2, h3, p {{
            margin: 0 0 10px 0;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(0, 100, 0, 0.8);
            border: none;
        }}

        .stTabs [data-baseweb="tab"].stTabs-tab--selected {{
            background: rgba(0, 100, 0, 1);
            color: #FFFFFF;
        }}

        .stTabs .stTabs-header {
            border-bottom: none;
        }

        .stTabs .stTabs-nav {
            background: transparent;
        }

        .stProgress > div > div > div {{
            background-color: #4CAF50;
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

        footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #FFFFFF;
        }}

        .result-card {{
            background-color: rgba(0,100,0,0.8);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }}

        .instructions-block {{
            line-height: 1.6;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_background_and_style()

# Barre latérale pour les paramètres
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
st.markdown('<div class="content-block"><h1>🌿 Reconnaissance de Maladies des Plantes</h1></div>', unsafe_allow_html=True)

# Onglets
tab_instructions, tab_analyse = st.tabs(["📖 Instructions", "🔬 Analyse"])

with tab_instructions:
    st.markdown(
        '<div class="content-block instructions-block">'
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
    st.info(f"ℹ️ **Description du modèle choisi :** {model_descriptions[model_choice]}")

    if lottie_json:
        st_lottie(lottie_json, height=200)

with tab_analyse:
    if uploaded_file is not None:
        with st.spinner("Analyse en cours..."):
            model = load_model(model_path)
            st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
            input_shape = model.input_shape[1:3]
            image_array = preprocess_image(uploaded_file, target_size=input_shape)
            predicted_class, confidence = predict_image(model, image_array)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"✅ **Résultat :** {predicted_class}")
        st.markdown("**Confiance :**")
        st.progress(confidence / 100.0)  
        st.write(f"{confidence:.2f}%")

        st.markdown("[En savoir plus sur cette maladie](https://www.google.com)", unsafe_allow_html=True)
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
