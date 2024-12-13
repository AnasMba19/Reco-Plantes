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

def set_custom_style():
    # Image de fond jardin
    background_url = "https://images.unsplash.com/photo-1601233742689-3cad38a171a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        html, body {{
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(rgba(0,0,0,0.25), rgba(0,0,0,0.25)), url("{background_url}") no-repeat center center fixed;
            background-size: cover;
        }}

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

        .stTabs [data-baseweb="tab"] {{
            background: rgba(34, 139, 34, 0.7);
            border: none;
            color: #fff;
        }}

        .stTabs [data-baseweb="tab"].stTabs-tab--selected {{
            background: rgba(34, 139, 34, 1);
            color: #FFFFFF;
        }}

        .stTabs .stTabs-header {{
            border-bottom: none !important;
        }}

        .stTabs .stTabs-nav {{
            background: transparent;
        }}

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
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, background-color 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: #45a049;
            transform: scale(1.03);
        }}

        [data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(4px);
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
        }}

        [data-testid="stSidebar"] .stButton > button {{
            background-color: #45a049;
        }}

        .result-card {{
            background-color: rgba(34,139,34,0.9);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}

        .instructions-block {{
            line-height: 1.6;
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

        /* Bannière avec logo */
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

        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# Barre latérale
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

# Bannière avec le logo et le titre (sans la feuille emoji)
st.markdown(
    """
    <div class="banner">
        <img src="https://raw.githubusercontent.com/AnasMba19/Reco-Plantes/main/logo_recoplantes.png" alt="Logo RecoPlantes">
        <h1>Reconnaissance de Maladies des Plantes</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Onglets: Instructions / Analyser / Contact
tab_instructions, tab_analyse, tab_contact = st.tabs(["📖 Instructions", "🔬 Analyse", "📩 Contact"])

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

    # Lien vers le manuel utilisateur
    st.markdown("<p><a href='https://docs.google.com/document/d/xyz' target='_blank'>Manuel utilisateur</a></p>", unsafe_allow_html=True)

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
        st.markdown("<p><a href='https://www.google.com' target='_blank'>En savoir plus sur cette maladie</a></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Veuillez télécharger une image valide dans la barre latérale.")

with tab_contact:
    st.markdown('<div class="content-block"><h2>Contactez-nous</h2><p>Veuillez laisser vos coordonnées et votre message.</p></div>', unsafe_allow_html=True)
    with st.form("contact_form"):
        name = st.text_input("Nom")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Envoyer")
        if submitted:
            # Vous pouvez ici ajouter du code pour envoyer un email ou enregistrer en base de données
            st.success("Merci pour votre message ! Nous vous répondrons dès que possible.")

# Footer
st.markdown(
    """
    <footer>
        © Reconnaissance des Maladies des Plantes 2024 | Développé par Leila BELMIR, Philippe BEUTIN et Anas MBARKI
        <br>
        <a href="https://github.com/AnasMba19/Reco-Plantes" target="_blank">GitHub</a> | 
        <a href="https://streamlit.io" target="_blank">Streamlit</a>
    </footer>
    """,
    unsafe_allow_html=True
)
