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
            background-color: #2E8B57; /* Couleur verte assortie */
            color: white;
            border-right: 2px solid #006400;
            padding: 20px;
        }}

        /* Titres de la barre latérale */
        [data-testid="stSidebar"] h2 {{
            color: #FFFFFF;
            font-size: 18px;
            margin-bottom: 10px;
            font-weight: bold;
            text-transform: uppercase;
        }}

        /* Boutons */
        [data-testid="stSidebar"] .stButton > button {{
            background-color: #3CB371;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 15px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: #45a049;
            transform: scale(1.05);
        }}

        /* Sélecteurs (Selectbox) */
        [data-testid="stSidebar"] .stSelectbox {{
            background-color: #FFFFFF;
            color: #2E8B57;
            border: 1px solid #006400;
            border-radius: 5px;
            padding: 5px;
            font-size: 14px;
            font-weight: bold;
        }}

        [data-testid="stSidebar"] .stSelectbox:hover {{
            background-color: #DFF2E1;
        }}

        /* Zone de téléchargement */
        [data-testid="stSidebar"] .stFileUploader {{
            background-color: rgba(255, 255, 255, 0.15);
            color: white;
            border: 1px solid #006400;
            border-radius: 5px;
            padding: 10px;
            margin-top: 20px;
        }}

        /* Espacement global */
        [data-testid="stSidebar"] > div {{
            margin-bottom: 20px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
