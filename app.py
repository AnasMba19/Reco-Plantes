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

        /* Custom list icons */
        .custom-list li {{
            list-style: none;
            margin: 10px 0;
        }}
        .custom-list li::before {{
            content: '\2713'; /* Checkmark icon */
            color: #2e8b57; /* Dark green */
            font-weight: bold;
            font-size: 20px;
            margin-right: 10px;
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
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply custom styles
set_custom_style()

# Main section with updated numbers in list
st.markdown(
    """
    <div class="content-block" style="background-color: #2e8b57; color: white;">
        <h2 class="subtitle">Bienvenue dans l'application !</h2>
        <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
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
