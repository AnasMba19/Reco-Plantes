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
            display: flex;
            align-items: center;
        }}
        .custom-list li::before {{
            content: '\\2713'; /* Checkmark icon */
            color: #2e8b57; /* Dark green */
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

        /* Backdrop filter for content block */
        .content-block {{
            backdrop-filter: blur(5px);
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}

        .stWarning {{
            margin-top: 20px; /* Add margin to separate from above */
            background-color: #D2B48C; /* Marron clair */
            color: black; /* Black text color */
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
            margin: 0 5px;
        }}
        footer a:hover {{
            color: #FFD700; /* Golden yellow for hover effect */
            text-decoration: underline;
        }}
        footer img {{
            width: 20px;
            vertical-align: middle;
            margin-right: 5px;
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

# Fonction pour obtenir les détails de la maladie
def get_disease_details(disease_name):
    disease_details = {
        "Black Spot": {
            "symptoms": "Petites taches noires circulaires, feuilles jaunies.",
            "impact": "Réduction du rendement si non traité.",
            "treatment": "Appliquez un fongicide naturel à base de cuivre.",
            "prevention": "Évitez l'excès d'humidité et nettoyez vos outils de taille.",
            "reference_image": "assets/images/black_spot_reference.png"  # Chemin vers l'image de référence
        },
        "Powdery Mildew": {
            "symptoms": "Poudre blanche sur les feuilles, croissance ralentie.",
            "impact": "Affaiblissement de la plante et réduction de la photosynthèse.",
            "treatment": "Utilisez des fongicides spécifiques ou des solutions à base de bicarbonate de soude.",
            "prevention": "Assurez une bonne circulation de l'air et évitez l'arrosage par le dessus.",
            "reference_image": "assets/images/powdery_mildew_reference.png"
        },
        # Ajoutez d'autres maladies ici
    }
    return disease_details.get(disease_name, None)

# Appliquer les styles personnalisés
set_custom_style()

# Sidebar
st.sidebar.title("Reco-Plantes")

model_choice = st.sidebar.selectbox(
    "Choisissez un modèle :",
    ["ResNet50 🍃", "MobileNetV2 🍃", "CNN 🍃"]  # Ajout de "CNN 🍃"
)

# Modèles avec l'extension .keras
models = {
    "ResNet50": "models/phil_resnet_best_20241202_v7_epoch25.keras",  # Ajout du .keras
    "MobileNetV2": "models/Anas_Essai_1_MOB_L2.keras",  # Si c'est .keras
    "CNN": "models/phil_cnn_2_best_20241122_v1_epoch61.keras",  # Si c'est .keras
}

# Normaliser le choix du modèle pour correspondre aux clés du dictionnaire
normalized_model_choice = model_choice.split()[0]  # Extrait "ResNet50", "MobileNetV2" ou "CNN"
model_path = models[normalized_model_choice]

# Vérification de l'existence du modèle avant de le charger
if not os.path.exists(model_path):
    st.error(f"Le modèle n'a pas été trouvé à {model_path}")
    model = None
else:
    try:
        model = load_model(model_path)
        print(f"Modèle {model_path} chargé avec succès")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        model = None

# Description du modèle dans la sidebar
model_descriptions = {
    "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
    "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
    "CNN": "Modèle CNN personnalisé pour une détection rapide des maladies.",  # Description pour CNN
}

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
if uploaded_file and model:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    with st.spinner("Analyse en cours... Veuillez patienter"):
        input_shape = model.input_shape[1:3]
        try:
            image_array = preprocess_image(uploaded_file, target_size=input_shape)
            predicted_class, confidence = predict_image(model, image_array)
        except Exception as e:
            st.error(f"Erreur lors du prétraitement ou de la prédiction de l'image : {e}")
            predicted_class, confidence = None, 0

    if predicted_class:
        if confidence >= 80:
            result_style = "result-success"
        elif confidence >= 50:
            result_style = "result-warning"
        else:
            result_style = "result-error"

        # Vérifier si la prédiction est "Sain" ou une maladie
        if predicted_class.lower() == "sain" or predicted_class.lower() == "healthy":
            diagnosis = "Feuille en bonne santé."
            recommendations = "<strong>Aucune action nécessaire.</strong>"
            disease_details = None
            reference_image_html = ""
        else:
            diagnosis = f"Maladie détectée - {predicted_class}."
            disease_details = get_disease_details(predicted_class)
            if disease_details:
                symptoms = disease_details["symptoms"]
                impact = disease_details["impact"]
                treatment = disease_details["treatment"]
                prevention = disease_details["prevention"]
                reference_image_path = disease_details["reference_image"]

                # Charger l'image de référence
                if os.path.exists(reference_image_path):
                    ref_image_base64 = get_image_base64(reference_image_path)
                    reference_image_html = f'<img src="{ref_image_base64}" alt="Photo de référence" style="width:300px;">'
                else:
                    reference_image_html = "<p>⚠️ Image de référence non disponible.</p>"

                recommendations = f"""
                <strong>Symptômes :</strong> {symptoms}<br>
                <strong>Impact :</strong> {impact}<br>
                <strong>Traitement :</strong> {treatment}<br>
                <strong>Prévention :</strong> {prevention}
                """
            else:
                recommendations = "<strong>Aucune recommandation disponible.</strong>"
                reference_image_html = ""

        # Message en cas d'incertitude
        if confidence < 50:
            diagnosis = "Nous ne sommes pas sûrs du diagnostic."
            recommendations = "Essayez de prendre une photo plus claire ou consultez un expert agricole."
            reference_image_html = ""

        st.markdown(
            f"""
            <div class="result-block {result_style}">
                <h2 class="subtitle">Résultat de l'Analyse</h2>
                <p><strong>Résultat :</strong> {diagnosis}</p>
                <p><strong>Confiance :</strong> {confidence:.2f}%</p>
                <hr>
                <div>
                    {recommendations}
                </div>
                <div style="margin-top: 10px;">
                    {reference_image_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Impossible de déterminer le résultat de l'analyse.")
else:
    if not uploaded_file:
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
