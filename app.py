import os 
import streamlit as st
import base64
from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter  # Utiliser l'interpréteur TFLite

# Fonction pour appliquer les styles personnalisés
def set_custom_style():
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

        /* Animation for the title */
        .title {{
            animation: fadeIn 2s ease-in-out;
            color: #004d00;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
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
        footer img {{
            width: 20px;
            vertical-align: middle;
            margin-right: 5px;
        }}

        /* Nouvelle classe pour la section Résultat de l'Analyse */
        .result-block {{
            background-color: #2e8b57; /* Même couleur que "Bienvenue dans l'application !" */
            color: white; /* Texte en blanc pour la lisibilité */
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }}

        /* Styles supplémentaires selon la confiance */
        .result-success {{
            border: 2px solid #28a745;
        }}
        .result-warning {{
            border: 2px solid #ffc107;
        }}
        .result-error {{
            border: 2px solid #dc3545;
        }}

        /* Styles pour le bloc de bienvenue */
        .content-block {{
            background-color: #2e8b57; /* Couleur de fond */
            color: white; /* Couleur du texte */
            padding: 15px;
            border-radius: 10px;
        }}

        .subtitle {{
            font-size: 24px;
            margin-bottom: 10px;
        }}

        .stWarning {{
            background-color: #ffcc00;
            color: black;
            padding: 10px;
            border-radius: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Fonction pour charger les modèles TFLite avec cache
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        st.write(f"Modèle chargé : {model_path}")  # Pour débogage
        return interpreter
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle TFLite : {e}")
        return None

# Fonction pour prétraiter l'image
def preprocess_image(uploaded_file, target_size):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array
    except Exception as e:
        st.error(f"Erreur lors du prétraitement de l'image : {e}")
        return None

# Fonction pour nettoyer les noms de classes
def clean_class_name(class_name):
    class_name = class_name.replace('_', ' ').replace('  ', ' ').strip()
    class_name = class_name.replace('(including sour)', 'including sour')
    class_name = class_name.replace('(Citrus greening)', 'Citrus greening')
    class_name = class_name.replace('(maize)', 'maize')
    # Ajouter d'autres remplacements si nécessaire
    return class_name

# Fonction pour obtenir les détails de la maladie
def get_disease_details(disease_name):
    disease_details = {
        'Apple Apple scab': {
            'symptoms': 'Taches brunes sur les feuilles, parfois en forme de cercle.',
            'impact': 'Réduction de la photosynthèse et du rendement des fruits.',
            'treatment': 'Appliquez un fongicide à base de cuivre.',
            'prevention': 'Éliminez les feuilles infectées et améliorez la circulation de l\'air.'
        },
        # ... (Les autres classes restent inchangées)
        'Tomato healthy': {
            'symptoms': 'Aucune maladie détectée.',
            'impact': 'Plante en bonne santé.',
            'treatment': 'Aucune action nécessaire.',
            'prevention': 'Maintenez des conditions de culture optimales.'
        }
        # Ajoutez les 10 autres classes restantes ici de manière similaire
    }
    return disease_details.get(disease_name, None)

# Fonction pour prédire la maladie et obtenir les détails
def predict_and_get_details(interpreter, image_array, class_names):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Préparation des données
        interpreter.set_tensor(input_details[0]['index'], image_array)
        
        # Faire la prédiction
        interpreter.invoke()
        
        # Récupérer les résultats
        output_data = interpreter.get_tensor(output_details[0]['index'])
        proba = output_data[0]
        predicted_class_idx = np.argmax(proba)
        predicted_proba = round(100 * proba[predicted_class_idx], 2)
        predicted_class_name = class_names[predicted_class_idx]
        return predicted_class_name, predicted_proba
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, 0

# Liste des noms de classes (38 classes)
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Appliquer les styles personnalisés
set_custom_style()

# Chemins locaux pour les modèles TFLite
model_local_path_resnet = "models/phil_resnet_best_20241202_v7_epoch25.tflite"
model_local_path_mobilenet = "models/Anas_Essai_1_MOB_L2.tflite"
model_local_path_cnn = "models/phil_cnn_2_best_20241122_v1_epoch61.tflite"

# Créer le dossier 'models' s'il n'existe pas
os.makedirs(os.path.dirname(model_local_path_resnet), exist_ok=True)

# Charger les modèles TFLite avec mise en cache
interpreter_resnet = load_tflite_model(model_local_path_resnet)
interpreter_mobilenet = load_tflite_model(model_local_path_mobilenet)
interpreter_cnn = load_tflite_model(model_local_path_cnn)

# Vérification de la charge des modèles
# st.write("Interpréteurs chargés :")
# st.write(f"ResNet50 : {interpreter_resnet is not None}")
# st.write(f"MobileNetV2 : {interpreter_mobilenet is not None}")
# st.write(f"CNN : {interpreter_cnn is not None}")

# Sidebar
st.sidebar.title("Reco-Plantes")

# Dictionnaire des interpréteurs des modèles
model_interpreters = {
    "ResNet50": interpreter_resnet,
    "MobileNetV2": interpreter_mobilenet,
    "CNN": interpreter_cnn,
}

# Description du modèle dans la sidebar
model_descriptions = {
    "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
    "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
    "CNN": "Modèle CNN personnalisé pour une détection rapide des maladies.",
}

# Sélection du modèle
selected_model = st.sidebar.selectbox("Choisissez un modèle :", list(model_interpreters.keys()))

# Vérifier que le modèle sélectionné a été chargé correctement
interpreter = model_interpreters.get(selected_model)

if interpreter is None:
    st.error(f"Le modèle {selected_model} n'a pas pu être chargé correctement.")
else:
    st.sidebar.markdown(
        f"""
        <div style="color:black; font-size:16px;">
            ℹ️ {model_descriptions.get(selected_model, 'Modèle non décrit.')}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Upload de l'image
    uploaded_file = st.sidebar.file_uploader("Téléchargez une image", type=["jpg", "png"])

    # Titre principal avec animation
    st.markdown('<h1 class="title">Reconnaissance de Maladies des Plantes</h1>', unsafe_allow_html=True)

    # Instructions avec liste personnalisée
    st.markdown(
        """
        <div class="content-block">
            <h2 class="subtitle">Bienvenue dans l'application !</h2>
            <p>Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes à partir d'images.</p>
            <p><strong>Comment utiliser :</strong></p>
            <ul>
                <li>Téléchargez une image via la barre latérale.</li>
                <li>Sélectionnez un modèle dans le menu latéral.</li>
                <li>Le résultat s'affichera automatiquement après analyse.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Analyse et résultats
    if uploaded_file and interpreter:
        st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
        with st.spinner("Analyse en cours... Veuillez patienter"):
            # Obtenir la taille d'entrée du modèle
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape'][1:3]
            image_array = preprocess_image(uploaded_file, target_size=input_shape)
            if image_array is not None:
                predicted_class, confidence = predict_and_get_details(interpreter, image_array, class_names)
            else:
                predicted_class, confidence = None, 0

        if predicted_class:
            # Nettoyer la prédiction
            predicted_class_clean = clean_class_name(predicted_class)

            # Récupérer les détails de la maladie
            disease_details = get_disease_details(predicted_class_clean)

            # Déterminer le style en fonction de la confiance
            if confidence >= 80:
                result_style = "result-success"
            elif confidence >= 50:
                result_style = "result-warning"
            else:
                result_style = "result-error"

            recommendations = ""

            if 'healthy' in predicted_class_clean.lower():
                diagnosis = "Feuille en bonne santé."
                recommendations = "<strong>Aucune action nécessaire.</strong>"
            else:
                diagnosis = f"Maladie détectée - {predicted_class_clean}."
                if disease_details:
                    recommendations = f"""
                    <strong>Symptômes :</strong> {disease_details['symptoms']}<br>
                    <strong>Impact :</strong> {disease_details['impact']}<br>
                    <strong>Traitement :</strong> {disease_details['treatment']}<br>
                    <strong>Prévention :</strong> {disease_details['prevention']}
                    """
                else:
                    recommendations = "Aucune recommandation disponible. Veuillez consulter un expert agricole."

            # Message en cas d'incertitude
            if confidence < 50:
                st.warning("⚠️ La confiance dans la prédiction est faible. Essayez une photo plus claire ou consultez un expert.")

            # Affichage des résultats
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


