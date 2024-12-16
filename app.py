import os
import streamlit as st
import gdown  # Assurez-vous que gdown est installé via requirements.txt
import base64
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

# Fonction pour télécharger le modèle depuis Google Drive
@st.cache_resource
def download_model_from_drive(drive_link, output_path):
    if not os.path.exists(output_path):
        st.info("Téléchargement du modèle, veuillez patienter...")
        try:
            gdown.download(drive_link, output_path, quiet=False)
            st.success("Modèle téléchargé avec succès.")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement du modèle : {e}")
            return False
    else:
        st.info("Modèle déjà téléchargé.")
    return True

# Fonction pour charger les modèles avec cache
@st.cache_resource
def load_classifier(model_path):
    try:
        classifier = load_model(model_path)
        return classifier
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

# Fonction pour prétraiter l'image
def preprocess_image(uploaded_file, target_size):
    try:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
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
        'Apple Black rot': {
            'symptoms': 'Taches noires et pourriture sur les feuilles et les fruits.',
            'impact': 'Provoque la chute prématurée des fruits et réduit le rendement.',
            'treatment': 'Appliquez un fongicide à base de cuivre ou de soufre.',
            'prevention': 'Éliminez les parties infectées et améliorez la circulation de l\'air.'
        },
        'Apple Cedar apple rust': {
            'symptoms': 'Développement de pustules orange sur les feuilles.',
            'impact': 'Affaiblissement de la plante et réduction du rendement.',
            'treatment': 'Utilisez des fongicides appropriés et éliminez les plants hôtes.',
            'prevention': 'Évitez l\'humidité excessive et améliorez la circulation de l\'air.'
        },
        'Cherry Powdery mildew': {
            'symptoms': 'Poudre blanche sur les feuilles et les bourgeons.',
            'impact': 'Ralentit la croissance de la plante et réduit le rendement.',
            'treatment': 'Appliquez des fongicides spécifiques ou des solutions à base de bicarbonate de soude.',
            'prevention': 'Assurez une bonne circulation de l\'air et évitez l\'arrosage par le dessus.'
        },
        'Corn Cercospora leaf spot Gray leaf spot': {
            'symptoms': 'Taches grises sur les feuilles avec des bords bruns.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides appropriés.',
            'prevention': 'Éliminez les résidus de culture et améliorez la rotation des cultures.'
        },
        'Corn Common rust': {
            'symptoms': 'Pustules rouges sur les feuilles.',
            'impact': 'Diminution de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides spécifiques.',
            'prevention': 'Utilisez des variétés résistantes et pratiquez la rotation des cultures.'
        },
        'Corn Northern Leaf Blight': {
            'symptoms': 'Taches longues et étroites sur les feuilles.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides spécifiques.',
            'prevention': 'Utilisez des variétés résistantes et pratiquez la rotation des cultures.'
        },
        'Grape Black rot': {
            'symptoms': 'Taches noires sur les feuilles, pourriture des raisins.',
            'impact': 'Réduction de la qualité et du rendement des raisins.',
            'treatment': 'Utilisez des fongicides spécifiques et retirez les parties infectées.',
            'prevention': 'Assurez une bonne aération et évitez l\'excès d\'humidité.'
        },
        'Grape Esca Black Measles': {
            'symptoms': 'Taches noires irrégulières sur les feuilles et les fruits.',
            'impact': 'Affaiblissement de la plante et réduction du rendement.',
            'treatment': 'Appliquez des fongicides et taillez les parties infectées.',
            'prevention': 'Utilisez des variétés résistantes et améliorez la ventilation.'
        },
        'Grape Leaf blight Isariopsis Leaf Spot': {
            'symptoms': 'Taches brunes sur les feuilles avec des bords jaunes.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides appropriés.',
            'prevention': 'Éliminez les feuilles infectées et améliorez la circulation de l\'air.'
        },
        'Orange Haunglongbing Citrus greening': {
            'symptoms': 'Feuilles jaunies, fruits déformés et amers.',
            'impact': 'Décimation de la plantation et réduction drastique du rendement.',
            'treatment': 'Il n\'existe actuellement aucun traitement efficace.',
            'prevention': 'Utilisez des variétés résistantes et contrôlez les insectes vecteurs.'
        },
        'Peach Bacterial spot': {
            'symptoms': 'Taches brunes sur les feuilles, les fruits et les branches.',
            'impact': 'Réduction de la photosynthèse et des rendements.',
            'treatment': 'Appliquez des fongicides à base de cuivre.',
            'prevention': 'Éliminez les feuilles infectées et améliorez la circulation de l\'air.'
        },
        'Pepper,_bell Bacterial spot': {
            'symptoms': 'Taches brunes sur les feuilles et les fruits.',
            'impact': 'Diminution de la qualité et du rendement des poivrons.',
            'treatment': 'Utilisez des fongicides spécifiques et éliminez les plantes infectées.',
            'prevention': 'Évitez l\'arrosage par le dessus et utilisez des variétés résistantes.'
        },
        'Potato Early blight': {
            'symptoms': 'Taches brunes avec des anneaux concentriques sur les feuilles.',
            'impact': 'Diminution de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides appropriés.',
            'prevention': 'Éliminez les feuilles infectées et pratiquez la rotation des cultures.'
        },
        'Potato Late blight': {
            'symptoms': 'Taches noires et vertes sur les feuilles et les tubercules.',
            'impact': 'Décimation rapide des plantations si non contrôlée.',
            'treatment': 'Appliquez immédiatement des fongicides spécifiques.',
            'prevention': 'Éliminez les plantes infectées et assurez une bonne aération.'
        },
        'Squash Powdery mildew': {
            'symptoms': 'Poudre blanche sur les feuilles et les tiges.',
            'impact': 'Ralentit la croissance de la plante et réduit le rendement.',
            'treatment': 'Utilisez des fongicides spécifiques ou des solutions à base de bicarbonate de soude.',
            'prevention': 'Assurez une bonne circulation de l\'air et évitez l\'excès d\'humidité.'
        },
        'Strawberry Leaf scorch': {
            'symptoms': 'Feuilles brûlées avec des bords brunis.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Utilisez des fongicides appropriés et améliorez la circulation de l\'air.',
            'prevention': 'Éliminez les feuilles infectées et évitez l\'excès d\'humidité.'
        },
        'Tomato Bacterial spot': {
            'symptoms': 'Taches brunes sur les feuilles, les tiges et les fruits.',
            'impact': 'Diminution de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides à base de cuivre.',
            'prevention': 'Éliminez les feuilles infectées et améliorez la circulation de l\'air.'
        },
        'Tomato Early blight': {
            'symptoms': 'Taches brunes avec des anneaux concentriques sur les feuilles.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides appropriés.',
            'prevention': 'Éliminez les feuilles infectées et pratiquez la rotation des cultures.'
        },
        'Tomato Late blight': {
            'symptoms': 'Taches noires et vertes sur les feuilles et les fruits.',
            'impact': 'Décimation rapide des plantations si non contrôlée.',
            'treatment': 'Appliquez immédiatement des fongicides spécifiques.',
            'prevention': 'Éliminez les plantes infectées et assurez une bonne aération.'
        },
        'Tomato Leaf Mold': {
            'symptoms': 'Croûte grise sur les feuilles.',
            'impact': 'Réduction de la photosynthèse et de la vigueur de la plante.',
            'treatment': 'Utilisez des fongicides spécifiques et améliorez la circulation de l\'air.',
            'prevention': 'Éliminez les feuilles infectées et évitez l\'excès d\'humidité.'
        },
        'Tomato Septoria leaf spot': {
            'symptoms': 'Taches brunes sur les feuilles avec des bords jaunes.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides appropriés.',
            'prevention': 'Éliminez les feuilles infectées et assurez une bonne circulation de l\'air.'
        },
        'Tomato Spider mites Two-spotted spider mite': {
            'symptoms': 'Petites taches jaunes et rougeâtres sur les feuilles, présence de toiles.',
            'impact': 'Diminution de la photosynthèse et affaiblissement de la plante.',
            'treatment': 'Utilisez des acaricides spécifiques ou des solutions naturelles comme le savon insecticide.',
            'prevention': 'Maintenez une bonne hygiène de la plantation et surveillez régulièrement les plantes.'
        },
        'Tomato Target Spot': {
            'symptoms': 'Taches circulaires brunes avec un anneau clair au centre.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Appliquez des fongicides spécifiques.',
            'prevention': 'Éliminez les feuilles infectées et améliorez la circulation de l\'air.'
        },
        'Tomato Tomato Yellow Leaf Curl Virus': {
            'symptoms': 'Feuilles jaunies et recroquevillées, croissance ralentie.',
            'impact': 'Réduction sévère du rendement et de la qualité des fruits.',
            'treatment': 'Il n\'existe actuellement aucun traitement efficace.',
            'prevention': 'Contrôlez les vecteurs insectes et utilisez des variétés résistantes.'
        },
        'Tomato Tomato mosaic virus': {
            'symptoms': 'Déformation des feuilles et des fruits, mosaïque de couleurs.',
            'impact': 'Réduction de la photosynthèse et du rendement.',
            'treatment': 'Il n\'existe actuellement aucun traitement efficace.',
            'prevention': 'Utilisez des variétés résistantes et éliminez les plantes infectées.'
        },
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
def predict_and_get_details(model, image_array, class_names):
    try:
        proba = model.predict(image_array)[0]
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

# Chemins locaux pour les modèles
model_local_path_resnet = "models/phil_resnet_best_20241202_v7_epoch25.keras"
model_local_path_mobilenet = "models/Anas_Essai_1_MOB_L2.keras"
model_local_path_cnn = "models/phil_cnn_2_best_20241122_v1_epoch61.keras"

# Créer le dossier 'models' s'il n'existe pas
os.makedirs(os.path.dirname(model_local_path_resnet), exist_ok=True)

# Lien de téléchargement direct Google Drive pour ResNet50
drive_link_resnet = "https://drive.google.com/uc?id=1iawxZGSfl8aR8NJL_rqQDWosmJ2SMWBj&export=download"

# Télécharger le modèle ResNet50 si nécessaire
download_success_resnet = download_model_from_drive(drive_link_resnet, model_local_path_resnet)

# Sidebar
st.sidebar.title("Reco-Plantes")

# Dictionnaire des chemins des modèles
model_paths = {
    "ResNet50": model_local_path_resnet,
    "MobileNetV2": model_local_path_mobilenet,
    "CNN": model_local_path_cnn,
}

# Vérifier que le modèle ResNet50 a été téléchargé avant de l'utiliser
if not download_success_resnet:
    st.error("Le modèle ResNet50 n'a pas pu être téléchargé. Veuillez vérifier le lien de téléchargement.")
else:
    # Sélection du modèle
    selected_model = st.sidebar.selectbox("Choisissez un modèle :", list(model_paths.keys()))

    # Chemin du modèle sélectionné
    model_path = model_paths.get(selected_model)

    # Charger le modèle sélectionné avec mise en cache
    classifier = load_classifier(model_path)

    # Description du modèle dans la sidebar
    model_descriptions = {
        "ResNet50": "Modèle ResNet50 optimisé pour une précision élevée.",
        "MobileNetV2": "Modèle MobileNetV2, léger et rapide pour les applications mobiles.",
        "CNN": "Modèle CNN personnalisé pour une détection rapide des maladies.",
    }

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
    if uploaded_file and classifier:
        st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
        with st.spinner("Analyse en cours... Veuillez patienter"):
            input_shape = classifier.input_shape[1:3]
            image_array = preprocess_image(uploaded_file, target_size=input_shape)
            if image_array is not None:
                predicted_class, confidence = predict_and_get_details(classifier, image_array, class_names)
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
