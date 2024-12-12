import streamlit as st
from models.main import load_model, preprocess_image, predict_image, class_names

# Titre principal
st.title("🌿 Reconnaissance de Maladies des Plantes")

# Description de l'application
st.markdown(
    """
    Bienvenue ! Cette application utilise des modèles d'apprentissage profond pour détecter les maladies des plantes.

    **Instructions :**
    1. Téléchargez une image en cliquant sur le bouton "Browse Files".
    2. Choisissez un modèle dans le menu déroulant.
    3. Obtenez des résultats instantanément ! 🌟
    """
)

# Séparateur visuel
st.divider()

# Sélection du modèle
model_choice = st.selectbox("Choisissez un modèle :", ["Model 1", "Model 2", "Model 3", "Model 4"])

# Mapping des modèles
models = {
    "Model 1": "models/Anas_Essai_1_MOB_L2.keras",
    "Model 2": "models/Anas_Essai_1_MOB_Repeat.keras",
    "Model 3": "models/leila_best_model_cnn_TEM3.keras",
    "Model 4": "models/model_cnn_4_best.keras",
}
model_path = models[model_choice]

# Chargement du modèle
model = load_model(model_path)

# Téléchargement de l'image
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
    input_shape = model.input_shape[1:3]
    image_array = preprocess_image(uploaded_file, target_size=input_shape)
    predicted_class, confidence = predict_image(model, image_array)
    st.success(f"Résultat : {predicted_class} ({confidence:.2f}%)")
