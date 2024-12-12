# -*- coding: utf-8 -*-
"""Script_Best_models_Github.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1pC4AYVAePerySAQtxT0oUKwimIbTkVzu
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

# Debug temporaire pour afficher les fichiers dans le dossier courant
print("Répertoire courant :", os.getcwd())
print("Fichiers dans le dossier courant :", os.listdir('.'))

# Liste des modèles disponibles
models = {
    "Model 1": "Anas_Essai_1_MOB_L2.keras",
    "Model 2": "Anas_Essai_1_MOB_Repeat.keras",
    "Model 3": "leila_best_model_cnn_TEM3.keras",
    "Model 4": "model_cnn_4_best.keras"
}

# Noms des classes complets (à adapter selon vos besoins)
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

def load_model(model_path):
    print(f"Chargement du modèle depuis : {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Modèle chargé avec succès.")
    return model

def preprocess_image(image_path, target_size):
    print(f"Prétraitement de l'image : {image_path}")
    img = load_img(image_path, target_size=target_size)  # Redimensionne l'image
    img_array = img_to_array(img)  # Convertit l'image en tableau NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Ajoute une dimension batch
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Normalisation
    return img_array

def predict_image(model, image_array):
    print("Prédiction...")
    predictions = model.predict(image_array)
    print(f"Prédictions brutes : {predictions}")  # Debug
    predicted_index = np.argmax(predictions)
    print(f"Index prédit : {predicted_index}")  # Debug
    if predicted_index >= len(class_names):
        print("Erreur : L'index prédit dépasse le nombre de classes définies.")
        return "Classe inconnue", 0.0
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

def display_image(image_path, predicted_class, confidence):
    img = load_img(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prédiction : {predicted_class} ({confidence:.2f}%)")
    plt.show()

if __name__ == "__main__":
    # Étape 1 : Liste des modèles
    print("Modèles disponibles :")
    for i, model_name in enumerate(models.keys(), start=1):
        print(f"{i}. {model_name}")

    # Étape 2 : Sélection du modèle
    choice = int(input("Entrez le numéro du modèle à utiliser : "))
    selected_model_path = models[list(models.keys())[choice - 1]]

    # Étape 3 : Charger le modèle
    model = load_model(selected_model_path)

    # Récupérer dynamiquement le nombre de classes
    num_classes = model.output_shape[-1]
    print(f"Nombre de classes dans le modèle : {num_classes}")
    if len(class_names) != num_classes:
        print("Attention : Le nombre de classes dans class_names ne correspond pas au modèle.")
        class_names = [f"Classe {i}" for i in range(num_classes)]

    # Récupérer la taille d'entrée du modèle
    input_shape = model.input_shape[1:3]  # Extrait (hauteur, largeur) attendu

    # Étape 4 : Prédire une image
    base_image_path = "C:/Users/anasm/Downloads/Reco-Plantes/test_images"  # Chemin absolu du dossier d'images
    image_name = input("Entrez le nom de l'image à prédire (ex : apple_scab_02.jpg) : ").strip()
    image_path = os.path.join(base_image_path, image_name)

    # Vérifier si le fichier existe
    print(f"Chemin complet généré pour l'image : {image_path}")  # Debug
    if not os.path.exists(image_path):
        print(f"Erreur : le fichier {image_path} n'existe pas.")
    else:
        image_array = preprocess_image(image_path, target_size=input_shape)
        predicted_class, confidence = predict_image(model, image_array)

        # Étape 5 : Afficher le résultat
        display_image(image_path, predicted_class, confidence)