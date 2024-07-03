# Plant Disease Classifier

Ce projet utilise TensorFlow et Google Cloud Storage pour classifier les plantes et détecter les maladies. Il identifie différents types de plantes et leurs états de santé à partir d'images, puis fournit des informations pertinentes en utilisant des modèles pré-entraînés et une base de données MongoDB.

## Fonctionnalités

- **Téléchargement de modèles** : Télécharge des modèles depuis Google Cloud Storage.
- **Prédiction** : Utilise TensorFlow pour classer les images de plantes en bonne santé ou malades.
- **Reconnaissance d'image** : Intègre l'API Google Cloud Vision pour la détection d'objets dans les images.
- **Base de données** : Récupère des informations détaillées sur les plantes et les pathogènes depuis MongoDB.
- **API REST** : Expose une API pour la prédiction de l'état de santé des plantes.

## Prérequis

- Python 3.7 ou supérieur
- TensorFlow
- Google Cloud SDK
- MongoDB
