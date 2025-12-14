# reconnaissance_faciale.py

import cv2
import os
import numpy as np

# --- Configuration et chemins ---
# Chemin vers le fichier Haar Cascade pour la détection de visage (doit être téléchargé)
HAAR_CASCADE_PATH = 'haarcascade_frontalface_alt.xml' 


# Dossier contenant les images de référence pour la comparaison
DATABASE_FOLDER = 'C:/Users/bastien/Desktop/acab/pp/tetes' 
# IMPORTANT : Remplacez ceci par le chemin réel de votre dossier.

# --- Globales pour l'entraînement et la détection ---
face_recognizer = None
known_faces_labels = {}
known_faces_names = [] # Liste des noms dans l'ordre de l'entraînement

def charger_haarcascade(path):
    """Charge le classifieur Haar Cascade."""
    if not os.path.exists(path):
        print(f"Erreur: Fichier Haar Cascade non trouvé à l'emplacement: {path}")
        return None
    return cv2.CascadeClassifier(path)

def preparer_base_de_donnees_visages(folder_path):
    """
    Entraîne un modèle de reconnaissance (ex: LBPH) avec les images du dossier.
    Structure du dossier attendue :
    - /dossier_visages/
        - /alice/
            - img1.jpg
            - img2.jpg
        - /bob/
            - img1.jpg
    """
    global face_recognizer, known_faces_names

    print(f"Préparation de la base de données faciale dans : {folder_path}...")
    
    # 1. Initialiser le modèle de reconnaissance (utilisons l'approche LBPH)
    # L'approche LBPH est généralement robuste et fournie avec OpenCV
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    label_id = 0
    known_faces_names = []
    
    # Parcourir chaque sous-dossier (chaque sous-dossier est une personne)
    for name in os.listdir(folder_path):
        subject_dir = os.path.join(folder_path, name)
        
        if os.path.isdir(subject_dir):
            known_faces_names.append(name)
            
            for image_name in os.listdir(subject_dir):
                image_path = os.path.join(subject_dir, image_name)
                
                # Lire l'image en niveaux de gris
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Pour l'entraînement, on suppose que l'image contient déjà un visage centré.
                    # Si non, il faudrait utiliser Haar Cascade ici pour extraire la zone de visage, 
                    # mais pour simplifier, on suppose que les images de la DB sont des visages.
                    
                    # Redimensionnement standard pour une meilleure performance
                    img = cv2.resize(img, (200, 200)) 
                    
                    faces.append(img)
                    labels.append(label_id)
            
            label_id += 1

    if faces:
        # 2. Entraînement
        print("Entraînement du modèle de reconnaissance faciale...")
        face_recognizer.train(faces, np.array(labels))
        print(f"Entraînement terminé avec {len(known_faces_names)} personnes.")
        return True
    else:
        print("Aucune image de visage trouvée pour l'entraînement.")
        return False

def identifier_visage(gray_face):
    """
    Identifie le visage fourni par rapport à la base de données entraînée.
    
    Args:
        gray_face (np.ndarray): La zone de visage en niveaux de gris.
        
    Returns:
        tuple: (nom_personne, confiance) ou (None, None)
    """
    global face_recognizer, known_faces_names
    
    if face_recognizer is None or not known_faces_names:
        return "Non entraîné", 0

    # Redimensionnement de l'image du visage détecté pour correspondre à l'entraînement
    resized_face = cv2.resize(gray_face, (200, 200))
    
    try:
        # Prédiction (label_id, confidence)
        label_id, confidence = face_recognizer.predict(resized_face)
        
        # Le seuil de confiance dépend de votre modèle/besoins. Plus la confiance est faible, mieux c'est (distance).
        # On utilise un seuil inverse (ex: < 80) pour considérer le match comme valide.
        CONFIDENCE_THRESHOLD = 90
        
        if confidence < CONFIDENCE_THRESHOLD and label_id < len(known_faces_names):
            name = known_faces_names[label_id]
            return name, confidence
        else:
            return "Inconnu", confidence
            
    except Exception as e:
        # print(f"Erreur de prédiction : {e}")
        return "Erreur", 0


def detecter_et_identifier_visages(frame_rgb, cascade):
    """
    Détecte les visages dans une frame et les identifie.
    
    Args:
        frame_rgb (np.ndarray): La frame actuelle en couleur.
        cascade (cv2.CascadeClassifier): Le modèle Haar Cascade.
        
    Returns:
        list: Une liste de dictionnaires [{'box': (x, y, w, h), 'name': '...', 'conf': '...'}]
    """
    if cascade is None:
        return []
        
    # Conversion en niveaux de gris pour la détection
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    
    # 1. Détection des visages (Haar Cascade)
    # Paramètres typiques : 
    # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    faces = cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60) # Ajusté un peu plus grand pour la vidéo
    )
    
    resultats_identification = []
    
    # 2. Identification de chaque visage détecté
    for (x, y, w, h) in faces:
        # Extraire la zone du visage
        face_roi = gray_frame[y:y + h, x:x + w]
        
        name, confidence = identifier_visage(face_roi)
        
        resultats_identification.append({
            'box': (x, y, w, h),
            'name': name,
            'conf': confidence
        })
        
    return resultats_identification

if __name__ == "__main__":
    print("Ce module contient la logique de reconnaissance faciale.")
    # Exécutez l'entraînement si le chemin est défini (pour tester)
    

    preparer_base_de_donnees_visages(DATABASE_FOLDER)
