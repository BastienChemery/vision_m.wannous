# test1.py (VERSION MODIFIÉE - AJOUT DE LA DÉTECTION DE TÊTE/PERSONNE)

import cv2
from ultralytics import YOLO
import time 
import numpy as np # Ajouté pour les types NumPy si nécessaire

def charger_modele(model_path, task="pose"):
    """
    Charge et retourne un modèle YOLO (pose ou detect).
    
    Args:
        model_path (str): Le chemin vers le fichier du modèle.
        task (str): La tâche du modèle ('pose' ou 'detect').
        
    Returns:
        YOLO: Le modèle chargé ou None en cas d'erreur.
    """
    print(f"Chargement du modèle de {task} : {model_path}...")
    try:
        # Tâche par défaut est 'pose' pour la compatibilité avec l'ancien code.
        model = YOLO(model_path, task=task)
        print(f"Modèle de {task} chargé avec succès.")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle de {task} : {e}")
        return None

# Renommé et adapté pour accepter le modèle en paramètre
def executer_inference_frame(model, frame):
    """
    Exécute la détection (pose ou detect) sur une seule frame.
    
    Args:
        model (YOLO): Le modèle de détection chargé.
        frame (np.ndarray): La frame à analyser.
        
    Returns:
        list: La liste des objets Result d'Ultralytics.
    """
    if model is None:
        return []
        
    # 'stream=False' pour une seule frame est plus direct.
    # Utilisation d'un seuil de confiance par défaut de 0.5
    results = model(frame, stream=False, conf=0.5, save=False, verbose=False)
    return results

if __name__ == "__main__":
    print("test1.py est maintenant un module d'utilitaires pour le chargement et l'inférence YOLO.")