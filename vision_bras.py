# vision_bras.py (VERSION FINALE - INTÉGRATION POSE ET FACIALE)

import cv2
import test1 
from detection_bras_lever import est_bras_leve 
from detection_bras_lever import est_debout 
import reconnaissance_faciale # Importation du nouveau module
import time
import numpy as np

# Configuration pour le dessin des keypoints (de l'original)
KEYPOINT_COLOR = (125,222, 0) # Cyan/Jaune
KEYPOINT_RADIUS = 5 
KEYPOINT_THICKNESS = -1 # Rempli

# Configuration pour le dessin des boîtes de détection YOLO (inchangé)
BOX_COLOR = (0, 255, 255) # Jaune vif
BOX_THICKNESS = 2

# Configuration pour le dessin des boîtes de visage (NOUVEAU)
FACE_BOX_COLOR_KNOWN = (0, 255, 0) # Vert pour connu
FACE_BOX_COLOR_UNKNOWN = (0, 0, 255) # Rouge pour inconnu
FACE_BOX_THICKNESS = 2


def run_pose_estimation_webcam_avec_detection(model_pose_path, model_detect_path):
    """
    Exécute la détection de pose, la détection d'objets (YOLO) et la reconnaissance faciale (Haar/LBPH).
    """
    
    # 1. Charger les modèles et la base de données faciale
    print("\n--- Chargement des modèles ---")
    model_pose = test1.charger_modele(model_pose_path, task="pose")
    model_detect = test1.charger_modele(model_detect_path, task="detect") 
    
    # Charger le classifieur Haar Cascade
    haar_cascade = reconnaissance_faciale.charger_haarcascade(
        reconnaissance_faciale.HAAR_CASCADE_PATH
    )
    
    # Préparer et entraîner le modèle de reconnaissance faciale
    db_entrainee = reconnaissance_faciale.preparer_base_de_donnees_visages(
        reconnaissance_faciale.DATABASE_FOLDER
    )
    
    if model_pose is None:
        print("ERREUR CRITIQUE: Modèle de pose manquant. Arrêt.")
        return
    if haar_cascade is None or not db_entrainee:
        print("ATTENTION: La reconnaissance faciale est désactivée (Haar Cascade ou DB non prêt).")

    # 2. Ouvrir la capture vidéo (Webcam)
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la webcam (index 0).")
        return

    # Initialisation pour le FPS
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_text = "FPS: N/A"

    # 3. Boucle de traitement des frames
    print("\nDémarrage de la détection. Appuyez sur 'q' pour quitter.")
    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            annotated_frame = frame.copy() 
            
            # --- 1. Reconnaissance Faciale (Haar + LBPH) ---
            resultats_visages = []
            if haar_cascade and db_entrainee:
                resultats_visages = reconnaissance_faciale.detecter_et_identifier_visages(
                    frame, haar_cascade
                )
                
                for visage in resultats_visages:
                    x, y, w, h = visage['box']
                    name = visage['name']
                    confidence = visage['conf']
                    
                    # Déterminer la couleur
                    color = FACE_BOX_COLOR_KNOWN if name not in ("Inconnu", "Erreur", "Non entraîné") else FACE_BOX_COLOR_UNKNOWN
                    
                    # Dessiner la boîte autour du visage
                    cv2.rectangle(
                        annotated_frame, (x, y), (x + w, y + h), color, FACE_BOX_THICKNESS
                    )
                    
                    # Afficher le nom et la confiance
                    label = f"{name} ({confidence:.1f})"
                    cv2.putText(
                        annotated_frame, label, 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )
            
            # --- 2. Inférence Modèle de Détection YOLO (Personnes/Objets) ---
            if model_detect:
                results_detect = test1.executer_inference_frame(model_detect, frame)
                
                for result in results_detect:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box[:4])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                        class_id = int(box[5]) if len(box) > 5 else None
                        label = f"Detect: Class {class_id}" if class_id is not None else "Detect"
                        cv2.putText(
                            annotated_frame, label, 
                            (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2
                        )
            
            # --- 3. Inférence du Modèle de Pose (Keypoints et Logique Bras Levé) ---
            results_pose = test1.executer_inference_frame(model_pose, frame)
            
            # 4. Traitement des résultats de Pose
            text_y_start = 50 
            person_count = 0

            for result in results_pose:
                
                if result.keypoints and result.keypoints.data.shape[0] > 0:
                    
                    for i in range(result.keypoints.data.shape[0]):
                        person_keypoints = result.keypoints.data[i].cpu().numpy() # [17, 3]
                        
                        # --- Dessiner les keypoints ---
                        for j in range(person_keypoints.shape[0]):
                            x, y, conf = person_keypoints[j]
                            if conf > 0.5: 
                                cv2.circle(
                                    annotated_frame, 
                                    (int(x), int(y)), 
                                    KEYPOINT_RADIUS, 
                                    KEYPOINT_COLOR, 
                                    KEYPOINT_THICKNESS
                                )
                        
                        # --- Logique de détection de bras ---
                        bras_droit_leve = est_bras_leve(person_keypoints, bras="droit", seuil_y=10)
                        bras_gauche_leve = est_bras_leve(person_keypoints, bras="gauche", seuil_y=10)
                        
                        # --- Afficher l'état du bras (Texte) ---
                        message = None
                        if bras_droit_leve and bras_gauche_leve:
                            message = "Bras Droit & Gauche Leve!"
                        elif bras_droit_leve:
                            message = "Bras Droit Leve!"
                        elif bras_gauche_leve:
                            message = "Bras Gauche Leve!"
                        
                        if message:
                            color = (0, 0, 255) # Rouge
                            cv2.putText(
                                annotated_frame, f"Pose {i+1}: {message}", 
                                (50, text_y_start + person_count * 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                            )
                            person_count += 1
            
            # 5. Calcul et affichage du FPS 
            fps_frame_count += 1
            if time.time() - fps_start_time >= 1.0: 
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_text = f"FPS: {fps:.2f}"
                fps_start_time = time.time()
                fps_frame_count = 0

            cv2.putText(
                annotated_frame, fps_text, 
                (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            # 6. Afficher la frame
            cv2.imshow("Multi-Model Vision", annotated_frame)
            
            # 7. Quitter la boucle
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("Erreur : Frame vide reçue.")
            break

    # 8. Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    print("Programme terminé.")


if __name__ == "__main__":
    # IMPORTANT: Remplacer ceci par le chemin réel de vos modèles YOLO
    MODEL_POSE_PATH = 'yolo11m-pose.pt' 
    MODEL_DETECT_PATH = yolov8n.pt' 
    
    # Note: Les chemins des fichiers de reconnaissance faciale sont dans 'reconnaissance_faciale.py'


    run_pose_estimation_webcam_avec_detection(MODEL_POSE_PATH, MODEL_DETECT_PATH)
