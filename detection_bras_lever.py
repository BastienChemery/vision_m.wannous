# detection_bras_lever.py (Avec détection Debout/Assis)

import numpy as np

# Définition des indices des keypoints COCO utilisés
# (Nouveaux indices ajoutés pour la détection Debout : hanches et genoux)
KEYPOINT_INDEX = {
    "épaule_gauche": 5,
    "épaule_droite": 6,
    "poignet_gauche": 9,
    "poignet_droit": 10,
    "hanche_gauche": 11,
    "hanche_droite": 12,
    "genou_gauche": 13,
    "genou_droite": 14,
    "nez": 0
}

def get_keypoints_coordinates(keypoints, index):
    """
    Extrait les coordonnées (x, y) et la confiance (conf) d'un keypoint.
    
    Args:
        keypoints (np.ndarray): Tableau des keypoints de forme (N, 3) où N=17
                                (x, y, confidence).
        index (int): L'indice du keypoint désiré.
        
    Returns:
        tuple: (x, y, conf) du keypoint ou None si le tableau est vide/invalide.
    """
    if keypoints.ndim == 2 and keypoints.shape[0] > index:
        # Retourne (x, y, confidence)
        return (int(keypoints[index, 0]), int(keypoints[index, 1]), keypoints[index, 2])
    return None

def est_bras_leve(keypoints, bras="droit", seuil_y=0, seuil_confiance=0.7):
    # (Le code de cette fonction est inchangé par rapport à la dernière version)
    
    if bras == "droit":
        épaule_index = KEYPOINT_INDEX["épaule_droite"]
        poignet_index = KEYPOINT_INDEX["poignet_droit"]
    elif bras == "gauche":
        épaule_index = KEYPOINT_INDEX["épaule_gauche"]
        poignet_index = KEYPOINT_INDEX["poignet_gauche"]
    else:
        return False
        
    épaule_data = get_keypoints_coordinates(keypoints, épaule_index)
    poignet_data = get_keypoints_coordinates(keypoints, poignet_index)
    
    if épaule_data is None or poignet_data is None:
        return False
        
    épaule_y = épaule_data[1]
    épaule_conf = épaule_data[2]
    
    poignet_y = poignet_data[1]
    poignet_conf = poignet_data[2]
    
    # VÉRIFICATION DE LA CONFIANCE
    if épaule_conf < seuil_confiance or poignet_conf < seuil_confiance:
        return False
    
    # VÉRIFICATION DE LA POSITION (Poignet plus haut que l'épaule)
    bras_leve = (poignet_y < épaule_y - seuil_y)
                
    return bras_leve

# NOUVELLE FONCTION AJOUTÉE
def est_debout(keypoints, seuil_ratio=0.5, seuil_confiance=0.7):
    """
    Détermine si une personne est debout en comparant la distance verticale 
    entre la tête/épaule et le genou par rapport à la distance genou/hanche.
    
    Logique simple (heuristique) : 
    Si la distance verticale (Épaule/Hanche) est significativement plus grande 
    que la distance verticale (Hanche/Genou), la personne est probablement debout.
    
    Args:
        keypoints (np.ndarray): Le tableau de keypoints d'une seule personne.
        seuil_ratio (float): Ratio minimum (Hanche-Genou)/(Épaule-Hanche) pour être considéré comme debout.
        seuil_confiance (float): Confiance minimale requise pour tous les keypoints.
        
    Returns:
        bool: True si la personne est considérée comme debout.
    """
    
    # Utiliser les keypoints les plus fiables (Moyenne des deux côtés ou le Nez)
    épaule_g_data = get_keypoints_coordinates(keypoints, KEYPOINT_INDEX["épaule_gauche"])
    épaule_d_data = get_keypoints_coordinates(keypoints, KEYPOINT_INDEX["épaule_droite"])
    hanche_g_data = get_keypoints_coordinates(keypoints, KEYPOINT_INDEX["hanche_gauche"])
    hanche_d_data = get_keypoints_coordinates(keypoints, KEYPOINT_INDEX["hanche_droite"])
    genou_g_data = get_keypoints_coordinates(keypoints, KEYPOINT_INDEX["genou_gauche"])
    genou_d_data = get_keypoints_coordinates(keypoints, KEYPOINT_droite["genou_droite"])

    # Collecter les données pour les points nécessaires
    points_necessaires = [épaule_g_data, épaule_d_data, hanche_g_data, hanche_d_data, genou_g_data, genou_d_data]
    
    # 1. Vérification de la sécurité et de la confiance
    for data in points_necessaires:
        if data is None or data[2] < seuil_confiance: # data[2] est la confiance
            return False # Si un seul point clé manque ou n'est pas fiable, on ne peut pas conclure
            
    # 2. Calcul des positions Y moyennes pour plus de robustesse
    épaule_y = (épaule_g_data[1] + épaule_d_data[1]) / 2
    hanche_y = (hanche_g_data[1] + hanche_d_data[1]) / 2
    genou_y = (genou_g_data[1] + genou_d_data[1]) / 2
    
    # 3. Calcul des distances verticales (Y décroit quand on monte, donc les différences sont positives)
    
    # Distance verticale Épaule -> Hanche (le tronc)
    dist_tronc = hanche_y - épaule_y
    
    # Distance verticale Hanche -> Genou (partie haute de la jambe)
    dist_haut_jambe = genou_y - hanche_y
    
    # Sécurité contre la division par zéro (bien que peu probable si les points sont détectés)
    if dist_tronc <= 0 or dist_haut_jambe <= 0:
        return False
    
    # 4. Logique de détection
    # Quand on est debout, le genou est généralement loin de la hanche. 
    # Quand on est assis, le genou est proche ou plus haut que la hanche (ce qui rend dist_haut_jambe petit ou négatif).
    # Dans ce cas, nous vérifions si la distance (Hanche -> Genou) est une proportion significative de la distance (Épaule -> Hanche).
    
    # Ratio = (Hanche -> Genou) / (Épaule -> Hanche)
    ratio_hanche_genou_vs_tronc = dist_haut_jambe / dist_tronc
    
    # Si le ratio est supérieur au seuil (ex: 0.5), cela signifie que la jambe est suffisamment étendue verticalement
    # pour que la personne soit considérée comme debout.
    est_debout_resultat = ratio_hanche_genou_vs_tronc > seuil_ratio
    
    return est_debout_resultat