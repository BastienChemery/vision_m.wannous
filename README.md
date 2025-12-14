# 🧑‍💻 Projet de Vision par Ordinateur : Analyse du Comportement en Classe

Ce projet a été développé dans le cadre d'un cours de vision par ordinateur et vise à analyser le comportement des élèves dans une salle de classe via un flux vidéo. L'objectif principal est de détecter et de suivre les positions corporelles et faciales pour déterminer l'état des élèves (assis, debout, main levée, etc.).

---

## 🎯 Objectif Principal

Déployer une caméra dans une salle de classe pour observer et analyser le comportement des élèves. L'analyse se base sur la détection des **points clés du corps (keypoints)** et des **boîtes englobantes (bounding boxes)** pour déterminer des actions spécifiques.

---

## 🏗️ Architecture du Code

Le projet est structuré en **trois composants principaux** qui travaillent ensemble pour réaliser l'analyse complète.

### 1. ⚙️ L'Orchestrateur

Le rôle de l'Orchestrateur est de servir de **point d'entrée principal** pour le système. Il est responsable de :

* **L'exécution séquentielle et parallèle** des autres modules.
* La **gestion du flux de données** entre le module de détection/tracking et le module d'analyse comportementale.
* La **configuration** et l'initialisation du système.

### 2 et 2.5 🧠 Module de Détection et de Tracking

Ce module gère le traitement bas niveau du flux vidéo. Il est chargé de :

* **Charger les modèles** de détection pré-entraînés (probablement basés sur des architectures comme Yolo, OpenPose, etc.).
* Effectuer la **Détection des Keypoints du Corps** (épaules, coudes, mains, genoux, etc.) pour déterminer la posture.
* Détecter les **Boîtes Englobantes du Corps et du Visage**.
* Assurer le **Tracking Multi-Objets** pour maintenir l'identité de chaque élève à travers les images.
* **Détecter les visages** presents dans la base de données.



### 3. 📊 Module d'Analyse Comportementale

Ce module reçoit les données de position et de tracking du module précédent et les interprète pour en tirer des conclusions comportementales. Ses fonctions incluent :

* L'**Analyse de Posture** pour déterminer l'état général (e.g., **Debout**, **Assis**).
* La **Détection d'Actions Spécifiques** (e.g., **Main Levée**).
* L'utilisation de la détection de visage pour la **Re-identification (Re-ID)** des élèves entre les cadres, garantissant un suivi précis des comportements individuels sur une longue période.

---

## 🛠️ Technologies Clés

*(À remplir si vous avez des librairies ou frameworks spécifiques : Python, OpenCV, TensorFlow/PyTorch, Yolo, OpenPose, etc.)*

---

## ▶️ Comment Démarrer

* pip install -r requirements.txt
* Mettre plusieurs photos de soi dans un fichier nommé de son nom dans le fichier tete.
* Puis lancez vision_bras.py

