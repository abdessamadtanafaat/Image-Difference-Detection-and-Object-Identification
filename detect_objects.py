import cv2  # Importation de la bibliothèque OpenCV pour le traitement d'images.
import numpy as np  # Importation de NumPy pour les opérations matricielles.
import sys  # Utilisé pour traiter les arguments de la ligne de commande.

# Fonction pour obtenir la région d'intérêt (ROI) spécifique à une pièce.
def get_ROI(room: str):
    """
    Retourne un polygone définissant la région d'intérêt (ROI) pour une pièce donnée.
    :param room: Nom de la pièce ('Chambre', 'Cuisine', 'Salon').
    :return: Coordonnées de la ROI sous forme de tableau NumPy.
    """
    if room == 'Chambre':
        # Coordonnées spécifiques pour la chambre.
        ROI = np.array([
            [3176, 792],
            [5544, 2172],
            [3584, 3912],
            [1287, 2136]
        ], np.int32)
    elif room == 'Cuisine':
        # Coordonnées spécifiques pour la cuisine.
        ROI = np.array([
            [1100, 3999],
            [2000, 2000],
            [3600, 2000],
            [4500, 3999]
        ], np.int32)
    elif room == 'Salon':
        # Coordonnées spécifiques pour le salon.
        ROI = np.array([
            [0, 3999],
            [5900, 3999],
            [3290, 2250],
            [264, 2893]
        ], np.int32)
    else:
        # Erreur si le nom de la pièce est invalide.
        raise ValueError('@param room must be one of the following: Chambre, Cuisine, Salon')

    return ROI

# Fonction pour appliquer un masque à une image en fonction de la ROI.
def mask_image(image, ROI):
    """
    Applique un masque sur l'image pour isoler uniquement la région d'intérêt (ROI).
    :param image: Image source.
    :param ROI: Coordonnées de la ROI.
    :return: Image masquée.
    """
    mask = np.zeros_like(image)  # Crée un masque noir de la même taille que l'image.
    cv2.fillPoly(mask, [ROI], (255, 255, 255))  # Remplit la ROI avec du blanc.
    return cv2.bitwise_and(image, mask)  # Applique le masque sur l'image.

# Fonction principale pour détecter les objets dans une pièce.
def detect_objects(ref_path: str, image_path: str, room: str, threshold_value: int = 80, show_steps: bool = False, save_steps: bool = False):
    """
    Détecte les objets dans une image en comparant avec une image de référence.
    :param ref_path: Chemin de l'image de référence.
    :param image_path: Chemin de l'image à analyser.
    :param room: Nom de la pièce ('Chambre', 'Cuisine', 'Salon').
    :param threshold_value: Valeur de seuil pour la détection.
    :param show_steps: Affiche les étapes intermédiaires si True.
    """
    # Chargement des images.
    ref = cv2.imread(ref_path)  # Image de référence.
    image = cv2.imread(image_path)  # Image actuelle.
    image_contours = image.copy()  # Copie de l'image pour y dessiner les contours.

    # Récupération de la ROI et masquage des images.
    ROI = get_ROI(room)
    ref = mask_image(ref, ROI)
    image = mask_image(image, ROI)

    # Conversion des images en différents espaces colorimétriques (LAB et HSV).
    ref_LAB = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    ref_HSV = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Séparation des canaux (L, A, B pour LAB et H, S, V pour HSV).
    ref_L, ref_A, ref_B = cv2.split(ref_LAB)
    ref_H, ref_S, ref_V = cv2.split(ref_HSV)
    image_L, image_A, image_B = cv2.split(image_LAB)
    image_H, image_S, image_V = cv2.split(image_HSV)

    # Calcul des différences absolues entre les canaux.
    diff_L = cv2.absdiff(ref_L, image_L)
    diff_A = cv2.absdiff(ref_A, image_A)
    diff_B = cv2.absdiff(ref_B, image_B)
    diff_H = cv2.absdiff(ref_H, image_H)
    diff_S = cv2.absdiff(ref_S, image_S)
    diff_V = cv2.absdiff(ref_V, image_V)

    # Combinaison pondérée des différences pour créer une carte de différences.
    diff = diff_L * 0.1 + diff_A * 0.3 + diff_B * 0.3 + diff_H * 0.4 + diff_S * 0.2
    if room == 'Salon':
        # Ajustement pour le salon.
        diff = diff * 0.3 + diff_V * 0.3 + diff_S * 0.2 + diff_A * 0.1 + diff_B * 0.1
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Floutage pour lisser la carte de différences.
    diff = cv2.blur(diff, (25, 25))

    # Affichage de la carte des différences si demandé.
    if show_steps:
        cv2.imshow(f'diff', cv2.resize(diff, (0, 0), fx=0.15, fy=0.15))

    # Binarisation de la carte des différences avec un seuil.
    _, diff_thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Affichage du seuil si demandé.
    if show_steps:
        cv2.imshow(f'diff_thresh', cv2.resize(diff_thresh, (0, 0), fx=0.15, fy=0.15))

    # Application d'opérations morphologiques pour améliorer le masque binaire.
    kernel_morph = np.ones((25, 25), np.uint8)
    diff_morph = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel_morph)
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_OPEN, kernel_morph)
    diff_morph = cv2.dilate(diff_morph, kernel_morph, iterations=2)

    # Affichage des résultats morphologiques si demandé.
    if show_steps:
        cv2.imshow(f'morph', cv2.resize(diff_morph, (0, 0), fx=0.15, fy=0.15))

    # Détection des contours des objets détectés.
    contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]  # Filtrage des petits contours.

    # Dessin des contours et rectangles englobants sur l'image.
    image_contours = cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 4)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_contours, (x, y), (x + w, y + h), (0, 0, 255), 5)

    # Dessin de la ROI sur l'image.
    image_contours = cv2.polylines(image_contours, [ROI], True, (255, 0, 0), 3)

    # Affichage final de l'image avec les contours.
    cv2.imshow(f'contours_{image_path}', cv2.resize(image_contours, (0, 0), fx=0.15, fy=0.15))
    cv2.waitKey(0)

# Exécution principale (ligne de commande).
if __name__ == '__main__':
    # Vérification des arguments.
    if len(sys.argv) < 4:
        print("Usage: python detect_objects.py <path/reference> <path/image> <room> [show_steps : '0' or '1'] optional")
        sys.exit(1)

    # Lecture des paramètres.
    ref_path = sys.argv[1]
    image_path = sys.argv[2]
    room = sys.argv[3]

    if len(sys.argv) == 5:
        show_steps = bool(int(sys.argv[4]))

    # Lancement de la détection.
    detect_objects(ref_path, image_path, room, show_steps=show_steps)
