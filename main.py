import cv2
import numpy as np
import img2pdf
from PIL import Image
import io

# Fonction pour récupérer la région d'intérêt (ROI) correspondant à une pièce spécifique
def get_ROI(room: str):
    """
    Renvoie une région d'intérêt (ROI) sous forme de tableau de points pour une pièce donnée.
    :param room: Nom de la pièce ('Chambre', 'Cuisine', ou 'Salon')
    :return: Tableau de coordonnées définissant la ROI
    :raises ValueError: Si le nom de la pièce est invalide
    """
    if room == 'Chambre':
        ROI = np.array([
            [3176, 792],
            [5544, 2172],
            [3584, 3912],
            [1287, 2136]
        ], np.int32)
    elif room == 'Cuisine':
        ROI = np.array([
            [1100, 3999],
            [2000, 2000],
            [3600, 2000],
            [4500, 3999]
        ], np.int32)
    elif room == 'Salon':
        ROI = np.array([
            [0, 3999],
            [5900, 3999],
            [3290, 2250],
            [264, 2893]
        ], np.int32)
    else:
        raise ValueError('@param room must be one of the following: Chambre, Cuisine, Salon')

    return ROI

# Fonction pour appliquer un masque sur une image en utilisant une ROI
def mask_image(image, ROI):
    """
    Applique un masque polygonal à une image pour isoler une région d'intérêt (ROI).
    :param image: Image d'entrée (numpy array)
    :param ROI: Région d'intérêt sous forme de tableau de points
    :return: Image masquée
    """
    mask = np.zeros_like(image)  # Crée un masque noir de la même taille que l'image
    cv2.fillPoly(mask, [ROI], (255, 255, 255))  # Dessine le polygone blanc sur le masque
    return cv2.bitwise_and(image, mask)  # Applique le masque à l'image

# Fonction pour sauvegarder une liste d'images dans un fichier PDF
def save_outputs2pdf(image_list, pdf_path):
    """
    Convertit une liste d'images en un fichier PDF.
    :param image_list: Liste d'images (numpy arrays)
    :param pdf_path: Chemin de sortie du fichier PDF
    """
    image_bytes_list = []  # Liste pour stocker les données d'image en bytes

    for img in image_list:
        # Convertit l'image de OpenCV (BGR) au format Pillow (RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)  # Convertit en objet Image de Pillow

        # Sauvegarde l'image dans un buffer mémoire en format JPEG
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_bytes_list.append(img_byte_arr.getvalue())  # Ajoute les bytes de l'image

    # Sauvegarde les images en un fichier PDF avec img2pdf
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(image_bytes_list))

# Fonction principale pour détecter les objets en comparant une image de référence et une image cible
def detect_objects(room: str, id: int, threshold_value: int = 80, show_steps: bool = False, save_steps: bool = False):
    """
    Détecte les objets en comparant une image de référence et une image cible dans une pièce donnée.
    :param room: Nom de la pièce ('Chambre', 'Cuisine', 'Salon')
    :param id: Identifiant de l'image à analyser
    :param threshold_value: Valeur du seuil pour la segmentation
    :param show_steps: Afficher ou non les étapes intermédiaires
    :param save_steps: Sauvegarder ou non les étapes intermédiaires dans un PDF
    :return: Image annotée avec les objets détectés
    """
    ref = cv2.imread(f'Images/{room}/Reference.JPG')  # Charge l'image de référence
    image = cv2.imread(f'Images/{room}/IMG_{id}.JPG')  # Charge l'image cible
    image_contours = image.copy()  # Copie de l'image pour annoter les contours

    # Récupère et applique le masque de ROI
    ROI = get_ROI(room)
    ref = mask_image(ref, ROI)
    image = mask_image(image, ROI)

    # Conversion des images en espaces de couleur LAB et HSV
    ref_LAB = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    ref_HSV = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    image_LAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calcul des différences entre les images dans différents canaux
    diff_L = cv2.absdiff(ref_L, image_L)
    diff_A = cv2.absdiff(ref_A, image_A)
    diff_B = cv2.absdiff(ref_B, image_B)
    diff_H = cv2.absdiff(ref_H, image_H)
    diff_S = cv2.absdiff(ref_S, image_S)
    diff_V = cv2.absdiff(ref_V, image_V)

    # Pondération des différences pour obtenir une carte de différences globales
    diff = diff_L * 0.1 + diff_A * 0.3 + diff_B * 0.3 + diff_H * 0.4 + diff_S * 0.2
    if room == 'Salon':
        diff = diff * 0.3 + diff_V * 0.3 + diff_S * 0.2 + diff_A * 0.1 + diff_B * 0.1
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    diff = cv2.blur(diff, (25, 25))  # Application d'un flou pour réduire le bruit

    # Seuil pour segmenter les zones d'intérêt
    _, diff_thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphologie pour nettoyer les contours
    kernel_morph = np.ones((25, 25), np.uint8)
    diff_morph = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel_morph)
    diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_OPEN, kernel_morph)
    diff_morph = cv2.dilate(diff_morph, kernel_morph, iterations=2)

    # Détection des contours
    contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 5000]  # Filtrage par taille

    # Annotation des contours et de la ROI sur l'image
    image_contours = cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 4)
    image_contours = cv2.polylines(image_contours, [ROI], True, (255, 0, 0), 3)

    # Optionnel : Afficher ou sauvegarder les étapes intermédiaires
    if show_steps:
        cv2.imshow(f'image_contours_{id}', cv2.resize(image_contours, (0, 0), fx=0.15, fy=0.15))
        cv2.waitKey(0)
    if save_steps:
        save_outputs2pdf([ref, image, diff, diff_thresh, diff_morph, image_contours], f'Outputs/{room}/steps_{id}.pdf')

    return image_contours

# Point d'entrée du script
if __name__ == '__main__':
    for room in ['Chambre', 'Cuisine', 'Salon']:
        outputs = []  # Réinitialise la liste des résultats pour chaque pièce
        for id in ids4room[room]:
            output = detect_objects(room, id, threshold_value=80, save_steps=True)
            outputs.append(output)

        # Sauvegarde toutes les images annotées finales dans un PDF
        save_outputs2pdf(outputs, f'Outputs/{room}/final_results_{room}.pdf')
