import cv2
import img2pdf
from PIL import Image
import io

def save_outputs2pdf(image_list, pdf_path):
    """
    Convertit une liste d'images en un fichier PDF et le sauvegarde à un chemin spécifié.

    Args:
        image_list (list): Liste d'images en format OpenCV (BGR).
        pdf_path (str): Chemin complet où le fichier PDF sera sauvegardé.
    """
    # Initialiser une liste pour stocker les données des images au format bytes
    image_bytes_list = []

    for img in image_list:
        # Convertir une image OpenCV (BGR) en format RGB pour être compatible avec Pillow
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Créer une image Pillow (PIL) à partir de l'image RGB
        pil_img = Image.fromarray(img_rgb)

        # Sauvegarder l'image Pillow dans un tampon mémoire (en bytes)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')  # Sauvegarder au format JPEG
        img_byte_arr = img_byte_arr.getvalue()  # Obtenir les données en bytes de l'image

        # Ajouter les bytes de l'image à la liste
        image_bytes_list.append(img_byte_arr)

    # Créer et sauvegarder un fichier PDF contenant toutes les images
    with open(pdf_path, "wb") as f:
        # Utiliser la bibliothèque img2pdf pour convertir les bytes en un fichier PDF
        f.write(img2pdf.convert(image_bytes_list))
