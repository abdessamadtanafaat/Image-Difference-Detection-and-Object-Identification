Image Difference Detection and Object Identification
Overview
This Python-based project implements image processing techniques to detect and identify differences between reference and test images. The goal is to analyze visual changes in various environments (e.g., rooms like 'Chambre', 'Cuisine', and 'Salon') by detecting regions of interest (ROI) and visualizing the differences with contours and bounding boxes. The project also generates detailed reports in PDF format containing both intermediate and final results.

Features
Image Preprocessing: Loading, conversion, and normalization of images.
ROI Detection: Defining regions of interest (ROI) for accurate image comparison.
Difference Detection: Identifying differences between the reference image and test images.
Morphological Operations: Applying thresholds and morphological transformations to refine the results.
Contour Detection: Extracting contours and visualizing detected changes with bounding boxes.
PDF Report Generation: Saving intermediate and final results as PDF documents.
Technologies Used
OpenCV: Image processing library for reading, transforming, and analyzing images.
NumPy: Numerical operations on image arrays and matrices.
Pillow (PIL): Image manipulation for converting and saving images.
img2pdf: Converting images into PDF format.
Getting Started
Prerequisites
Ensure that you have the following dependencies installed:

Python 3.x
OpenCV
NumPy
Pillow
img2pdf
You can install the required libraries with the following command:

bash
Copy code
pip install opencv-python numpy pillow img2pdf
Setup
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/image-difference-detection.git
Running the Program
Place your reference and test images in the Images/<room> directory. Replace <room> with the appropriate room name (e.g., 'Chambre', 'Cuisine', 'Salon').
Run the script with the following command:
bash
Copy code
python detect_objects.py Images/Chambre/Reference.JPG Images/Chambre/IMG_1.JPG Chambre 1
This will process the images and generate output in the Outputs/<room> directory.

Output
The script will generate PDF reports containing:

Intermediate steps (difference maps, thresholded images, morphological transformations).
Final results with contours and bounding boxes drawn around detected differences.
Example Output:
After running the script, the following files will be generated:

Outputs/Chambre/steps_1.pdf: Contains intermediate steps for the first test image.
Outputs/Chambre/final_results_Chambre.pdf: Contains the final results for the entire room.
