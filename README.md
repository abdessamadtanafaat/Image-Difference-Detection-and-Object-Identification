# Image Difference Detection and Object Identification
## Overview
This Python-based project implements image processing techniques to detect and identify differences between reference and test images. The goal is to analyze visual changes in various environments (e.g., rooms like 'Chambre', 'Cuisine', and 'Salon') by detecting regions of interest (ROI) and visualizing the differences with contours and bounding boxes. The project also generates detailed reports in PDF format containing both intermediate and final results.

## Features
Image Preprocessing: Loading, conversion, and normalization of images.
ROI Detection: Defining regions of interest (ROI) for accurate image comparison.
Difference Detection: Identifying differences between the reference image and test images.
Morphological Operations: Applying thresholds and morphological transformations to refine the results.
Contour Detection: Extracting contours and visualizing detected changes with bounding boxes.
PDF Report Generation: Saving intermediate and final results as PDF documents.
## Technologies Used
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

pip install opencv-python numpy pillow img2pdf
