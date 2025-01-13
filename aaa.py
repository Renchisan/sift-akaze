import cv2
import numpy as np

# Load the image
image_path = "reference_images/IMG_comsoc.JPG"  # Replace with the path to your image
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image to range [0, 1]
normalized_gray_image = gray_image / 255.0

# Print the numerical representation
print("Numerical representation of the grayscale image (normalized):")
print(image)

#
