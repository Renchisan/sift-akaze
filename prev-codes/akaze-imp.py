import cv2
import os
import numpy as np

# Function to load reference images from the 'reference_images' folder
def load_reference_images(reference_folder):
    reference_images = []
    filenames = os.listdir(reference_folder)
    for filename in filenames:
        image_path = os.path.join(reference_folder, filename)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            reference_images.append((image, filename))
    return reference_images

# Function to detect AKAZE keypoints and descriptors
def detect_and_compute(image):
    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(image, None)
    if descriptors is None:
        print("No descriptors found!")
        return [], None
    return keypoints, descriptors

# Function to match features between two images
def match_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return []

    # Ensure descriptors are in the right format (float32)
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Using FLANN based matcher
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return matches, good_matches

# Function to draw matches between query image and reference image
def draw_matches(query_image, reference_image, keypoints_query, keypoints_reference, good_matches):
    img_matches = cv2.drawMatches(query_image, keypoints_query, reference_image, keypoints_reference, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

# Load the query image
query_image = cv2.imread('../IMG_comsoc3.JPG', cv2.IMREAD_GRAYSCALE)

# Load reference images from the folder
reference_images = load_reference_images('reference_images')

# Detect and compute keypoints and descriptors for the query image
keypoints_query, descriptors_query = detect_and_compute(query_image)

# Loop over the reference images and match them with the query image
for reference_image, filename in reference_images:
    # Detect and compute keypoints and descriptors for the reference image
    keypoints_reference, descriptors_reference = detect_and_compute(reference_image)

    # Match the descriptors between the query image and the reference image
    matches, good_matches = match_features(descriptors_query, descriptors_reference)

    # If there are good matches, draw them
    if len(good_matches) > 0:
        img_matches = draw_matches(query_image, reference_image, keypoints_query, keypoints_reference, good_matches)

        # Scale down the image for better display
        img_matches_resized = cv2.resize(img_matches, (800, 600))

        # Show the image with matches
        cv2.imshow(f'Matches with {filename}', img_matches_resized)

        # Print statistics
        print(f"Statistics for {filename}:")
        print(f"  Number of descriptors in query image: {len(descriptors_query)}")
        print(f"  Number of descriptors in reference image: {len(descriptors_reference)}")
        print(f"  Total number of matches: {len(matches)}")
        print(f"  Number of good matches: {len(good_matches)}")
        print(f"  Match accuracy (good matches / total matches): {len(good_matches) / len(matches):.2f}")
        print("-" * 50)

        cv2.waitKey(0)  # Wait for a key press to close the match window

cv2.destroyAllWindows()
