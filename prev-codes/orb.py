import cv2
import numpy as np
import os
from sklearn.cluster import KMeans


# Function to detect keypoints using AKAZE (without computing descriptors)
def detect_keypoints_with_akaze(image):
    # Create AKAZE detector
    akaze = cv2.AKAZE_create()
    # Detect only keypoints
    keypoints = akaze.detect(image, None)
    return keypoints


# Function to compute SIFT descriptors for given keypoints
def compute_sift_descriptors(image, keypoints):
    # Create SIFT detector
    sift = cv2.SIFT_create()
    # Compute SIFT descriptors using the provided keypoints
    keypoints, descriptors = sift.compute(image, keypoints)
    return keypoints, descriptors


# Normal FLANN-based matching function
def match_features_with_flann(descriptors1, descriptors2):
    """
    Matches descriptors using the normal FLANN-based matcher.
    """
    # Ensure descriptors are in float32 format
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Use FLANN-based matcher
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches


# Optimized KD-Tree matching function with KMeans
def match_features_with_grouped_kd_tree(descriptors1, descriptors2, k=5):
    """
    Matches descriptors using a grouped KD-Tree approach with KMeans clustering.
    """
    # Group the descriptors using KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    descriptors_combined = np.vstack((descriptors1, descriptors2))
    kmeans.fit(descriptors_combined)

    # Labels for each descriptor based on the cluster
    labels1 = kmeans.labels_[:len(descriptors1)]
    labels2 = kmeans.labels_[len(descriptors1):]

    # Create KD-Trees for each group of descriptors
    flann = cv2.FlannBasedMatcher_create()

    good_matches = []

    for label in np.unique(labels1):  # Iterate over unique clusters
        # Get the descriptors for the current label/group
        group1 = descriptors1[labels1 == label]
        group2 = descriptors2[labels2 == label]

        if len(group1) == 0 or len(group2) == 0:
            continue

        # Perform FLANN-based matching for this group of descriptors
        group_matches = flann.knnMatch(group1, group2, k=2)

        # Filter matches based on ratio test (0.7 is the default threshold)
        for m, n in group_matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    return good_matches


# Function to resize images for display
def resize_image_for_display(image, width=800):
    # Calculate the ratio of the new width to the old width
    height = int((float(image.shape[0]) / float(image.shape[1])) * width)
    resized_image = cv2.resize(image, (width, height))
    return resized_image


# Define paths for images
query_image_path = '../IMG_comsoc3.JPG'
reference_image_folder = 'reference_images'

# Check if the reference images folder exists
if not os.path.exists(reference_image_folder):
    print(f"Error: Folder '{reference_image_folder}' does not exist.")
    exit()

# Check if the query image exists
if not os.path.exists(query_image_path):
    print("Error: Query image 'IMG_test.JPG' does not exist.")
    exit()

# Load the query image
query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

# Check if the query image was loaded correctly
if query_image is None:
    print(f"Error: Could not load query image at {query_image_path}.")
    exit()

# Detect keypoints using AKAZE for the query image
keypoints_query = detect_keypoints_with_akaze(query_image)

# Compute SIFT descriptors for the query image
keypoints_query, descriptors_query = compute_sift_descriptors(query_image, keypoints_query)

# Variables to track the best match
best_match_name_flann = None
best_match_count_flann = 0
best_match_image_flann = None

best_match_name_grouped = None
best_match_count_grouped = 0
best_match_image_grouped = None

# Loop through all reference images in the folder
for reference_image_name in os.listdir(reference_image_folder):
    reference_image_path = os.path.join(reference_image_folder, reference_image_name)

    # Load the reference image
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

    # Skip files that are not images
    if reference_image is None:
        continue

    # Detect keypoints using AKAZE for the reference image
    keypoints_reference = detect_keypoints_with_akaze(reference_image)

    # Compute SIFT descriptors for the reference image
    keypoints_reference, descriptors_reference = compute_sift_descriptors(reference_image, keypoints_reference)

    # Match descriptors between query image and reference image using normal FLANN
    good_matches_flann = match_features_with_flann(descriptors_query, descriptors_reference)

    # Update the best match if this one has more good matches (FLANN)
    if len(good_matches_flann) > best_match_count_flann:
        best_match_count_flann = len(good_matches_flann)
        best_match_name_flann = reference_image_name
        best_match_image_flann = reference_image

    # Match descriptors using grouped KD-Tree approach
    good_matches_grouped = match_features_with_grouped_kd_tree(descriptors_query, descriptors_reference)

    # Update the best match if this one has more good matches (Grouped KD-Tree)
    if len(good_matches_grouped) > best_match_count_grouped:
        best_match_count_grouped = len(good_matches_grouped)
        best_match_name_grouped = reference_image_name
        best_match_image_grouped = reference_image

    # Draw matches for both methods
    img_matches_flann = cv2.drawMatches(query_image, keypoints_query, reference_image, keypoints_reference, good_matches_flann, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_grouped = cv2.drawMatches(query_image, keypoints_query, reference_image, keypoints_reference, good_matches_grouped, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Resize the images for display
    img_matches_resized_flann = resize_image_for_display(img_matches_flann, width=800)
    img_matches_resized_grouped = resize_image_for_display(img_matches_grouped, width=800)

    # Show matches
    cv2.imshow(f"FLANN Matches with {reference_image_name}", img_matches_resized_flann)
    cv2.imshow(f"Grouped KD-Tree Matches with {reference_image_name}", img_matches_resized_grouped)
    cv2.waitKey(0)

    # Output the number of descriptors and matches for both methods
    print(f"Comparing with {reference_image_name}:")
    print(f"Number of keypoints in query image: {len(keypoints_query)}")
    print(f"Number of keypoints in reference image: {len(keypoints_reference)}")
    print(f"Number of good FLANN matches: {len(good_matches_flann)}")
    print(f"Number of good Grouped KD-Tree matches: {len(good_matches_grouped)}")

# Print the best match details for both methods
if best_match_name_flann:
    print(f"\nBest FLANN match is with image '{best_match_name_flann}' with {best_match_count_flann} good matches.")
if best_match_name_grouped:
    print(f"\nBest Grouped KD-Tree match is with image '{best_match_name_grouped}' with {best_match_count_grouped} good matches.")

cv2.destroyAllWindows()
