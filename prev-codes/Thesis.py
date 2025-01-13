import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def sift_keypoint_matching_multiple_references(query_image_path, reference_folder):
    # Step 1: Load the query image
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

    # Step 2: Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Step 3: Detect keypoints and descriptors for the query image
    kp_query, des_query = sift.detectAndCompute(query_image, None)

    # Step 4: FLANN-based matcher initialization
    index_params = dict(algorithm=1, trees=5)  # KDTree algorithm
    search_params = dict(checks=50)  # Number of checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Step 5: Get all reference images from the folder
    reference_images_paths = [os.path.join(reference_folder, f) for f in os.listdir(reference_folder) if f.endswith(('.jpg', '.png'))]

    for reference_image_path in reference_images_paths:
        # Step 6: Load the reference image
        reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

        # Step 7: Detect keypoints and descriptors for the reference image
        kp_ref, des_ref = sift.detectAndCompute(reference_image, None)

        # Print number of descriptors for query and reference image
        print(f"Query Image Descriptors: {len(des_query)}")
        print(f"Reference Image ({os.path.basename(reference_image_path)}) Descriptors: {len(des_ref)}")

        # Step 8: Match descriptors using k-Nearest Neighbors (k=2)
        matches = flann.knnMatch(des_ref, des_query, k=2)

        # Step 9: Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # Print the number of matches
        print(f"Total Matches: {len(matches)}")
        print(f"Good Matches: {len(good_matches)}")

        # Step 10: Calculate the accuracy (as a ratio of good matches to total matches)
        accuracy = len(good_matches) / len(matches) * 100 if len(matches) > 0 else 0

        # Step 11: Draw matches and display
        img_matches = cv2.drawMatches(
            reference_image, kp_ref, query_image, kp_query, good_matches, None,
            matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Step 12: Display the results
        plt.figure(figsize=(10, 6))
        plt.imshow(img_matches)
        plt.title(f"SIFT Matching: {os.path.basename(reference_image_path)} - Accuracy: {accuracy:.2f}%")
        plt.axis("off")
        plt.show()

        # Print the accuracy for each reference image
        print(f"Matching with {os.path.basename(reference_image_path)}: {accuracy:.2f}% good matches\n")

# Define the paths to the query image and the reference folder
query_image_path = '../IMG_comsoc1.JPG'  # Path to the query image
reference_folder = 'reference_images'  # Folder containing reference images

# Call the function to perform SIFT matching
sift_keypoint_matching_multiple_references(query_image_path, reference_folder)
