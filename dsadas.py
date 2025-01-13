import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA


# Function to detect keypoints using AKAZE
def detect_keypoints_with_akaze(image):
   akaze = cv2.AKAZE_create()
   keypoints = akaze.detect(image, None)
   return keypoints


# Function to compute SIFT descriptors
def compute_sift_descriptors(image, keypoints):
   sift = cv2.SIFT_create()
   keypoints, descriptors = sift.compute(image, keypoints)
   return keypoints, descriptors


# Function to threshold descriptors (95% of max value in each descriptor)
def threshold_descriptors(descriptors, threshold_fraction=0.05):
   thresholded_descriptors = np.copy(descriptors)
   max_values = np.max(thresholded_descriptors, axis=1, keepdims=True)
   thresholded_descriptors[thresholded_descriptors < threshold_fraction * max_values] = 0
   return thresholded_descriptors






# Function to build a KD-Tree
def build_kd_tree(thresholded_descriptors, leafsize=10):
   tree = cKDTree(thresholded_descriptors, leafsize=leafsize)
   return tree


# Function to perform approximate matching with KD-Tree (Approximate Nearest Neighbor Search)
def custom_kd_tree_match(descriptors1, descriptors2, k=2, leafsize=10, ratio_thresh=0.7):
   # Threshold descriptors before building KD-Tree
   descriptors1_thresholded = threshold_descriptors(descriptors1)
   descriptors2_thresholded = threshold_descriptors(descriptors2)


   tree = build_kd_tree(descriptors2_thresholded, leafsize=leafsize)  # Build KD-Tree for reference descriptors
   matches = []


   # Perform approximate nearest neighbor search with threading for parallelism(CHATGPT)
   with ThreadPoolExecutor() as executor:
       results = list(executor.map(lambda descriptor: tree.query(descriptor, k=k), descriptors1_thresholded))


   for i, (distances, indices) in enumerate(results):
       if len(distances) == 2:
           m, n = distances[0], distances[1]
           idx_m, idx_n = indices[0], indices[1]
           # Apply the ratio test (m is the closest, n is the second closest)
           if m < ratio_thresh * n:  # Adjust this threshold as needed
               matches.append((i, idx_m, m))  # (Query Index, Match Index, Distance)


   return matches


# Function with PCA for KD-Tree Matching
def pca_kd_tree_match(descriptors1, descriptors2, n_components=64, k=2, leafsize=10, ratio_thresh=0.7):
   # Apply PCA for dimensionality reduction
   pca = PCA(n_components=n_components)
   descriptors1_reduced = pca.fit_transform(descriptors1)
   descriptors2_reduced = pca.transform(descriptors2)


   # Threshold descriptors after PCA
   descriptors1_thresholded = threshold_descriptors(descriptors1_reduced)
   descriptors2_thresholded = threshold_descriptors(descriptors2_reduced)


   # Build KD-Tree for reference descriptors
   tree = build_kd_tree(descriptors2_thresholded, leafsize=leafsize)
   matches = []


   # Perform approximate nearest neighbor search with threading for parallelism
   with ThreadPoolExecutor() as executor:
       results = list(executor.map(lambda descriptor: tree.query(descriptor, k=k), descriptors1_thresholded))


   for i, (distances, indices) in enumerate(results):
       if len(distances) == 2:
           m, n = distances[0], distances[1]
           idx_m, idx_n = indices[0], indices[1]
           # Apply the ratio test (m is the closest, n is the second closest)
           if m < ratio_thresh * n:  # Adjust this threshold as needed
               matches.append((i, idx_m, m))  # (Query Index, Match Index, Distance)


   return matches


# Function to these visualize matches
def visualize_matches(query_image, reference_image, keypoints_query, keypoints_reference, matches, title):
   match_image = cv2.drawMatches(query_image, keypoints_query, reference_image, keypoints_reference, matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   match_image = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
   plt.figure(figsize=(15, 7))
   plt.title(title)
   plt.imshow(match_image)
   plt.axis('off')
   plt.show()


def visualize_query_and_best_match(query_image, reference_image, title):
   # Convert images from grayscale to RGB for visualization
   query_image_rgb = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
   reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2RGB)


   # Display both images side by side using matplotlib
   plt.figure(figsize=(12, 6))
   plt.subplot(1, 2, 1)
   plt.title("Query Image")
   plt.imshow(query_image_rgb)
   plt.axis('off')


   plt.subplot(1, 2, 2)
   plt.title(f"The Query Image Contains: {title}")
   plt.imshow(reference_image_rgb)
   plt.axis('off')


   plt.show()


# Define a function to print the location based on the image name
def get_location_from_image(image_name):
   if '3775' in image_name:
       return "You're in the vicinity of CCIS Lobby"
   elif '3784' in image_name:
       return "You're in the vicinity of the COMSOC AND ITSOC Office"
   elif '3821' in image_name:
       return "You're in front the Hyflex Room 1"
   else:
       return "Location Unknown"




query_image_path = 'IMG_comsoc1.JPG'
reference_image_folder = 'reference_images'


if not os.path.exists(reference_image_folder):
   print(f"Error: Folder '{reference_image_folder}' does not exist.")
   exit()


if not os.path.exists(query_image_path):
   print("Error: Query image does not exist.")
   exit()


query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
if query_image is None:
   print(f"Error: Could not load query image at {query_image_path}.")
   exit()


keypoints_query = detect_keypoints_with_akaze(query_image)
keypoints_query, descriptors_query = compute_sift_descriptors(query_image, keypoints_query)


best_match_name_kd_tree = None
best_match_count_kd_tree = 0


best_match_name_pca = None
best_match_count_pca = 0


for reference_image_name in os.listdir(reference_image_folder):
   reference_image_path = os.path.join(reference_image_folder, reference_image_name)
   reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
   if reference_image is None:
       continue


   keypoints_reference = detect_keypoints_with_akaze(reference_image)
   keypoints_reference, descriptors_reference = compute_sift_descriptors(reference_image, keypoints_reference)


   # Custom KD-Tree Matching
   start_time = time.time()
   good_matches_kd_tree = custom_kd_tree_match(descriptors_query, descriptors_reference)
   kd_tree_matching_time = time.time() - start_time
   print(f"\nProcessing {reference_image_name}:")
   print(f"Custom KD-Tree Matching Time: {kd_tree_matching_time:.4f} seconds")
   print(f"Number of good Custom KD-Tree matches: {len(good_matches_kd_tree)}")


   if len(good_matches_kd_tree) > best_match_count_kd_tree:
       best_match_name_kd_tree = reference_image_name
       best_match_count_kd_tree = len(good_matches_kd_tree)


   # PCA-based KD-Tree Matching
   start_time = time.time()
   good_matches_pca = pca_kd_tree_match(descriptors_query, descriptors_reference, n_components=64)
   pca_matching_time = time.time() - start_time
   print(f"PCA KD-Tree Matching Time: {pca_matching_time:.4f} seconds")
   print(f"Number of good PCA KD-Tree matches: {len(good_matches_pca)}")


   if len(good_matches_pca) > best_match_count_pca:
       best_match_name_pca = reference_image_name
       best_match_count_pca = len(good_matches_pca)


   # Visualization for both methods
   matches_for_visualization_kd_tree = [
       cv2.DMatch(match[0], match[1], match[2]) for match in good_matches_kd_tree
   ]
   matches_for_visualization_pca = [
       cv2.DMatch(match[0], match[1], match[2]) for match in good_matches_pca
   ]


   visualize_matches(query_image, reference_image, keypoints_query, keypoints_reference,
                     matches_for_visualization_kd_tree,
                     f"Custom KD-Tree Matches with {reference_image_name}")
   visualize_matches(query_image, reference_image, keypoints_query, keypoints_reference,
                     matches_for_visualization_pca,
                     f"PCA KD-Tree Matches with {reference_image_name}")


# After the matching process and before visualization, add the location message
print(f"\nBest Match with Custom KD-Tree: {best_match_name_kd_tree} with {best_match_count_kd_tree} matches")
print(get_location_from_image(best_match_name_kd_tree))


print(f"Best Match with PCA KD-Tree: {best_match_name_pca} with {best_match_count_pca} matches")
print(get_location_from_image(best_match_name_pca))


# Load the best reference image for visualization
best_reference_image_path_kd_tree = os.path.join(reference_image_folder, best_match_name_kd_tree)
best_reference_image_kd_tree = cv2.imread(best_reference_image_path_kd_tree, cv2.IMREAD_GRAYSCALE)


# Visualize the query image and the best matching reference image for KD-Tree method
visualize_query_and_best_match(query_image, best_reference_image_kd_tree, best_match_name_kd_tree)


# Load the best reference image for PCA visualization
best_reference_image_path_pca = os.path.join(reference_image_folder, best_match_name_pca)
best_reference_image_pca = cv2.imread(best_reference_image_path_pca, cv2.IMREAD_GRAYSCALE)


# Visualize the query image and the best matching reference image for PCA method
visualize_query_and_best_match(query_image, best_reference_image_pca, best_match_name_pca)

