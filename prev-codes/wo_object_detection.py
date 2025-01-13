import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def filter_good_matches(matches, threshold=0.7):
   """
   Filters matches using Lowe's ratio test.
   """
   good_matches = []
   for m, n in matches:
       if m.distance < threshold * n.distance:
           good_matches.append(m)
   return good_matches

def visualize_keypoint_matches(reference_img, query_img, keypoints1, keypoints2, matches):
   """
   Visualizes the keypoint matching between the reference and query images.
   """
   # Draw matches
   matched_img = cv2.drawMatches(reference_img, keypoints1, query_img, keypoints2, matches, None,
                                 matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=2)
   return matched_img

def draw_keypoint_patches(image, keypoints, patch_size=32):
   """
   Draws patches around the keypoints in the image and arranges them in a grid.
   """
   patches = []
   for kp in keypoints:
       # Ensure the center is an integer tuple
       center = (int(np.round(kp.pt[0])), int(np.round(kp.pt[1])))
       # Extract the patch from the image
       patch = cv2.getRectSubPix(image, (patch_size, patch_size), center)
       # Append the patch to the list
       patches.append(patch)
   # Create a grid to display the patches
   num_patches = len(patches)
   grid_size = int(np.ceil(np.sqrt(num_patches)))  # Create a square grid
   # Resize all patches to fit the grid
   patch_grid = np.zeros((grid_size * patch_size, grid_size * patch_size), dtype=np.uint8)


   for i, patch in enumerate(patches):
       row = i // grid_size
       col = i % grid_size
       patch_resized = cv2.resize(patch, (patch_size, patch_size))  # Resize to ensure consistent size
       patch_grid[row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size] = patch_resized


   return patch_grid

def print_descriptors(image_path, sift):
    """
    Prints the SIFT descriptors for keypoints in the given image.
    """
    # Read the image in grayscale
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        print(f"Error: Image at {image_path} could not be loaded.")
        return

    # Detect Key Points and Extract Features using SIFT
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)

    # Check if descriptors are found
    if descriptors is None:
        print(f"No descriptors found in the image {image_path}.")
        return

    # Print each descriptor for every keypoint
    for i, descriptor in enumerate(descriptors):
        print(f"Descriptor {i} for Keypoint {i}: {descriptor}")

def find_best_match(reference_images_folder, query_image_path):
   """
   Matches a query image to all reference images in a folder and returns the best match.
   """
   # Read the query image in grayscale
   query_image_gray = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)


   if query_image_gray is None:
       print(f"Error: Query image at {query_image_path} could not be loaded.")
       return None, None, None


   # Detect Key Points and Extract Features using SIFT
   sift = cv2.SIFT_create()
   scene_keypoints, scene_descriptors = sift.detectAndCompute(query_image_gray, None)


   if len(scene_keypoints) == 0:
       print("Error: No keypoints detected in the query image.")
       return None, None, None


   best_matches_count = 0
   best_result_img = query_image_gray
   best_reference_image = None
   best_reference_image_path = None  # Store the path of the best reference image


   # Print descriptors for the query image
   print("SIFT Descriptors for Query Image:")
   print_descriptors(query_image_path, sift)

   for reference_image_name in os.listdir(reference_images_folder):
       reference_image_path = os.path.join(reference_images_folder, reference_image_name)


       # Read the reference image in grayscale
       reference_image_gray = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)


       if reference_image_gray is None:
           print(f"Error: Reference image at {reference_image_path} could not be loaded.")
           continue


       # Detect Key Points and Extract Features using SIFT
       reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_image_gray, None)


       if len(reference_keypoints) == 0:
           print(f"Error: No keypoints detected in the reference image {reference_image_name}.")
           continue


       # Print descriptors for the reference image
       print(f"\nSIFT Descriptors for Reference Image: {reference_image_name}")
       print_descriptors(reference_image_path, sift)


       # Match Features using FLANN-based matcher
       FLANN_INDEX_KDTREE = 1
       index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
       search_params = dict(checks=50)


       flann = cv2.FlannBasedMatcher(index_params, search_params)
       matches = flann.knnMatch(reference_descriptors, scene_descriptors, k=2)


       # Filter matches using Lowe's ratio test
       good_matches = filter_good_matches(matches)


       # Visualize keypoint matches
       matched_img = visualize_keypoint_matches(reference_image_gray, query_image_gray,
                                                reference_keypoints, scene_keypoints, good_matches)


       # Display the matched keypoints image
       plt.figure(figsize=(10, 5))
       plt.imshow(matched_img)
       plt.title(f"Keypoint Matches for {reference_image_name}")
       plt.axis('off')
       plt.show()


       # Draw and visualize descriptor patches in a grid
       patch_grid = draw_keypoint_patches(reference_image_gray, reference_keypoints)
       plt.figure(figsize=(10, 10))
       plt.imshow(patch_grid, cmap='gray')
       plt.title('Keypoint Patches in a Grid')
       plt.axis('off')
       plt.show()


       # Check if this is the best match
       if len(good_matches) > best_matches_count:
           best_matches_count = len(good_matches)
           best_result_img = matched_img
           best_reference_image = reference_image_name
           best_reference_image_path = reference_image_path


   if best_reference_image is None:
       print("Error: No good matches found in any reference images.")
       return None, None, None


   return best_reference_image, best_result_img, best_reference_image_path




# Define the folder containing reference images and the query image
reference_images_folder = r'.\reference_images'
query_image_path = r'../IMG_csoffice1.jpg'  # Query image path


# Find the best match
best_reference_image, best_result_img, best_reference_image_path = find_best_match(reference_images_folder,
                                                                                  query_image_path)

# Display the best match result
if best_reference_image:
   print(f"Best match found with reference image: {best_reference_image}")


   # Display the query image with the keypoint matches
   plt.figure(figsize=(10, 5))
   plt.subplot(1, 2, 1)
   plt.imshow(best_result_img, cmap='gray')
   plt.title('Best Match Result')
   plt.axis('off')


   # Display the closest reference image in grayscale
   reference_img_gray = cv2.imread(best_reference_image_path,
                                   cv2.IMREAD_GRAYSCALE)  # Read the closest reference image in grayscale
   plt.subplot(1, 2, 2)
   plt.imshow(reference_img_gray, cmap='gray')  # Display it in grayscale
   plt.title(f'Closest Reference: {best_reference_image}')
   plt.axis('off')


   plt.show()
else:
   print("No good matches found.")
