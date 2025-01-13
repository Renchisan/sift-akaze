import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Functions for Gaussian Pyramid, DoG, and Keypoint Detection
def gaussian_blur(image, sigma):
  return cv2.GaussianBlur(image, (0, 0), sigma)

def create_gaussian_pyramid(image, num_octaves, num_scales):
  pyramid = []
  for i in range(num_octaves):
      octave = []
      for j in range(num_scales):
          sigma = 1.6 * (2 ** (j / num_scales))
          blurred = gaussian_blur(image, sigma)
          octave.append(blurred)
      pyramid.append(octave)
      image = octave[-1][::2, ::2]  # Downsample for the next octave
  return pyramid

def difference_of_gaussian(pyramid):
  dog_pyramid = []
  for octave in pyramid:
      dog_octave = []
      for i in range(len(octave) - 1):
          dog = cv2.subtract(octave[i], octave[i + 1])
          dog_octave.append(dog)
      dog_pyramid.append(dog_octave)
  return dog_pyramid

def detect_keypoints(dog_pyramid):
  keypoints = []
  for octave in dog_pyramid:
      for dog in octave:
          # Detect keypoints by finding local extrema in the DoG space
          local_keypoints = cv2.goodFeaturesToTrack(dog, maxCorners=1000, qualityLevel=0.01, minDistance=10)
          if local_keypoints is not None:
              # Convert local keypoints into cv2.KeyPoint objects
              for pt in local_keypoints:
                  keypoints.append(cv2.KeyPoint(pt[0][0], pt[0][1], 1))  # (x, y, size)
  return keypoints

def compute_descriptors(keypoints, image):
  descriptors = []
  for keypoint in keypoints:
      x, y = keypoint.pt  # Extract (x, y) directly from the .pt attribute
      # Extract 16x16 patch around the keypoint and compute descriptor
      patch = image[int(y) - 8:int(y) + 8, int(x) - 8:int(x) + 8]
      if patch.shape == (16, 16):
          descriptor = patch.flatten()  # Example: Flatten the patch
          descriptors.append(descriptor)
  return descriptors

# Function to draw object detection
def draw_object_detection(reference_img, query_img, keypoints1, keypoints2, matches, min_match_count=10):
  """
  Detects the object in the scene using homography and draws the bounding box on the query image.
  """
  if len(matches) >= min_match_count:
      # Extract points from the matches
      src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
      dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
      # Find homography matrix
      homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

      if homography is not None:
          print("Homography matrix found, transforming corners...")
          # Get the corners of the reference image
          h, w = reference_img.shape
          corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
          # Transform corners to the query image perspective
          transformed_corners = cv2.perspectiveTransform(corners, homography)
          print("Transformed corners:", transformed_corners)
          # Draw the bounding box on the query image
          result_img = query_img.copy()
          result_img = cv2.polylines(result_img, [np.int32(transformed_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
          return result_img, mask.ravel().tolist()
      else:
          print("Homography not found.")
          return query_img, None
  else:
      print(f"Not enough matches are found - {len(matches)}/{min_match_count}")
      return query_img, None

def visualize_keypoint_matching(reference_img, query_img, keypoints1, keypoints2, matches):
  """
  Visualize keypoint matching between reference and query images.
  Resizes the image matching result and displays it in a subplot.
  """
  # Draw the matches between the reference and query images
  img_matches = cv2.drawMatches(
      reference_img, keypoints1, query_img, keypoints2, matches, None,
      matchColor=(0, 255, 0),  # Green for good matches
      singlePointColor=(255, 0, 0),  # Red for keypoints
      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
  )
  # Resize the image matching result to fit into the subplot
  img_matches_resized = cv2.resize(img_matches, (600, 400))  # Resize the image to a specific size
  return img_matches_resized

def visualize_keypoints(image, keypoints):
   image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))
   cv2.imshow("Keypoints", image_with_keypoints)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def visualize_descriptors(keypoints, image, patch_size=16, grid_size=(100, 100)):
   """
   Visualize the computed descriptors by displaying the image patches around each keypoint
   in a 20x20 grid layout.
   """
   patches = []
   for keypoint in keypoints:
       x, y = keypoint.pt  # Extract (x, y) directly from the .pt attribute
       # Extract a patch around the keypoint, here we assume a 16x16 patch
       patch = image[int(y) - patch_size // 2:int(y) + patch_size // 2,
               int(x) - patch_size // 2:int(x) + patch_size // 2]
       if patch.shape == (patch_size, patch_size):
           patches.append(patch)


   num_patches = len(patches)
   if num_patches == 0:
       print("No patches to display.")
       return
   # Calculate grid dimensions based on the number of patches
   rows = grid_size[0]
   cols = grid_size[1]
   # If there are more patches than the grid can handle, extend the grid size
   if num_patches > rows * cols:
       rows = (num_patches // cols) + (1 if num_patches % cols != 0 else 0)
   # Create an empty image to hold the grid of patches
   patch_image = np.zeros((patch_size * rows, patch_size * cols), dtype=np.uint8)

   for i, patch in enumerate(patches):
       row = i // cols
       col = i % cols
       patch_image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = patch

   # Show the patch image in a separate window
   cv2.imshow("Keypoint Descriptors", patch_image)
   cv2.waitKey(0)  # Wait for a key press to close the window
   cv2.destroyAllWindows()  # Close the window after key press

def find_best_match(reference_images_folder, query_image_path, min_match_count=10):
   query_image_gray = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
   # Step 2: Create Gaussian Pyramid and compute DoG
   num_octaves = 4
   num_scales = 5
   gaussian_pyramid = create_gaussian_pyramid(query_image_gray, num_octaves, num_scales)
   dog_pyramid = difference_of_gaussian(gaussian_pyramid)
   # Detect keypoints and compute descriptors for the query image
   query_keypoints = detect_keypoints(dog_pyramid)
   query_descriptors = compute_descriptors(query_keypoints, query_image_gray)
   # Visualize the descriptors in a separate window
   visualize_descriptors(query_keypoints, query_image_gray)

   best_matches_count = 0
   best_result_img = query_image_gray
   best_reference_image = None
   best_reference_image_path = None

   # FLANN parameters
   index_params = dict(algorithm=1, trees=5)  # Use KDTree
   search_params = dict(checks=50)  # Number of checks

   flann = cv2.FlannBasedMatcher(index_params, search_params)

   for reference_image_name in os.listdir(reference_images_folder):
       reference_image_path = os.path.join(reference_images_folder, reference_image_name)
       reference_image_gray = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

       # Repeat the pyramid and DoG process for the reference image
       ref_gaussian_pyramid = create_gaussian_pyramid(reference_image_gray, num_octaves, num_scales)
       ref_dog_pyramid = difference_of_gaussian(ref_gaussian_pyramid)

       reference_keypoints = detect_keypoints(ref_dog_pyramid)
       reference_descriptors = compute_descriptors(reference_keypoints, reference_image_gray)

       if len(reference_descriptors) > 0 and len(query_descriptors) > 0:
           # Match features using FLANN matcher
           matches = flann.knnMatch(
               np.array(reference_descriptors, dtype=np.float32),
               np.array(query_descriptors, dtype=np.float32),
               k=2
           )


           # Apply Lowe's ratio test to filter matches
           good_matches = []
           for m, n in matches:
               if m.distance < 0.7 * n.distance:
                   good_matches.append(m)


           # If there are enough good matches, proceed
           if len(good_matches) >= min_match_count:
               print(f"Sufficient matches found with {reference_image_name}. Detecting object...")


               # Visualize keypoints and matches
               img_matches_resized = visualize_keypoint_matching(
                   reference_image_gray, query_image_gray,
                   reference_keypoints, query_keypoints, good_matches
               )


               # Localize the object using homography
               result_img, matches_mask = draw_object_detection(
                   reference_image_gray, query_image_gray,
                   reference_keypoints, query_keypoints, good_matches
               )


               # If this image has the best match so far, store it
               if len(good_matches) > best_matches_count:
                   best_matches_count = len(good_matches)
                   best_result_img = result_img  # Save result_img (with object detected)
                   best_reference_image = reference_image_name  # Save reference image name
                   best_reference_image_path = reference_image_path  # Save reference image path


               # Debugging output to see the match count and current best
               print(f"Current best match count: {best_matches_count} for {best_reference_image}")

   # If no valid match is found, return None or a default message
   if best_reference_image is None:
       print("No sufficient matches found in any of the reference images.")
       return None, None, None, None

   return best_reference_image, best_result_img, best_reference_image_path, img_matches_resized

# Define the folder containing reference images and the query image
reference_images_folder = r'C:\Users\April\PycharmProjects\PythonProject\reference_images'
query_image_path = r'../IMG_comsoc1.JPG'

# Find the best match
best_reference_image, best_result_img, best_reference_image_path, img_matches_resized = find_best_match(reference_images_folder, query_image_path)

# Display the best match result along with the keypoint matching visualization
print(f"Best match found with reference image: {best_reference_image}")

# Plot the results using subplots
plt.figure(figsize=(12, 6))

# Best Match Result
plt.subplot(1, 3, 1)
plt.imshow(best_result_img, cmap='gray')
plt.title('Best Match Result')
plt.axis('off')

# Closest Reference Image
reference_img_gray = cv2.imread(best_reference_image_path, cv2.IMREAD_GRAYSCALE)
plt.subplot(1, 3, 2)
plt.imshow(reference_img_gray, cmap='gray')
plt.title(f'Closest Reference: {best_reference_image}')
plt.axis('off')

# Keypoint Matching Visualization
plt.subplot(1, 3, 3)
plt.imshow(img_matches_resized)
plt.title('Keypoint Matching')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()