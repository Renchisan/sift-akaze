import cv2
import os
import numpy as np

# Paths
reference_images_dir = r'../reference_images'
query_image_path = r'../IMG_comsoc1.JPG'

# Load query image
query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
if query_image is None:
    print("Query image not found!")
    exit()

# Load reference images
reference_images = []
for file in os.listdir(reference_images_dir):
    file_path = os.path.join(reference_images_dir, file)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        reference_images.append((file, image))
    else:
        print(f"Failed to load {file}")

# Function to match features
def match_features(detector, query_img, ref_img):
    # Detect and compute keypoints and descriptors
    kp_query, des_query = detector.detectAndCompute(query_img, None)
    kp_ref, des_ref = detector.detectAndCompute(ref_img, None)

    # Match descriptors using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_query, des_ref)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance

    return kp_query, kp_ref, matches

# Function to create labeled image
def create_labeled_image(title, image, width=300):
    height, orig_width = image.shape[:2]
    scale = width / orig_width
    resized_image = cv2.resize(image, (width, int(height * scale)))

    # Create a white label with the same width as the resized image
    label = np.zeros((50, resized_image.shape[1], 3), dtype=np.uint8) + 255
    cv2.putText(label, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Stack the label and resized image
    return np.vstack((label, resized_image))

# Function to apply a detector and display results
def apply_and_display(query_image, reference_images, method_name, detector):
    print(f"Applying {method_name}...")
    labeled_images = []

    for file, ref_img in reference_images:
        kp_query, kp_ref, matches = match_features(detector, query_image, ref_img)

        # Draw matches
        matched_img = cv2.drawMatches(query_image, kp_query, ref_img, kp_ref, matches[:50], None)
        labeled_img = create_labeled_image(file, matched_img)
        labeled_images.append(labeled_img)

    # Display all matches in a single window
    grid = create_image_grid(labeled_images)
    cv2.imshow(f"{method_name} Matches", grid)
    cv2.waitKey(0)

# Function to create a grid of images
def create_image_grid(images, cols=2, padding=10):
    rows = (len(images) + cols - 1) // cols
    max_width = max(img.shape[1] for img in images)
    max_height = max(img.shape[0] for img in images)

    grid_height = rows * max_height + (rows - 1) * padding
    grid_width = cols * max_width + (cols - 1) * padding

    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8) + 255  # White background

    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        y, x = row * (max_height + padding), col * (max_width + padding)

        # Place the image in the grid
        grid[y:y + img.shape[0], x:x + img.shape[1]] = img

    return grid

# Run the processes
apply_and_display(query_image, reference_images, "ORB", cv2.ORB_create())
apply_and_display(query_image, reference_images, "AKAZE", cv2.AKAZE_create())
apply_and_display(query_image, reference_images, "Combined ORB & AKAZE", cv2.AKAZE_create())  # Example with AKAZE for now

cv2.destroyAllWindows()
