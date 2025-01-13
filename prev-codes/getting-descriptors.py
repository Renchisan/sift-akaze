import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    """Load and preprocess an image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    return img_gray


def compute_keypoints_and_descriptors(image):
    """Compute keypoints and descriptors for an image using SIFT."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def categorize_descriptors(descriptors, num_bins=8):
    """
    Categorize descriptors into bins.

    Each descriptor value is mapped into one of `num_bins` bins.
    """
    binned_descriptors = []
    for descriptor in descriptors:
        bins = np.digitize(descriptor, np.linspace(0, 256, num_bins + 1)) - 1
        binned_descriptors.append(bins)
    return np.array(binned_descriptors)


def compare_descriptors(ref_binned, query_binned, threshold=5):
    """
    Match descriptors by comparing their binned values.

    Threshold defines the maximum acceptable number of bin mismatches.
    """
    matches = []
    for i, query_desc in enumerate(query_binned):
        for j, ref_desc in enumerate(ref_binned):
            # Count mismatches between bins
            mismatches = np.sum(query_desc != ref_desc)
            if mismatches <= threshold:
                matches.append((i, j))  # Match index from query to reference
    return matches


def draw_matches(ref_image, ref_keypoints, query_image, query_keypoints, matches):
    """
    Draw matches between keypoints in the reference and query images.
    """
    matched_image = cv2.drawMatches(
        query_image, query_keypoints,
        ref_image, ref_keypoints,
        [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0) for i, j in matches],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matched_image


# Main function
def main(reference_image_path, query_image_path, num_bins=8, mismatch_threshold=5):
    # Preprocess images
    ref_image = preprocess_image(reference_image_path)
    query_image = preprocess_image(query_image_path)

    # Compute keypoints and descriptors
    ref_keypoints, ref_descriptors = compute_keypoints_and_descriptors(ref_image)
    query_keypoints, query_descriptors = compute_keypoints_and_descriptors(query_image)

    # Categorize descriptors into bins
    ref_binned = categorize_descriptors(ref_descriptors, num_bins)
    query_binned = categorize_descriptors(query_descriptors, num_bins)

    # Compare descriptors and find matches
    matches = compare_descriptors(ref_binned, query_binned, mismatch_threshold)

    # Calculate accuracy
    accuracy = len(matches) / len(query_descriptors) if len(query_descriptors) > 0 else 0

    print(f"Number of Matches: {len(matches)}")
    print(f"Matching Accuracy: {accuracy * 100:.2f}%")

    # Draw matches
    matched_image = draw_matches(ref_image, ref_keypoints, query_image, query_keypoints, matches)

    # Display matches
    plt.figure(figsize=(15, 10))
    plt.title("Keypoint Matches")
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Provide the paths to your reference and query images
reference_image_path = r"../reference_images/IMG_comsoc.JPG"  # Replace with your path
query_image_path = r"../IMG_comsoc1.JPG"  # Replace with your path

main(reference_image_path, query_image_path)
