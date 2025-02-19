import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_sift_descriptors(image, keypoints, sigma_0=10):

    descriptors = []
    dominant_orientations = []

    for kp in keypoints:
        x, y, octave_idx, sublevel_idx, response = kp
        # Compute the size of the keypoint region based on octave and sublevel
        scale = sigma_0 * (2 ** octave_idx) * (2 ** (sublevel_idx / 4))

        # Define the size of the region to extract around the keypoint
        region_size = int(2 * scale)  # Width and height
        half_size = region_size // 2

        #is region valid
        region = image[max(0, y - half_size):min(image.shape[0], y + half_size),
                 max(0, x - half_size):min(image.shape[1], x + half_size)]
        if region.shape[0] == 0 or region.shape[1] == 0:
            continue  # Skip if region is invalid

        #gradients in the region
        grad_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)

        #gradient magnitudes and orientations
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 360  # Convert to degrees

        #histograms for the 16 sub-regions, 8 bins, and bins for orientation
        hist = np.zeros((4, 4, 8))
        hist_orientation = np.zeros(8)
        # Loop over each pixel in the region
        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                # Determine the sub-region
                sub_region_x = i * 4 // region.shape[0]  # Row in the 4x4 grid
                sub_region_y = j * 4 // region.shape[1]  # Column in the 4x4 grid

                # Compute the bin for the orientation
                bin_idx = int(grad_orientation[i, j] / 45) % 8  # 8 bins (0-360 degrees)

                # Accumulate the weighted contribution to the histogram
                hist[sub_region_x, sub_region_y, bin_idx] += grad_magnitude[i, j]
                hist_orientation[bin_idx] += grad_magnitude[i, j]

        #flatten to single descriptor
        descriptor = hist.flatten()

        #normalize
        descriptor /= np.linalg.norm(descriptor) + 1e-7  #avoid division by zero
        descriptors.append(descriptor)
        dominant_orientations.append((np.argmax(hist_orientation)) * 45)

    # Return both descriptors and dominant orientations
    return np.array(descriptors), dominant_orientations


def load_keypoints(json_path):
    with open(json_path, "r") as f:
        keypoints_data = json.load(f)

    keypoints = [
        (kp["x"], kp["y"], kp["octave"], kp["sublevel"], kp["response"])
        for kp in keypoints_data["keypoints"]
    ]
    return keypoints, keypoints_data["image"]  # Return keypoints and image path

def save_descriptors(descriptors, output_path):
    np.save(output_path, descriptors)  # Save as a NumPy .npy file
    print(f"Descriptors saved to {output_path}")

if __name__ == "__main__":
    # Load keypoints from JSON
    json_path = "keypoints.json"  # Replace with your actual JSON file path
    keypoints, image_path = load_keypoints(json_path)

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sift_descriptors, orientations = compute_sift_descriptors(image, keypoints)

    output_file = "descriptors.npy"
    save_descriptors(sift_descriptors, output_file)