import cv2
import numpy as np
from itertools import product
import time
import json

def compute_contrast_factor(img, num_bins=256):
    #Compute the contrast factor based on the gradient of the image.
    smoothed_img = cv2.GaussianBlur(img, (5, 5), sigmaX=1)
    grad_x = cv2.Sobel(smoothed_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed_img, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    hmax = np.max(np.abs(grad_magnitude))
    # Normalize gradient magnitudes
    histogram, _ = np.histogram(grad_magnitude.flatten() / hmax, bins=num_bins, range=(0, 1))
    cumulative_hist = np.cumsum(histogram)
    percentile_index = np.searchsorted(cumulative_hist, 0.7 * cumulative_hist[-1])
    k = (hmax * percentile_index) / num_bins
    return k

def compute_divergence(grad_x, grad_y, diffusivity):
    # Compute the divergence in the x and y directions
    divergence_x = cv2.Sobel(grad_x * diffusivity, cv2.CV_32F, 1, 0, ksize=3)
    divergence_y = cv2.Sobel(grad_y * diffusivity, cv2.CV_32F, 0, 1, ksize=3)
    return divergence_x + divergence_y

def nonlinear_diffusion_filter(img, n_octaves, n_sublevels, sigma_0, time_step, kappa=None):
    #Construct the nonlinear scale-space using Fast Explicit Diffusion (FED).
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    scale_space = {}
    if kappa is None:
        kappa = compute_contrast_factor(img)
    for octave in range(n_octaves):
        octave_levels = []
        for sublevel in range(n_sublevels):
            sigma_i = sigma_0 * (2 ** (octave + sublevel / n_sublevels))
            # Compute gradients after image update
            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3).astype(np.float32)
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3).astype(np.float32)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            # Compute the CONDUCTION FUNCTION
            diffusivity = (np.exp(- (grad_magnitude / kappa) ** 2)).astype(np.float32)
            divergence = cv2.Laplacian(grad_x * diffusivity + grad_y * diffusivity, cv2.CV_32F)
            # Update the image using the diffusion process
            img += time_step * divergence
            img = np.clip(img, 0, 1)
            octave_levels.append(img.copy())
        scale_space[f'Octave {octave + 1}'] = octave_levels
        # Downsample the image for the next octave
        if octave < n_octaves - 1:
            img = cv2.pyrDown(img)
    return scale_space

def non_maximum_suppression(keypoints, distance_threshold):
    suppressed_keypoints = []
    keypoints = sorted(keypoints, key=lambda kp: kp[4], reverse=True)  # Sort by response value
    while keypoints:
        #take the keypoint with the highest response
        current_kp = keypoints.pop(0)
        suppressed_keypoints.append(current_kp)
        #remove keypoints too close to the current keypoint
        keypoints = [
            kp for kp in keypoints
            if np.sqrt((kp[0] - current_kp[0]) ** 2 + (kp[1] - current_kp[1]) ** 2) > distance_threshold
        ]
    return suppressed_keypoints

def detect_keypoints(scale_space, threshold_factor):
    keypoints = []
    for octave_idx, (octave_name, octave_images) in enumerate(scale_space.items()):
        for sublevel_idx, image in enumerate(octave_images):
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected an image of type np.ndarray, got {type(image)}.")
            image = image.astype(np.float32)
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) #second-order derivatives
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_xx = cv2.Sobel(sobel_x, cv2.CV_64F, 1, 0, ksize=3)
            sobel_yy = cv2.Sobel(sobel_y, cv2.CV_64F, 0, 1, ksize=3)
            sobel_xy = cv2.Sobel(sobel_x, cv2.CV_64F, 0, 1, ksize=3)
            determinant = sobel_xx * sobel_yy - sobel_xy ** 2 #determinant of Hessian matrix
            determinant = cv2.normalize(determinant, None, 0, 255, cv2.NORM_MINMAX) # Normalize determinant for visualization
            hessian_response = determinant.astype(np.uint8)
            threshold = hessian_response.mean() + threshold_factor * hessian_response.std()
            rows, cols = hessian_response.shape

            for y in range(1, rows - 1):
                for x in range(1, cols - 1):
                    if (
                        hessian_response[y, x] > threshold
                        and hessian_response[y, x] > hessian_response[y - 1, x]
                        and hessian_response[y, x] > hessian_response[y + 1, x]
                        and hessian_response[y, x] > hessian_response[y, x - 1]
                        and hessian_response[y, x] > hessian_response[y, x + 1]
                    ):
                        scaling_factor = 2 ** octave_idx  # Scaling factor for each octave
                        adjusted_x = int(x * scaling_factor)
                        adjusted_y = int(y * scaling_factor)
                        keypoints.append((adjusted_x, adjusted_y, octave_idx, sublevel_idx, hessian_response[y, x])) # Append custom tuple with extra info
    return non_maximum_suppression(keypoints, distance_threshold=10) # Apply non-maximum suppression to the detected keypoints


def save_keypoints(image_path, keypoints):
    keypoints_data = {
        "image": image_path,
        "keypoints": [
            {
                "x": int(kp[0]),
                "y": int(kp[1]),
                "octave": int(kp[2]),
                "sublevel": int(kp[3]),
                "response": float(kp[4])  # Convert float32 to Python float
            }
            for kp in keypoints
        ]
    }

    with open("keypoints.json", "w") as f:
        json.dump(keypoints_data, f, indent=4)

if __name__ == "__main__":
    start_total = time.time()
    image_path = 'reference_images/IMG_comsoc.JPG'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    scales = [(3, 4), (3, 5), (4, 4), (4, 5), (5, 4), (5, 5)]
    thresholds = [3, 5, 7, 10]
    results = []
    print("\n===== Running Experiments for All Combinations =====")
    # Generate all (scale, threshold) combinations
    for (n_octaves, n_sublevels), threshold in product(scales, thresholds):
        start_time = time.time()  # Start timing this combination
        # Step 1: Generate scale space
        scale_space = nonlinear_diffusion_filter(image, n_octaves=n_octaves, n_sublevels=n_sublevels, sigma_0=10, time_step=0.03, kappa=None)
        # Step 2: Detect keypoints
        keypoints = detect_keypoints(scale_space, threshold_factor=threshold)
        save_keypoints(image_path, keypoints)

        end_time = time.time()  # Stop timing this combination

        computation_time = end_time - start_time
        results.append((n_octaves, n_sublevels, threshold, len(keypoints), computation_time))

        print(f"Scale ({n_octaves}, {n_sublevels}) | Threshold {threshold} -> {len(keypoints)} keypoints | Time: {computation_time:.4f} s")

    print("\n===== Experiment Summary =====")
    for idx, (n_octaves, n_sublevels, threshold, keypoint_count, time_taken) in enumerate(results):
        print(
            f"{idx}) Scale ({n_octaves}, {n_sublevels}) | Threshold {threshold} -> {keypoint_count} keypoints | Time: {time_taken:.4f} s")

    end_total = time.time()
    print(f"\n===== Total Computation Time: {end_total - start_total:.4f} s =====")

