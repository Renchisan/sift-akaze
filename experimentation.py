import cv2
import numpy as np
from itertools import product
import time
import json
from scipy.spatial import cKDTree

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
    if kappa is None:
        kappa = compute_contrast_factor(img)
    scale_space = {}
    for octave in range(n_octaves):
        octave_levels = []
        for sublevel in range(n_sublevels):
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

def non_maximum_suppression(keypoints, distance_threshold=5):
    if not keypoints:
        return []
    tree = cKDTree([(kp[0], kp[1]) for kp in keypoints])
    suppressed = []
    taken = set()
    for idx, kp in enumerate(sorted(keypoints, key=lambda k: -k[4])):
        if idx in taken:
            continue
        suppressed.append(kp)
        neighbors = tree.query_ball_point((kp[0], kp[1]), distance_threshold)
        taken.update(neighbors)
    return suppressed

def fast_hessian_response(image):
    int_img = cv2.integral(image).astype(np.float32)
    Hxx = cv2.Sobel(int_img, cv2.CV_32F, 2, 0, ksize=3)
    Hyy = cv2.Sobel(int_img, cv2.CV_32F, 0, 2, ksize=3)
    Hxy = cv2.Sobel(int_img, cv2.CV_32F, 1, 1, ksize=3)
    det = (Hxx * Hyy) - (Hxy ** 2)
    return cv2.normalize(det, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def detect_keypoints(scale_space, threshold_factor):
    keypoints = []
    for octave_idx, (octave_name, octave_images) in enumerate(scale_space.items()):
        for sublevel_idx, image in enumerate(octave_images):
            hessian_response = fast_hessian_response(image)
            threshold = hessian_response.mean() + threshold_factor * hessian_response.std()
            keypoints.extend([(x, y, octave_idx, sublevel_idx, hessian_response[y, x])
                              for y, x in np.argwhere(hessian_response > threshold)])
    return non_maximum_suppression(keypoints, distance_threshold=3) # Apply non-maximum suppression to the detected keypoints


def save_keypoints(image_path, keypoints):
    keypoints_data = []
    for kp in keypoints:
        keypoints_data.append(
            {"x": int(kp[0]), "y": int(kp[1]), "octave": int(kp[2]), "sublevel": int(kp[3]), "response": float(kp[4])})

    with open("keypoints.json", "a") as f:
        json.dump({"image": image_path, "keypoints": keypoints_data}, f, indent=4)
        f.write("\n")  # Appends newline for readability


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
        scale_space = nonlinear_diffusion_filter(image, n_octaves=n_octaves, n_sublevels=n_sublevels, sigma_0=10, time_step=0.03, kappa=None)
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

    print(f"\n===== Total Computation Time: {time.time() - start_total:.4f} s =====")

