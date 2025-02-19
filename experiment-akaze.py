import cv2
import numpy as np
from itertools import product
import time
import json
from concurrent.futures import ThreadPoolExecutor
import os


def compute_contrast_factor(img, num_bins=256):
    # Use integral image for faster gradient computation
    integral_img = cv2.integral(img.astype(np.float32))
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Vectorized histogram computation
    hmax = np.max(np.abs(grad_magnitude))
    histogram = np.histogram(grad_magnitude.ravel() / hmax, bins=num_bins, range=(0, 1))[0]
    k = (hmax * np.searchsorted(np.cumsum(histogram), 0.7 * np.sum(histogram))) / num_bins
    return k


def compute_divergence(grad_x, grad_y, diffusivity):
    # Compute divergence separately for x and y components
    diff_x = grad_x * diffusivity
    diff_y = grad_y * diffusivity

    # Calculate divergence using Sobel operators
    div_x = cv2.Sobel(diff_x, cv2.CV_32F, 1, 0, ksize=3)
    div_y = cv2.Sobel(diff_y, cv2.CV_32F, 0, 1, ksize=3)

    return div_x + div_y


def nonlinear_diffusion_filter(img, n_octaves, n_sublevels, sigma_0, time_step, kappa=None):
    img = img.astype(np.float32) / 255.0
    scale_space = {}

    if kappa is None:
        kappa = compute_contrast_factor(img)

    # Pre-compute kernel for Gaussian blur
    kernel_size = 5
    kernel = cv2.getGaussianKernel(kernel_size, sigma_0)
    kernel = kernel @ kernel.T

    for octave in range(n_octaves):
        octave_levels = []

        for sublevel in range(n_sublevels):
            # Compute gradients
            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            # Compute diffusivity
            diffusivity = np.exp(-(grad_magnitude / kappa) ** 2).astype(np.float32)

            # Compute divergence
            divergence = compute_divergence(grad_x, grad_y, diffusivity)

            # Update image
            img = np.clip(img + time_step * divergence, 0, 1)
            octave_levels.append(img.copy())

        scale_space[f'Octave {octave + 1}'] = octave_levels
        if octave < n_octaves - 1:
            img = cv2.pyrDown(img)

    return scale_space


def detect_keypoints(scale_space, threshold_factor):
    keypoints = []

    for octave_idx, (octave_name, octave_images) in enumerate(scale_space.items()):
        for sublevel_idx, image in enumerate(octave_images):
            # Compute Hessian matrix elements
            sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

            # Compute Hessian components
            hessian = {
                'xx': cv2.Sobel(sobel_x, cv2.CV_32F, 1, 0, ksize=3),
                'yy': cv2.Sobel(sobel_y, cv2.CV_32F, 0, 1, ksize=3),
                'xy': cv2.Sobel(sobel_x, cv2.CV_32F, 0, 1, ksize=3)
            }

            # Compute determinant without normalization
            determinant = hessian['xx'] * hessian['yy'] - hessian['xy'] ** 2

            # Use absolute values for thresholding
            abs_det = np.abs(determinant)
            threshold = abs_det.mean() + threshold_factor * abs_det.std()

            rows, cols = determinant.shape

            # Parallel processing of keypoint detection
            num_threads = 4
            row_chunks = np.array_split(range(1, rows - 1), num_threads)

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                batch_args = [(chunk, range(1, cols - 1), abs_det, threshold)
                              for chunk in row_chunks]
                results = list(executor.map(process_keypoint_batch, batch_args))

            for batch_keypoints in results:
                for x, y, response in batch_keypoints:
                    scaling_factor = 2 ** octave_idx
                    keypoints.append((
                        int(x * scaling_factor),
                        int(y * scaling_factor),
                        octave_idx,
                        sublevel_idx,
                        float(response)  # Ensure response is float
                    ))

    return non_maximum_suppression(keypoints, distance_threshold=10)


def process_keypoint_batch(args):
    y_range, x_range, hessian_response, threshold = args
    local_keypoints = []

    window_size = 3
    half_window = window_size // 2

    for y in y_range:
        for x in x_range:
            # Get the local window
            window = hessian_response[y - half_window:y + half_window + 1,
                     x - half_window:x + half_window + 1]
            center_value = hessian_response[y, x]

            # Check if center is maximum in local window and above threshold
            if center_value > threshold and center_value == window.max():
                local_keypoints.append((x, y, float(center_value)))

    return local_keypoints

def non_maximum_suppression(keypoints, distance_threshold):
    if not keypoints:
        return []

    # Convert to numpy array for faster operations
    keypoints_array = np.array(keypoints)
    suppressed_indices = []

    # Sort by response value
    sort_indices = np.argsort(keypoints_array[:, 4])[::-1]
    keypoints_array = keypoints_array[sort_indices]

    while len(keypoints_array) > 0:
        suppressed_indices.append(sort_indices[0])

        # Compute distances vectorized
        distances = np.sqrt(
            (keypoints_array[1:, 0] - keypoints_array[0, 0]) ** 2 +
            (keypoints_array[1:, 1] - keypoints_array[0, 1]) ** 2
        )

        # Keep points beyond threshold
        mask = distances > distance_threshold
        keypoints_array = keypoints_array[1:][mask]
        sort_indices = sort_indices[1:][mask]

    return [tuple(keypoints[i]) for i in suppressed_indices]


def save_keypoints(image_path, keypoints):
    keypoints_data = {
        "image": image_path,
        "keypoints": [
            {
                "x": int(kp[0]),
                "y": int(kp[1]),
                "octave": int(kp[2]),
                "sublevel": int(kp[3]),
                "response": float(kp[4])
            }
            for kp in keypoints
        ]
    }

    with open("keypoints.json", "w") as f:
        json.dump(keypoints_data, f, indent=4)


def process_scale_threshold_combination(args):
    """Process an image for a given scale-threshold combination."""
    image, image_name, n_octaves, n_sublevels, threshold = args
    start_time = time.time()

    # Compute keypoints (your existing function)
    scale_space = nonlinear_diffusion_filter(
        image, n_octaves=n_octaves, n_sublevels=n_sublevels,
        sigma_0=10, time_step=0.01, kappa=None
    )
    keypoints = detect_keypoints(scale_space, threshold_factor=threshold)

    return (image_name, n_octaves, n_sublevels, threshold, keypoints, time.time() - start_time)


if __name__ == "__main__":
    start_total = time.time()

    # Ensure "refs" folder exists
    image_folder = "refs"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.JPG'))]
    if not image_files:
        print("No images found in 'refs' folder.")
        exit()

    scales = [(3, 4), (4, 4), (4, 5), (5, 5)]
    thresholds = [1, 3, 5, 7, 10]  # Using the adjusted threshold values

    # Parallel processing for all images
    with ThreadPoolExecutor(max_workers=4) as executor:
        combinations = []
        for image_name in image_files:
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: Failed to load {image_name}. Skipping...")
                continue

            for (n_oct, n_sub), thresh in product(scales, thresholds):
                combinations.append((image, image_name, n_oct, n_sub, thresh))

        results = list(executor.map(process_scale_threshold_combination, combinations))

    print("\n===== Experiment Summary =====")

    # Ensure performance directory exists
    performance_dir = "descriptors/performance"
    os.makedirs(performance_dir, exist_ok=True)
    performance_file = os.path.join(performance_dir, "performance.txt")

    # Save results
    for idx, result in enumerate(results):
        image_name, n_octaves, n_sublevels, threshold, keypoints, time_taken = result

        # Create a subfolder for each image inside "keypoints"
        keypoints_folder = os.path.join("keypoints", os.path.splitext(image_name)[0])
        os.makedirs(keypoints_folder, exist_ok=True)

        # Create a filename based on the parameters
        filename = f"keypoints_oct{n_octaves}_sub{n_sublevels}_thresh{threshold}.json"
        filepath = os.path.join(keypoints_folder, filename)

        # Save keypoints to JSON
        keypoints_data = {
            "image": image_name,
            "parameters": {
                "octaves": n_octaves,
                "sublevels": n_sublevels,
                "threshold": threshold
            },
            "keypoints": [
                {
                    "x": int(kp[0]),
                    "y": int(kp[1]),
                    "octave": int(kp[2]),
                    "sublevel": int(kp[3]),
                    "response": float(kp[4])
                }
                for kp in keypoints
            ]
        }
        with open(filepath, "w") as f:
            json.dump(keypoints_data, f, indent=4)

        # Save performance data
        with open(performance_file, "a") as f:
            f.write(f"{idx}) {image_name} | Scale ({n_octaves}, {n_sublevels}) | "
                    f"Threshold {threshold} -> {len(keypoints)} keypoints | "
                    f"Time: {time_taken:.4f} s | Saved to: {filepath}\n")

    print(f"\n===== Total Computation Time: {time.time() - start_total:.4f} s =====")