import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_contrast_factor(img, num_bins=300):
    """
    Compute the contrast factor based on the gradient of the image.
    """
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

def conduction_function(gradient_magnitude, kappa):
    """
    Compute the conduction function based on the gradient magnitude.
    """
    return 1 / (1 + (gradient_magnitude / kappa) ** 2)

def compute_divergence(grad_x, grad_y, diffusivity):
    """
    Compute the divergence of the gradient field modulated by the diffusivity.
    """
    # Ensure grad_x and grad_y are of type np.float32
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)

    # Ensure diffusivity is the same shape as grad_x and grad_y
    diffusivity = diffusivity.astype(np.float32)  # Ensure type consistency

    # Compute the divergence in the x and y directions
    divergence_x = cv2.Sobel(grad_x * diffusivity, cv2.CV_32F, 1, 0, ksize=3)
    divergence_y = cv2.Sobel(grad_y * diffusivity, cv2.CV_32F, 0, 1, ksize=3)

    return divergence_x + divergence_y

def nonlinear_diffusion_filter(img, n_octaves=5, n_sublevels=4, sigma_0=10, time_step=0.03, kappa=None):
    """
    Construct the nonlinear scale-space using Fast Explicit Diffusion (FED).
    """
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    scale_space = {}

    if kappa is None:
        kappa = compute_contrast_factor(img)

    for octave in range(n_octaves):
        octave_levels = []

        for sublevel in range(n_sublevels):
            sigma_i = sigma_0 * (2 ** (octave + sublevel / n_sublevels))
            time_t = 1 / (2 * sigma_i ** 2)

            # Compute gradients after image update
            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            # Compute the conduction function
            diffusivity = conduction_function(grad_magnitude, kappa)
            divergence = compute_divergence(grad_x, grad_y, diffusivity)

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
    """
    Apply non-maximum suppression to the detected keypoints.
    """
    suppressed_keypoints = []
    keypoints = sorted(keypoints, key=lambda kp: kp[4], reverse=True)  # Sort by response value

    while keypoints:
        # Take the keypoint with the highest response
        current_kp = keypoints.pop(0)
        suppressed_keypoints.append(current_kp)

        # Remove keypoints that are too close to the current keypoint
        keypoints = [
            kp for kp in keypoints
            if np.sqrt((kp[0] - current_kp[0]) ** 2 + (kp[1] - current_kp[1]) ** 2) > distance_threshold
        ]

    return suppressed_keypoints

def detect_keypoints(scale_space, threshold_factor):
    """
    Detect keypoints from the constructed scale space.
    """
    keypoints = []

    for octave_idx, (octave_name, octave_images) in enumerate(scale_space.items()):
        for sublevel_idx, image in enumerate(octave_images):
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected an image of type np.ndarray, got {type(image)}.")

            # Ensure the image is floating-point
            image = image.astype(np.float32)

            # Compute second-order derivatives
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_xx = cv2.Sobel(sobel_x, cv2.CV_64F, 1, 0, ksize=3)
            sobel_yy = cv2.Sobel(sobel_y, cv2.CV_64F, 0, 1, ksize=3)
            sobel_xy = cv2.Sobel(sobel_x, cv2.CV_64F, 0, 1, ksize=3)

            # Compute determinant of Hessian matrix
            determinant = sobel_xx * sobel_yy - sobel_xy ** 2

            # Normalize determinant for visualization
            determinant = cv2.normalize(determinant, None, 0, 255, cv2.NORM_MINMAX)
            hessian_response = determinant.astype(np.uint8)
            threshold = hessian_response.mean() + threshold_factor * hessian_response.std()
            rows, cols = hessian_response.shape

            for y in range(1, rows - 1):
                for x in range(1, cols - 1):
                    # Check if this pixel is a local maximum
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
                        # Append custom tuple with extra info
                        keypoints.append((adjusted_x, adjusted_y, octave_idx, sublevel_idx, hessian_response[y, x]))

        # Apply non-maximum suppression to the detected keypoints
    return non_maximum_suppression(keypoints, distance_threshold=5)


def compute_sift_descriptors(image, keypoints, sigma_0=10):
    """
    Compute SIFT descriptors for the given keypoints in the image.
    """
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

if __name__ == "__main__":
    image_path = 'reference_images/IMG_comsoc.JPG'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # After generating the nonlinear scale-space
    scale_space = nonlinear_diffusion_filter(image, n_octaves=3, n_sublevels=6)

    # Detect keypoints
    keypoints = detect_keypoints(scale_space, threshold_factor=7)

    # Compute SIFT descriptors
    sift_descriptors, orientations = compute_sift_descriptors(image, keypoints)

    # Print the descriptors
    for i, desc in enumerate(sift_descriptors):
        print(f"Descriptor for Keypoint {i}: {desc}")

    # Plot the keypoints with orientation on the original image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization

    sigma_0 = 10
    cv_keypoints = []
    for kp, orientation in zip(keypoints, orientations):
        x, y, octave_idx, sublevel_idx, response = kp
        size = sigma_0 * (2 ** octave_idx) * (2 ** (sublevel_idx / 4))
        angle = float(orientation)  # Ensure the orientation is treated as a float

        cv_keypoints.append(cv2.KeyPoint(x, y, size, angle))

    # Draw the keypoints on the image using drawKeypoints
    img_with_keypoints = cv2.drawKeypoints(
        image_color, cv_keypoints, None, color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )

    # Display the image with keypoints and orientations
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axes
    plt.show()