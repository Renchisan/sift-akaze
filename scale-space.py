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
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)
    diffusivity = diffusivity.astype(np.float32)

    # divergence
    divergence_x = cv2.Sobel(grad_x * diffusivity, cv2.CV_32F, 1, 0, ksize=3)
    divergence_y = cv2.Sobel(grad_y * diffusivity, cv2.CV_32F, 0, 1, ksize=3)

    return divergence_x + divergence_y

def nonlinear_diffusion_filter(img, n_octaves=3, n_sublevels=4, sigma_0=10, time_step=0.03, kappa=None):
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    scale_space = []

    if kappa is None:
        kappa = compute_contrast_factor(img)

    for octave in range(n_octaves):
        octave_levels = []

        for sublevel in range(n_sublevels):
            sigma_i = sigma_0 * (2 ** (octave + sublevel / n_sublevels))
            time_t = 1 / (2 * sigma_i ** 2)

            grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

            #conduction function
            diffusivity = conduction_function(grad_magnitude, kappa)

            #divergence
            divergence = compute_divergence(grad_x, grad_y, diffusivity)

            #update the image using the diffusion process
            img += time_step * divergence
            img = np.clip(img, 0, 1)

            octave_levels.append(img.copy())

        scale_space.append(octave_levels)

        #downsample
        if octave < n_octaves - 1:
            img = cv2.pyrDown(img)

    return scale_space

# Example usage
if __name__ == "__main__":
    image_path = 'reference_images/IMG_comsoc.JPG'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # Generate the nonlinear scale-space using FED
        scale_space = nonlinear_diffusion_filter(image, n_octaves=3, n_sublevels=4, kappa=None)

        # Display the results grouped by octaves
        plt.figure(figsize=(15, 15))
        for octave_idx, octave_levels in enumerate(scale_space):
            for sublevel_idx, scale_image in enumerate(octave_levels):
                plt.subplot(len(scale_space), len(octave_levels), octave_idx * len(octave_levels) + sublevel_idx + 1)
                plt.title(f"Octave {octave_idx + 1} - Sublevel {sublevel_idx + 1}")
                plt.imshow(scale_image, cmap='gray')
                plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Error: Unable to load the image.")
