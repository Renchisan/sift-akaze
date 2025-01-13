import cv2
import numpy as np
import matplotlib.pyplot as plt

def nonlinear_diffusion_filter(img, filter_size, n_iter, time_step=0.03):
    img = img.astype(np.float32)
    sobel_x = get_sobel_x(filter_size)
    sobel_y = get_sobel_y(filter_size)

    kappa = 50  # Contrast threshold
    for i in range(n_iter):
        # Compute gradients in x and y directions
        grad_x = cv2.filter2D(src=img, ddepth=-1, kernel=sobel_x)
        grad_y = cv2.filter2D(src=img, ddepth=-1, kernel=sobel_y)

        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        diffusivity = 1 / (1 + (grad_magnitude / kappa) ** 2)

        # Update the image based on the divergence of the conductance
        img += time_step * (
                cv2.filter2D(grad_x * diffusivity, ddepth=-1, kernel=sobel_x) +
                cv2.filter2D(grad_y * diffusivity, ddepth=-1, kernel=sobel_y)
        )

        img = np.clip(img, 0, 255)  # Ensure pixel values are in [0, 255]

    return img.astype(np.uint8)

def get_sobel_x(size_n):
    custom_sobel = np.zeros((size_n, size_n), dtype=np.float32)
    center = size_n // 2
    for i in range(size_n):
        for j in range(size_n):
            if j != center:
                custom_sobel[i, j] = (j - center) / ((i - center) ** 2 + (j - center) ** 2)
    return custom_sobel

def get_sobel_y(size_n):
    custom_sobel = np.zeros((size_n, size_n), dtype=np.float32)
    center = size_n // 2
    for i in range(size_n):
        for j in range(size_n):
            if i != center:
                custom_sobel[i, j] = (i - center) / ((i - center) ** 2 + (j - center) ** 2)
    return -1 * custom_sobel

def compute_hessian_response(image):
    """
    Compute the Hessian matrix determinant response at each pixel.
    """
    # Smooth the image to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Compute second-order derivatives
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_xx = cv2.Sobel(sobel_x, cv2.CV_64F, 1, 0, ksize=3)
    sobel_yy = cv2.Sobel(sobel_y, cv2.CV_64F, 0, 1, ksize=3)
    sobel_xy = cv2.Sobel(sobel_x, cv2.CV_64F, 0, 1, ksize=3)

    # Compute determinant of the Hessian matrix
    determinant = sobel_xx * sobel_yy - sobel_xy ** 2

    # Normalize for better visualization
    determinant = cv2.normalize(determinant, None, 0, 255, cv2.NORM_MINMAX)
    return determinant.astype(np.uint8)

def extract_keypoints(hessian_response, threshold_factor=4):
    """
    Extract keypoints from the Hessian response map.
    """
    keypoints = []
    threshold = hessian_response.mean() + threshold_factor * hessian_response.std()
    rows, cols = hessian_response.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            # Check if this pixel is a local maximum and above the threshold
            if (
                hessian_response[y, x] > threshold and
                hessian_response[y, x] > hessian_response[y-1, x] and
                hessian_response[y, x] > hessian_response[y+1, x] and
                hessian_response[y, x] > hessian_response[y, x-1] and
                hessian_response[y, x] > hessian_response[y, x+1]
            ):
                keypoints.append(cv2.KeyPoint(x, y, hessian_response[y, x]))
    return keypoints

# Example usage
if __name__ == "__main__":
    # Load a sample image (make sure to provide the correct path)
    image_path = 'reference_images/IMG_csoffice.jpg'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # Apply nonlinear diffusion filter
        filtered_image = nonlinear_diffusion_filter(image, filter_size=3, n_iter=50, time_step=0.01)

        # Compute Hessian response
        hessian_response = compute_hessian_response(filtered_image)

        # Extract keypoints
        keypoints = extract_keypoints(hessian_response)

        # Output the number of keypoints
        print(f"Number of keypoints detected: {len(keypoints)}")

        # Draw keypoints on the filtered image
        output_image = cv2.drawKeypoints(filtered_image, keypoints, None, color=(0, 255, 0))

        # Display the results
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Filtered Image")
        plt.imshow(filtered_image, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Keypoints ({len(keypoints)} detected)")
        plt.imshow(output_image, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("Error: Unable to load the image.")
