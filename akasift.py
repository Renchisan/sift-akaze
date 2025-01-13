import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_hessian_response(image):
    """Compute the Hessian matrix determinant response at each pixel."""
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
    """Extract keypoints from the Hessian response map."""
    keypoints = []
    threshold = hessian_response.mean() + threshold_factor * hessian_response.std()
    rows, cols = hessian_response.shape

    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            # Check if this pixel is a local maximum
            #
            # and above the threshold
            if (
                hessian_response[y, x] > threshold and
                hessian_response[y, x] > hessian_response[y-1, x] and
                hessian_response[y, x] > hessian_response[y+1, x] and
                hessian_response[y, x] > hessian_response[y, x-1] and
                hessian_response[y, x] > hessian_response[y, x+1]
            ):
                keypoints.append(cv2.KeyPoint(x, y, hessian_response[y, x]))
    return keypoints

# Load the image
image_path = "reference_images/IMG_csoffice.jpg"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Unable to load the image. Check the path.")
else:
    # Step 1: Compute the Hessian determinant response
    hessian_response = compute_hessian_response(image)

    # Step 2: Extract keypoints
    keypoints = extract_keypoints(hessian_response)

    # Visualize keypoints
    output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(output_image, cmap="gray")
    plt.title(f"Improved Keypoint Detection - Keypoints Detected: {len(keypoints)}")
    plt.axis("off")
    plt.show()

    # Save the output image
    # cv2.imwrite("improved_keypoints.jpg", output_image)
plt.close('all')  # Closes all matplotlib figures
cv2.destroyAllWindows()  # Closes all OpenCV windows
