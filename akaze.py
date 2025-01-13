import cv2
from matplotlib import pyplot as plt

def extract_keypoints_and_descriptors(image, detector):
    """
    Detect keypoints and extract descriptors using A-KAZE.
    :param image: Grayscale image.
    :param detector: A-KAZE feature detector object.
    :return: keypoints, descriptors
    """
    print("Detecting keypoints and extracting descriptors...")
    keypoints, descriptors = detector.detectAndCompute(image, None)
    print(f"  - Detected {len(keypoints)} keypoints.")
    return keypoints, descriptors

def draw_keypoints(image, keypoints, title):
    """
    Draw keypoints on an image.
    :param image: Original image.
    :param keypoints: Keypoints to be drawn.
    :param title: Title of the image for display.
    """
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    plt.title(title)
    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis("off")

def main():
    # Load the reference and query images
    reference_image_path = r'reference_images/IMG_comsoc.JPG'
    query_image_path = 'IMG_comsoc1.JPG'

    print("Loading images...")
    reference_image = cv2.imread(reference_image_path)
    query_image = cv2.imread(query_image_path)

    # Convert images to grayscale
    print("Converting images to grayscale...")
    ref_gray_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    query_gray_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    # Initialize A-KAZE detector
    print("Initializing A-KAZE detector...")
    akaze_detector = cv2.AKAZE_create()

    # Detect keypoints and extract descriptors
    print("Processing reference image...")
    ref_keypoints, ref_descriptors = extract_keypoints_and_descriptors(ref_gray_image, akaze_detector)

    print("Processing query image...")
    query_keypoints, query_descriptors = extract_keypoints_and_descriptors(query_gray_image, akaze_detector)

    # Display results
    print("Displaying results...")
    plt.figure(figsize=(12, 6))

    # Reference image keypoints
    plt.subplot(1, 1, 1)
    draw_keypoints(reference_image, ref_keypoints, "Reference Image Keypoints")

    # # Query image keypoints
    # plt.subplot(1, 2, 2)
    # draw_keypoints(query_image, query_keypoints, "Query Image Keypoints")

    plt.tight_layout()
    plt.show()

    print("Process complete.")

if __name__ == "__main__":
    main()
