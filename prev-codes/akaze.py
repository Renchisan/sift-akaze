import cv2
import numpy as np
import os
import time  # For tracking runtime


def non_linear_diffusion(image, iterations=10, kappa=0.1):
    """Apply Perona-Malik non-linear diffusion for scale space construction."""
    print("Starting non-linear diffusion...")
    start_time = time.time()

    img = image.astype(np.float32) / 255.0
    for _ in range(iterations):
        grad_x = np.diff(img, axis=1)
        grad_y = np.diff(img, axis=0)

        c_x = np.exp(-(grad_x / kappa) ** 2)
        c_y = np.exp(-(grad_y / kappa) ** 2)

        diff_x = np.pad(c_x * grad_x, ((0, 0), (0, 1)), mode='constant')
        diff_y = np.pad(c_y * grad_y, ((0, 1), (0, 0)), mode='constant')
        img += diff_x + diff_y

    end_time = time.time()
    print(f"Non-linear diffusion completed in {end_time - start_time:.4f} seconds.")

    return (img * 255).astype(np.uint8)


def detect_keypoints(image, threshold=0.01):
    """Detect keypoints using Hessian determinant."""
    print("Detecting keypoints...")
    start_time = time.time()

    hessian = cv2.cornerHarris(image, blockSize=3, ksize=3, k=0.04)
    keypoints = np.argwhere(hessian > threshold * hessian.max())
    keypoints_list = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in keypoints]

    end_time = time.time()
    print(f"Keypoint detection completed in {end_time - start_time:.4f} seconds.")

    return keypoints_list


def compute_mldb_descriptors(image, keypoints, patch_size=31):
    """Compute binary descriptors for keypoints."""
    print("Computing MLDB descriptors...")
    start_time = time.time()

    descriptors = []
    half_patch = patch_size // 2
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = image[max(0, y - half_patch):y + half_patch + 1,
                max(0, x - half_patch):x + half_patch + 1]
        patch = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_LINEAR)
        descriptor = []
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                region = patch[i:i + 4, j:j + 4]
                mean_intensity = region.mean()
                descriptor.extend(region.flatten() > mean_intensity)
        descriptors.append(np.packbits(descriptor))

    end_time = time.time()
    print(f"Descriptor computation completed in {end_time - start_time:.4f} seconds.")

    return np.array(descriptors, dtype=np.uint8)


def match_descriptors(des1, des2):
    """Match binary descriptors using Hamming distance."""
    print("Matching descriptors...")
    start_time = time.time()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    end_time = time.time()
    print(f"Descriptor matching completed in {end_time - start_time:.4f} seconds.")

    return sorted(matches, key=lambda x: x.distance)


def visualize_keypoints(image, keypoints, window_name="Keypoints"):
    """Visualize detected keypoints."""
    print("Visualizing keypoints...")
    start_time = time.time()

    output = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
    resized_output = cv2.resize(output, (400, 300))  # Resize for smaller display
    cv2.imshow(window_name, resized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Keypoint visualization completed in {end_time - start_time:.4f} seconds.")


def visualize_matches(query_image, query_kps, ref_image, ref_kps, matches):
    """Visualize matches between query and reference images."""
    print("Visualizing matches...")
    start_time = time.time()

    # Filter out invalid matches
    valid_matches = [m for m in matches if m.queryIdx < len(query_kps) and m.trainIdx < len(ref_kps)]

    matched_img = cv2.drawMatches(query_image, query_kps, ref_image, ref_kps, valid_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    resized_matched_img = cv2.resize(matched_img, (800, 400))  # Resize for smaller display
    cv2.imshow("Matches", resized_matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    end_time = time.time()
    print(f"Match visualization completed in {end_time - start_time:.4f} seconds.")


def main():
    query_image_path = r'../IMG_csoffice1.jpg'
    reference_folder_path = r'../reference_images/'

    print("Loading query image...")
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    query_image = cv2.resize(query_image, (800, 600))  # Resize for smaller processing

    # Create non-linear scale space for the query image
    query_diffused = non_linear_diffusion(query_image)

    print("Detecting keypoints for query image...")
    query_keypoints = detect_keypoints(query_diffused)
    query_descriptors = compute_mldb_descriptors(query_diffused, query_keypoints)

    best_match_image = None
    best_match_count = 0
    best_matches = []

    # Loop through all images in the reference folder
    print(f"Processing reference images in folder: {reference_folder_path}")
    for ref_image_name in os.listdir(reference_folder_path):
        ref_image_path = os.path.join(reference_folder_path, ref_image_name)

        if not ref_image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        reference_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
        reference_image = cv2.resize(reference_image, (800, 600))

        reference_diffused = non_linear_diffusion(reference_image)

        print(f"Detecting keypoints for reference image: {ref_image_name}")
        reference_keypoints = detect_keypoints(reference_diffused)

        reference_descriptors = compute_mldb_descriptors(reference_diffused, reference_keypoints)

        matches = match_descriptors(query_descriptors, reference_descriptors)

        if len(matches) > best_match_count:
            best_match_count = len(matches)
            best_matches = matches
            best_match_image = reference_image
            best_match_name = ref_image_name

    # Visualize the best match result
    if best_match_image is not None:
        print(f"Best match found: {best_match_name} with {best_match_count} matches.")
        visualize_matches(query_diffused, query_keypoints, best_match_image, reference_keypoints, best_matches)
    else:
        print("No matches found.")


if __name__ == "__main__":
    main()
