import json
import cv2
import numpy as np
import os
import glob
import time


def compute_sift_descriptors(image, keypoints, sigma_0=10):
    start_time = time.time()
    descriptors = []
    dominant_orientations = []

    for kp in keypoints:
        x, y, octave_idx, sublevel_idx, response = kp
        scale = sigma_0 * (2 ** octave_idx) * (2 ** (sublevel_idx / 4))
        region_size = int(2 * scale)
        half_size = region_size // 2
        region = image[max(0, y - half_size):min(image.shape[0], y + half_size),
                 max(0, x - half_size):min(image.shape[1], x + half_size)]
        if region.shape[0] == 0 or region.shape[1] == 0:
            continue

        grad_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_orientation = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 360

        hist = np.zeros((4, 4, 8))
        hist_orientation = np.zeros(8)

        for i in range(region.shape[0]):
            for j in range(region.shape[1]):
                sub_region_x = i * 4 // region.shape[0]
                sub_region_y = j * 4 // region.shape[1]
                bin_idx = int(grad_orientation[i, j] / 45) % 8
                hist[sub_region_x, sub_region_y, bin_idx] += grad_magnitude[i, j]
                hist_orientation[bin_idx] += grad_magnitude[i, j]

        descriptor = hist.flatten()
        descriptor /= np.linalg.norm(descriptor) + 1e-7

        descriptors.append(descriptor)
        dominant_orientations.append((np.argmax(hist_orientation)) * 45)

    computation_time = time.time() - start_time
    return np.array(descriptors), dominant_orientations, computation_time


def load_keypoints(json_path):
    with open(json_path, "r") as f:
        keypoints_data = json.load(f)
    keypoints = [
        (kp["x"], kp["y"], kp["octave"], kp["sublevel"], kp["response"])
        for kp in keypoints_data["keypoints"]
    ]
    return keypoints, keypoints_data["image"]


def save_descriptors(descriptors, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, descriptors)
    print(f"Descriptors saved to {output_path}")


if __name__ == "__main__":
    start_total = time.time()
    output_folder = "descriptors"
    os.makedirs(output_folder, exist_ok=True)
    performance_dir = os.path.join(output_folder, "performance")
    os.makedirs(performance_dir, exist_ok=True)
    performance_file = os.path.join(performance_dir, "performance.txt")

    keypoint_files = glob.glob("keypoints/keypoints_oct*_sub*_thresh*.json")
    with open(performance_file, "a") as perf_f:
        perf_f.write("\n===== Descriptor Computation Summary =====\n")

    for idx, json_path in enumerate(keypoint_files):
        keypoints, image_path = load_keypoints(json_path)
        output_file = os.path.join(output_folder, os.path.basename(json_path).replace('.json', '_descriptors.npy'))
        orientation_file = os.path.join(output_folder,
                                        os.path.basename(json_path).replace('.json', '_orientations.npy'))

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image {image_path} for {json_path}")
            continue

        print(f"\nProcessing {json_path}")
        print(f"Number of keypoints: {len(keypoints)}")

        sift_descriptors, orientations, time_taken = compute_sift_descriptors(image, keypoints)

        save_descriptors(sift_descriptors, output_file)
        np.save(orientation_file, orientations)

        with open(performance_file, "a") as perf_f:
            perf_f.write(f"{idx}) {json_path} -> {len(sift_descriptors)} descriptors | Time: {time_taken:.4f} s\n")

        print(f"Generated {len(sift_descriptors)} descriptors")
        print(f"Saved to: {output_file}")
        print(f"Orientations saved to: {orientation_file}")

    total_time = time.time() - start_total
    with open(performance_file, "a") as perf_f:
        perf_f.write(f"\n===== Total Descriptor Computation Time: {total_time:.4f} s =====\n")
    print(f"\n===== Total Computation Time: {total_time:.4f} s =====")

    data = np.load("descriptors/keypoints_oct3_sub4_thresh1_descriptors.npy")

    # Print or inspect the data
    print(data)
