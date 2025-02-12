import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_contrast_factor(img, num_bins=300):
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
    print("compute_contrast_factor")
    return k

def conduction_function(gradient_magnitude, kappa):
    #Compute the conduction function based on the gradient magnitude.
    print("conduction function")
    return 1 / (1 + (gradient_magnitude / kappa) ** 2)

def compute_divergence(grad_x, grad_y, diffusivity):
    #Compute the divergence of the gradient field modulated by the diffusivity.
    #Ensure grad_x and grad_y are of type np.float32
    grad_x = grad_x.astype(np.float32)
    grad_y = grad_y.astype(np.float32)

    # Ensure diffusivity is the same shape as grad_x and grad_y
    diffusivity = diffusivity.astype(np.float32)  # Ensure type consistency

    # Compute the divergence in the x and y directions
    divergence_x = cv2.Sobel(grad_x * diffusivity, cv2.CV_32F, 1, 0, ksize=3)
    divergence_y = cv2.Sobel(grad_y * diffusivity, cv2.CV_32F, 0, 1, ksize=3)
    print("compute divergence")
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
    print("nonlinear diffusion filter")
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
    print("nonmaximum suppresion")
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
    print("detect keypoints")
    return non_maximum_suppression(keypoints, distance_threshold=10) # Apply non-maximum suppression to the detected keypoints

if __name__ == "__main__":
    image_path = 'reference_images/IMG_comsoc.JPG'  # Replace with your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    scalespace_experiment = []
    for n in range(3, 7, 1):
        scale_space  = nonlinear_diffusion_filter(image, n_octaves=n, n_sublevels=n+1, sigma_0=10, time_step=0.03, kappa=None)
        scalespace_experiment.append(scale_space)

    #scale_space = nonlinear_diffusion_filter(image, n_octaves=3, n_sublevels=4,  sigma_0=10, time_step=0.03, kappa=None)
    threshold_experiment = []
    for ss in scalespace_experiment:
        for n in range(7,8,2):
            keypoints = detect_keypoints(ss, threshold_factor=n )
            threshold_experiment.append(keypoints)
            print("appended!",  n)

    for index, keypoints in enumerate(threshold_experiment):
        #keypoints = detect_keypoints(scale_space, threshold_factor=7)
        print("This in #", index)
        #for kp in keypoints:
            #print(f"Keypoint at (x={kp[0]}, y={kp[1]}, octave={kp[2]}, sublevel={kp[3]}, response={kp[4]})")
        print(f"Detected Keypoints: {len(keypoints)}")

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    sigma_0=10
    for index, keypoints in enumerate(threshold_experiment):
        cv_keypoints = [
            cv2.KeyPoint(x=kp[0], y=kp[1], size=sigma_0 * (2 ** kp[2]) * (2 ** (kp[3] / 4)))
            for kp in keypoints
        ]

        # Draw the keypoints on the image using drawKeypoints
        img_with_keypoints = cv2.drawKeypoints(image_color, cv_keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

        # Display the image with keypoints
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide the axes
        plt.show()
