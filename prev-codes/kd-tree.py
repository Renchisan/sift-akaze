import numpy as np
from sklearn.neighbors import KDTree


# Generate synthetic 128D descriptors for testing
def generate_descriptors(num_descriptors, dimension=128, noise=0.1):
    base_descriptors = np.random.rand(num_descriptors, dimension)
    noisy_descriptors = base_descriptors + noise * np.random.randn(num_descriptors, dimension)
    return base_descriptors, noisy_descriptors


# Build KD-Tree and perform matching
def match_with_kd_tree(query_descriptors, target_descriptors, threshold=None):
    tree = KDTree(target_descriptors)
    distances, indices = tree.query(query_descriptors, k=1)
    matches = [(i[0], d) for i, d in zip(indices, distances) if threshold is None or d <= threshold]
    return matches


# Grouped KD-Tree matching
# Grouped KD-Tree matching with fixes
def match_with_grouped_kd_tree(query_descriptors, target_descriptors, bins=8, threshold=None):
    descriptor_dim = query_descriptors.shape[1]
    bin_size = descriptor_dim // bins
    grouped_matches = []

    # Grouping the descriptors into bins and finding the best match per bin
    for i in range(bins):
        start = i * bin_size
        end = start + bin_size
        tree = KDTree(target_descriptors[:, start:end])
        distances, indices = tree.query(query_descriptors[:, start:end], k=1)
        grouped_matches.append([(ind[0], dist) for ind, dist in zip(indices, distances)])

    # Aggregate grouped matches
    aggregated_matches = []
    for matches in zip(*grouped_matches):
        best_match = None
        best_distance = float('inf')
        for match in matches:
            if match[1] < best_distance:
                best_match = match
                best_distance = match[1]
        # Append only the index for evaluation, or None if no valid match is found
        aggregated_matches.append(
            best_match[0] if best_match and (threshold is None or best_distance <= threshold) else None)

    return aggregated_matches


# Evaluate matching accuracy
def evaluate(matches, ground_truth):
    # Ensure we avoid errors when encountering None values
    correct = sum(1 for m, gt in zip(matches, ground_truth) if m is not None and m == gt)
    accuracy = correct / len(ground_truth) * 100
    return accuracy


# Main
if __name__ == "__main__":
    # Generate synthetic descriptors
    num_descriptors = 1000
    descriptors1, descriptors2 = generate_descriptors(num_descriptors, noise=0.1)

    # Ground truth: descriptors1 corresponds directly to descriptors2
    ground_truth = np.arange(num_descriptors)

    # Define thresholds for confidence
    thresholds = [0.2, 0.5, 1.0, 2.0]

    # Test different thresholds
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")

        # Normal KD-Tree matching
        normal_matches = match_with_kd_tree(descriptors1, descriptors2, threshold=threshold)
        normal_accuracy = evaluate([m[0] for m in normal_matches], ground_truth)
        print(f"Normal KD-Tree: {normal_accuracy:.2f}%")

        # Optimized KD-Tree (same implementation for testing)
        optimized_matches = match_with_kd_tree(descriptors1, descriptors2, threshold=threshold)
        optimized_accuracy = evaluate([m[0] for m in optimized_matches], ground_truth)
        print(f"Optimized KD-Tree: {optimized_accuracy:.2f}%")

        # Proposed Grouped KD-Tree matching
        grouped_matches = match_with_grouped_kd_tree(descriptors1, descriptors2, bins=8, threshold=threshold)
        grouped_accuracy = evaluate(grouped_matches, ground_truth)
        print(f"Proposed Grouped KD-Tree: {grouped_accuracy:.2f}%")
