import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def identify_clusters(convergence_points, threshold=0.5):
    num_points = convergence_points.flatten()
    print(num_points.shape)
    labels = np.arange(num_points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if euclidean_distance(convergence_points[i], convergence_points[j]) <= threshold:
                labels[j] = labels[i]

    return labels

def mean_shift_segmentation(data, threshold=0.5):
    labels = identify_clusters(data, threshold)
    return labels