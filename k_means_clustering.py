import numpy as np

def euclidean_distance(x1 : np.ndarray, x2 : np.ndarray) -> float:
    return np.sqrt(((x1 - x2) ** 2).sum(axis=1))

def k_means_clustering(points : list[tuple[float , float]] , k : int , 
                       initial_centroids : list[tuple[float , float]] ,
                       max_iterations : int ) -> list[tuple[float , float]]:
    """
    This function performs the k-means clustering algorithm.
    Args:
        points : list[tuple[float , float]] : The data points.
        k : int : The number of clusters.
        initial_centroids : list[tuple[float , float]] : The initial centroids.
        max_iterations : int : The maximum number of iterations.
    Returns:
        list[tuple[float , float]] : The final centroids.
    """


    points : np.ndarray = np.array(points)
    centroids : np.ndarray = np.array(initial_centroids)

    for _ in range(max_iterations):
        distances : np.ndarray = np.array([euclidean_distance(points , centroid) for centroid in centroids])

        assignment : np.ndarray = np.argmin(distances , axis = 0)

        new_centroids : np.ndarray = np.array([points[assignment == i].mean(axis = 0) if len(points[assignment == i]) != 0 else centroids[i] for i in range(k)])

        if np.all(centroids == new_centroids):  
            break
        centroids = new_centroids
        centroids = np.round(centroids , 4)

    return [tuple (centroid) for centroid in centroids] 
