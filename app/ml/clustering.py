import numpy as np

class KMeansScratch:
    def __init__(self, k=3, init='kmeans++', max_iter=300, tol=1e-4):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.history = [] # Tracks inertia over iterations

    def _init_centroids(self, X):
        n_samples, n_features = X.shape
        if self.init == 'random':
            indices = np.random.choice(n_samples, self.k, replace=False)
            return X[indices]
        else:
            # K-means++ initialization
            centroids = [X[np.random.choice(n_samples)]]
            for _ in range(1, self.k):
                # Distance of each point to the nearest centroid already chosen
                dists = np.array([min(np.sum((x - c) ** 2) for c in centroids) for x in X])
                # Probabilities proportional to squared distance
                probs = dists / np.sum(dists)
                next_centroid_idx = np.random.choice(n_samples, p=probs)
                centroids.append(X[next_centroid_idx])
            return np.array(centroids)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.centroids = self._init_centroids(X)
        self.history = []

        for i in range(self.max_iter):
            # 1. Assignment Step
            # Compute distances to all centroids
            dists = np.zeros((n_samples, self.k))
            for j in range(self.k):
                dists[:, j] = np.sum((X - self.centroids[j]) ** 2, axis=1)
            
            labels = np.argmin(dists, axis=1)
            
            # Compute current inertia (sum of squared distances)
            inertia = float(np.sum(np.min(dists, axis=1)))
            self.history.append(inertia)
            
            # 2. Update Step
            new_centroids = np.zeros((self.k, n_features))
            for j in range(self.k):
                points = X[labels == j]
                if len(points) > 0:
                    new_centroids[j] = np.mean(points, axis=0)
                else:
                    # If cluster is empty, keep old centroid or pick random point
                    new_centroids[j] = self.centroids[j]
            
            # Check for convergence
            center_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            self.labels = labels
            
            if center_shift < self.tol:
                break
                
        self.inertia_ = self.history[-1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        dists = np.zeros((n_samples, self.k))
        for j in range(self.k):
            dists[:, j] = np.sum((X - self.centroids[j]) ** 2, axis=1)
        return np.argmin(dists, axis=1)


class HierarchicalClusteringScratch:
    """
    Agglomerative Hierarchical Clustering from scratch.
    Fits and assigns points to k clusters.
    """
    def __init__(self, k=3, linkage='average'):
        self.k = k
        self.linkage = linkage
        self.labels = None
        self.history = [] # Stores merge history (merge distance at each step)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        
        # Start with each point as its own cluster
        # Map: cluster_id -> list of sample indices
        clusters = {i: [i] for i in range(n_samples)}
        
        # Compute initial distance matrix between samples
        # Map: (cluster1, cluster2) -> distance
        # We can optimize this by keeping distance between clusters
        # but for clean, standard from-scratch code:
        def cluster_dist(c1_indices, c2_indices):
            # Compute distance between two clusters based on linkage
            c1_pts = X[c1_indices]
            c2_pts = X[c2_indices]
            
            # Compute all pairwise squared Euclidean distances
            dists = np.sqrt(np.sum((c1_pts[:, np.newaxis, :] - c2_pts[np.newaxis, :, :]) ** 2, axis=-1))
            
            if self.linkage == 'single':
                return np.min(dists)
            elif self.linkage == 'complete':
                return np.max(dists)
            else: # average
                return np.mean(dists)

        self.history = []
        
        # Merge until we have exactly self.k clusters
        while len(clusters) > self.k:
            cluster_keys = list(clusters.keys())
            min_dist = float('inf')
            to_merge = (None, None)
            
            # Find the two closest clusters
            for i in range(len(cluster_keys)):
                for j in range(i + 1, len(cluster_keys)):
                    k1 = cluster_keys[i]
                    k2 = cluster_keys[j]
                    dist = cluster_dist(clusters[k1], clusters[k2])
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (k1, k2)
            
            k1, k2 = to_merge
            if k1 is None:
                break
                
            # Merge cluster k2 into k1
            clusters[k1].extend(clusters[k2])
            del clusters[k2]
            
            # Save distance to history (useful for training visualization)
            self.history.append(min_dist)
            
        # Assign labels
        labels = np.zeros(n_samples, dtype=int)
        for label_idx, (cluster_key, indices) in enumerate(clusters.items()):
            labels[indices] = label_idx
            
        self.labels = labels
        return self

    def predict(self, X):
        # In hierarchical clustering, predictions on new data are usually done by assigning
        # to the nearest cluster representative (e.g. centroid).
        # We will compute centroids of final clusters and assign X to the nearest.
        # This is the standard extension of Agglomerative clustering to out-of-sample prediction.
        raise NotImplementedError("Prediction on new data is not directly supported in Agglomerative Clustering. Use fit_predict instead.")
