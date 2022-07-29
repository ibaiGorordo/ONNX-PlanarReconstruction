import numpy as np
import cv2

class PlaneCluster:
    def __init__(self, init_cluster_num=10, max_cluster_distance=70):
        self.init_cluster_num = init_cluster_num
        self.max_cluster_distance = max_cluster_distance

    def __call__(self, embedding):
        return self.cluster(embedding)

    def cluster(self, embedding):
        emb_feature = self.get_emb_feature(embedding)
        clusters, center = self.cluster_emb_feature(emb_feature)
        cluster_centroids = self.get_cluster_cetroids(clusters)

        cluster_data = np.hstack((cluster_centroids, center))
        cluster_labels = self.combine_clusters(cluster_data)
        for i in range(cluster_labels.shape[0]):
            clusters[clusters == i] = cluster_labels[i]

        return clusters

    def get_emb_feature(self, embedding):
        emb_norm = self.normalize_embedding(embedding)
        emb_mul = (np.multiply(emb_norm[0, :, :].astype(np.float32),
                               emb_norm[1, :, :].astype(np.float32)) / 255).astype(np.uint8)
        emb_comb = np.concatenate((np.expand_dims(emb_norm[0, :, :], axis=2),
                                   np.expand_dims(emb_norm[1, :, :], axis=2),
                                   np.expand_dims(emb_mul, axis=2)), axis=2)
        return emb_comb

    def normalize_embedding(self, embedding):
        max_embedding = np.amax(embedding, axis=(1, 2), keepdims=True)
        min_embedding = np.amin(embedding, axis=(1, 2), keepdims=True)
        return ((embedding - min_embedding) / (max_embedding - min_embedding) * 255).astype(np.uint8)

    def cluster_emb_feature(self, emb_feature):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(emb_feature.astype(np.float32).reshape((-1, 3)),
                                        self.init_cluster_num, None, criteria, 10,
                                        cv2.KMEANS_RANDOM_CENTERS)
        label = label.reshape(emb_feature.shape[:2])
        return label, center

    def get_cluster_cetroids(self, label):
        num_clusters = label.max() + 1
        cluster_centroids = np.zeros((num_clusters, 2))
        for i in range(num_clusters):
            rows, cols = np.where(label == i)
            cluster_centroids[i, :] = np.mean(np.stack((rows, cols), axis=1), axis=0)

        return cluster_centroids.astype(int)

    def combine_clusters(self, cluster_data):

        distances = np.zeros((cluster_data.shape[0], cluster_data.shape[0]))
        for i in range(cluster_data.shape[0]):
            for j in range(cluster_data.shape[0]):
                if i == j:
                    distances[i, j] = np.inf
                    continue
                distances[i, j] = np.linalg.norm(cluster_data[i, :] - cluster_data[j, :])

        # Find the closest clusters
        min_distances = np.amin(distances, axis=1)
        closest_clusters = np.argmin(distances, axis=1)
        closest_clusters[min_distances > self.max_cluster_distance] = \
            np.arange(cluster_data.shape[0])[min_distances > self.max_cluster_distance]

        return closest_clusters
