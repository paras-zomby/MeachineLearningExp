import numpy as np


class KMeans(object):
    def __init__(self, num_cluster):
        super().__init__()
        self.num_cluster = num_cluster
        self.centers = [None] * num_cluster  # [None, None]

    def fit(self, x: np.ndarray):
        cluster_center = x[np.random.choice(range(len(x)), size=self.num_cluster, replace=False)]
        last_center = np.zeros_like(cluster_center)
        cluster = np.zeros(x.shape[0], dtype=int)
        while max([self.calc_dis(x, y) for x, y in zip(cluster_center, last_center)]) > 0.00001:
            last_center = cluster_center
            for idx, item in enumerate(x):
                min_idx = np.argmin([self.calc_dis(item, y) for y in cluster_center])
                cluster[idx] = min_idx
            for idx in range(self.num_cluster):
                cluster_center[idx] = np.mean(x[cluster == idx], axis=0)
        self.centers = cluster_center

    def predict(self, x: np.ndarray):
        if self.centers is None:
            raise Exception("Please fit model first")
        cluster = np.zeros(x.shape[0], dtype=int)
        for idx, item in enumerate(x):
            min_idx = np.argmin([self.calc_dis(item, y) for y in self.centers])
            cluster[idx] = min_idx
        return cluster

    @staticmethod
    def calc_dis(x: np.ndarray, y: np.ndarray):
        return np.power(x - y, 2).sum()
