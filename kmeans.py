import numpy as np


class KMeans(object):
    def __init__(self, num_cluster, threshold=1e-4):
        super().__init__()
        self.num_cluster = num_cluster
        self.threshold = threshold
        self.centers = [None] * num_cluster  # [None, None]

    def fit(self, x: np.ndarray):
        cluster_center = x[np.random.choice(range(len(x)), size=self.num_cluster, replace=False)].copy()
        last_center = np.zeros_like(cluster_center)
        cluster = np.zeros(x.shape[0], dtype=int)
        cnt = 0
        while max([self.calc_dis(x, y) for x, y in zip(cluster_center, last_center)]) > self.threshold:
            print(f"INFO: epoch = {cnt}, distance change = {max([self.calc_dis(x, y) for x, y in zip(cluster_center, last_center)]):9.8f}")
            cnt += 1
            last_center = cluster_center.copy()
            for idx, item in enumerate(x):
                min_idx = np.argmin([self.calc_dis(item, y) for y in cluster_center])
                cluster[idx] = min_idx
            for idx in range(self.num_cluster):
                cluster_center[idx] = np.mean(x[cluster == idx], axis=0)
        print(f"INFO: epoch = {cnt}, distance change = {max([self.calc_dis(x, y) for x, y in zip(cluster_center, last_center)]):9.8f} [FINAL]")
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
