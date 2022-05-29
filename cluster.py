# -*- coding: utf-8 -*-
import os

import numpy as np
# from sklearn.cluster import KMeans
from kmeans import KMeans
import cv2 as cv

if __name__ == '__main__':
    # 聚类数2,6,30
    list_info = os.walk(r'dataset/cloud').__next__()
    pics = (os.path.join(list_info[0], x) for x in list_info[2])
    MAX_SIZE = 600000
    # cluster = KMeans(n_clusters=2, init='k-means++')
    cluster = KMeans(num_cluster=2)
    # cluster = DBSCAN(eps=100)

    for img_path in pics:
        img = cv.imread(img_path)
        if np.prod(img.shape[:2]) > MAX_SIZE:
            ratio = MAX_SIZE / np.prod(img.shape[:2])
            img = cv.resize(img, None, None, ratio, ratio)

        img_f = (img.astype(np.float32) / 255).reshape((-1, 3))
        cluster.fit(img_f[np.random.randint(0, img_f.shape[0], size=int(img_f.shape[0] * 0.7))])
        result = cluster.predict(img_f)
        mask = result.reshape(img.shape[:2]).astype(bool)
        print(f"sum class 1 = {mask.sum()}, sum class 2 = {(~mask).sum()}")
        # img[mask] = np.array([0, 0, 255])
        cv.imshow("raw img", img)
        img[mask] = cv.addWeighted(img[mask], 0.7, np.zeros_like(img[mask]) + np.array([0, 0, 255]), 0.3, 0, dtype=cv.CV_8UC3)
        img[~mask] = cv.addWeighted(img[~mask], 0.7, np.zeros_like(img[~mask]) + np.array([0, 255, 0]), 0.3, 0, dtype=cv.CV_8UC3)
        cv.imshow("cls img", img)
        cv.waitKey(0)
