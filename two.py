from turtle import pd

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MeanShift
import matplotlib.pyplot as plt
from sklearn.metrics import v_measure_score





# a)使用sklearn.datasets模块的load_iris()函数载入Iris数据集。尝试对Iris数据集进行基本的描述（如样本数，属性，类别等）。
def load_iris_data():
    iris = load_iris()

    # b)检查数据完整性，将4维特征使用箱图进行可视化，并添加列名
    plt.boxplot(iris.data)
    plt.xticks([1, 2, 3, 4], iris.feature_names)
    plt.show()

    # c)使用特征提取的PCA方法，将数据维度将至2维，用散点图进行可视化。并对对应的颜色添加标注
    # 进行PCA降维
    # 对数据进行PCA降维
    # 对数据进行PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(iris.data)

    # 绘制2D散点图，根据不同类别设置不同颜色和标签
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # 添加颜色标注
    target_names = iris.target_names
    handles = []
    labels = []
    for target_value, target_name in enumerate(target_names):
        color = plt.cm.viridis(target_value / (len(target_names) - 1))
        handles.append(plt.Line2D([0], [0], linestyle='none', marker='o', mec='k', mfc=color))
        labels.append(target_name)
    plt.legend(handles, labels, loc='best')

    plt.show()

    # d)在Iris数据集降维之后的基础上，实现K-means、AgglomerativeClustering、Birch、DBSCAN、MeanShift方法，进行聚类，并且将聚类结果用散点图可视化显示。
    # 创建K-means聚类器（k=3）
    kmeans = KMeans(n_clusters=3, random_state=42)

    # 创建AgglomerativeClustering聚类器
    agg = AgglomerativeClustering(n_clusters=3)

    # 创建Birch聚类器
    birch = Birch(n_clusters=3)

    # 创建DBSCAN聚类器
    dbscan = DBSCAN(eps=0.5, min_samples=5)

    # 创建MeanShift聚类器
    meanshift = MeanShift()

    # 对数据进行聚类
    labels_kmeans = kmeans.fit_predict(X_pca)
    labels_agg = agg.fit_predict(X_pca)
    labels_birch = birch.fit_predict(X_pca)
    labels_dbscan = dbscan.fit_predict(X_pca)
    labels_meanshift = meanshift.fit_predict(X_pca)

    # 可视化聚类结果
    plt.figure(figsize=(12, 10))

    plt.subplot(321)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis')
    plt.title("K-means")



    plt.show()








if __name__ == '__main__':

    load_iris_data()
