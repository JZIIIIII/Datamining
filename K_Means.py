import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def K_Means(numbers):
    
    
    data_list_origin = []
    data_list_kmeans = []


    # 读取数据
    data = pd.read_csv('iris.csv')
    # 将二元属性转换为数值型数据
    data['Species'] = data['Species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})
    
    # 提取特征向量并分类标签
    X = data.drop('Species', axis=1)
    origin_label = data['Species']

    # 使用PCA进行降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    for i in range(len(X_pca)):
        data_dict ={tuple(X_pca[i]): origin_label[i] for i in range(len(X_pca))}
        data_list_origin.append(data_dict)

    

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=numbers)
    kmeans.fit(X_pca)

    # 打印聚类结果
    labels = kmeans.labels_

    for i in range(len(X_pca)):
        data_dict ={tuple(X_pca[i]): labels[i] for i in range(len(X_pca))}
        data_list_kmeans.append(data_dict)

    return data_list_origin,data_list_kmeans


    