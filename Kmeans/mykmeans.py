import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# 生成带高斯噪声的空间椭圆
def generate_elliptical_data(center, axes, angle, num_points):
    t = np.linspace(0, 2 * np.pi, num_points)
    Elliptical_data = np.column_stack([axes[0] * np.cos(t), axes[1] * np.sin(t)])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    Elliptical_data = np.dot(Elliptical_data, rotation_matrix) + center
    return Elliptical_data + np.random.normal(scale=0.1, size=Elliptical_data.shape)

# 生成四个椭圆
num_points_per_ellipse = 100
data = np.vstack([
    generate_elliptical_data(center=(2, 4), axes=(2, 1.5), angle=np.pi / 6, num_points=num_points_per_ellipse),
    generate_elliptical_data(center=(-3, -2.9), axes=(1, 2.2), angle=-np.pi / 4, num_points=num_points_per_ellipse),
    generate_elliptical_data(center=(4, -3), axes=(1, 1.5), angle=np.pi / 3, num_points=num_points_per_ellipse),
    generate_elliptical_data(center=(-4, 2), axes=(2.1, 2), angle=-np.pi / 6, num_points=num_points_per_ellipse)
])

# 最基础的K-means算法
kmeans_basic = KMeans(n_clusters=6, random_state=0) #don't know the real number of the cluster
labels_basic = kmeans_basic.fit_predict(data)

# 带分裂和合并机制的自适应K-means算法
def adaptive_kmeans(data, initial_k=2, max_iterations=100, split_threshold=0.5, merge_threshold=0.2):
    kmeans = KMeans(n_clusters=initial_k, random_state=0)
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    for _ in range(max_iterations):
        max_variance = np.max([np.var(data[labels == i], axis=0) for i in range(initial_k)])
        
        # 分裂过程
        if max_variance > split_threshold * 11.0:
            initial_k += 1
            kmeans = KMeans(n_clusters=initial_k, random_state=0)
            labels = kmeans.fit_predict(data)
            centroids = kmeans.cluster_centers_
        else:
            break

    # 合并过程
    while True:
        min_distance = np.min(cdist(centroids, centroids, metric='euclidean') + np.eye(initial_k) * 2)
        if min_distance < merge_threshold:
            initial_k -= 1
            kmeans = KMeans(n_clusters=initial_k, random_state=0)
            labels = kmeans.fit_predict(data)
            centroids = kmeans.cluster_centers_
        else:
            break

    return labels

labels_adaptive = adaptive_kmeans(data)

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(data[:, 0], data[:, 1], c=labels_basic, cmap='viridis', s=30)
axes[0].set_title('Basic K-means')

axes[1].scatter(data[:, 0], data[:, 1], c=labels_adaptive, cmap='viridis', s=30)
axes[1].set_title('Adaptive K-means with Splitting and Merging')

plt.show()
