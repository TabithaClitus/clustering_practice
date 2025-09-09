📊 Clustering Algorithms – KMeans, Hierarchical & DBSCAN
This project demonstrates unsupervised learning techniques using synthetic data.
We apply KMeans, Hierarchical Clustering, and DBSCAN to group data points into clusters and visualize the results.

🚀 Project Workflow
1. Generate Synthetic Data
We use make_blobs from sklearn.datasets to create a dataset with 300 points and 3 clusters.

from sklearn.datasets import make_blobs
X,_ = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=42)
2. KMeans Clustering
Idea: Partition data into k clusters by minimizing distances to cluster centers.
Steps:
Fit KMeans model
Assign cluster labels
Plot data points with cluster centers
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3, random_state=42)
kmeans_labels=kmeans.fit_predict(X)
📌 Visualization:

Points are colored by cluster
Red X marks the cluster centers
3. Hierarchical Clustering
Idea: Build a tree (dendrogram) by successively merging clusters.
Ward’s method: Minimizes variance within clusters.
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
Z=linkage(X,method='ward')
hier_labels=fcluster(Z,t=3,criterion='maxclust')
📌 Visualization:

Scatter plot of 3 clusters
Dendrogram showing cluster merges
4. DBSCAN Clustering
Idea: Groups points that are closely packed and marks outliers as noise.
Parameters:
eps=1.0 → neighborhood radius
min_samples=5 → minimum points to form a cluster
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=1.0, min_samples=5)
dbscan_labels=dbscan.fit_predict(X)
📌 Visualization:

Points in clusters have unique colors
Noise points are labeled -1
