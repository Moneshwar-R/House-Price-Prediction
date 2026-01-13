# =========================================
# CLUSTERING: K-MEANS + HIERARCHICAL
# (Mall Customers Dataset)
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv("data/Mall_Customers.csv")

# Select relevant features for clustering
df = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print(df.head())

# =========================================
# 2. FEATURE SCALING
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================================
# 3. ELBOW METHOD
# =========================================

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

# =========================================
# 4. SILHOUETTE ANALYSIS
# =========================================

sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

plt.figure(figsize=(6,4))
plt.plot(range(2,11), sil_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid(True)
plt.show()

# =========================================
# 5. FINAL K-MEANS MODEL
# =========================================

kmeans = KMeans(n_clusters=5, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# =========================================
# 6. VISUALIZE K-MEANS CLUSTERS
# =========================================

plt.figure(figsize=(6,5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_Cluster"],
    cmap="viridis"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Mall Customer Segmentation (K-Means)")
plt.grid(True)
plt.show()

# =========================================
# 7. PREDICT CLUSTER FOR NEW CUSTOMER
# =========================================

new_customer = np.array([[60, 65]])  # income, spending
new_customer_scaled = scaler.transform(new_customer)

cluster_id = kmeans.predict(new_customer_scaled)
print("New customer belongs to cluster:", cluster_id[0])

# =========================================
# 8. HIERARCHICAL CLUSTERING
# =========================================

linked = linkage(X_scaled, method="ward")

# =========================================
# 9. DENDROGRAM
# =========================================

plt.figure(figsize=(8,5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# =========================================
# 10. ASSIGN HIERARCHICAL CLUSTERS
# =========================================

df["Hierarchical_Cluster"] = fcluster(
    linked,
    t=5,
    criterion="maxclust"
)

print(df.head())
