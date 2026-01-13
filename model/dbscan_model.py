# =========================================
# DBSCAN CLUSTERING - MALL CUSTOMER DATASET
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv("data/Mall_Customers.csv")

# Select relevant numeric features
df = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print(df.head())

# =========================================
# 2. FEATURE SCALING (MANDATORY)
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =========================================
# 3. DBSCAN MODEL
# =========================================

dbscan = DBSCAN(
    eps=0.4,          # try tuning this
    min_samples=5
)

df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# =========================================
# 4. IDENTIFY NOISE
# =========================================

noise_count = (df["DBSCAN_Cluster"] == -1).sum()
print("Noise points detected:", noise_count)

# =========================================
# 5. VISUALIZE DBSCAN RESULT
# =========================================

plt.figure(figsize=(6,5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["DBSCAN_Cluster"],
    cmap="tab10"
)
plt.title("DBSCAN Clustering (−1 = Noise)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

# =========================================
# 6. SILHOUETTE SCORE (REMOVE NOISE)
# =========================================

filtered_df = df[df["DBSCAN_Cluster"] != -1]

if filtered_df["DBSCAN_Cluster"].nunique() > 1:
    sil_score = silhouette_score(
        filtered_df[["Annual Income (k$)", "Spending Score (1-100)"]],
        filtered_df["DBSCAN_Cluster"]
    )
    print("Silhouette Score (DBSCAN):", sil_score)
else:
    print("Silhouette score not defined (only one cluster)")

# =========================================
# 7. OPTIONAL: COMPARE WITH K-MEANS
# =========================================

kmeans = KMeans(n_clusters=5, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(6,5))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_Cluster"],
    cmap="viridis"
)
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()
