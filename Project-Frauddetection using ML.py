import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

data = pd.read_csv("creditcard.csv")
features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4']
X = data[features]
X[['Time', 'Amount']] = StandardScaler().fit_transform(X[['Time', 'Amount']])

labels = DBSCAN(eps=1.5, min_samples=5).fit_predict(X)

mask = labels != -1
X_core = X[mask]
labels_core = labels[mask]

n_total = len(labels)
n_outliers = np.sum(labels == -1)
n_clusters = len(set(labels_core))
silhouette = silhouette_score(X_core, labels_core) if n_clusters > 1 else None
db_index = davies_bouldin_score(X_core, labels_core) if n_clusters > 1 else None

summary = [
    ["Total transactions", n_total],
    ["Clustered transactions", n_total - n_outliers],
    ["Outliers", n_outliers],
    ["Silhouette Score", f"{silhouette:.3f}" if silhouette else "NA"],
    ["Davies-Bouldin Index", f"{db_index:.3f}" if db_index else "NA"],
]

fig, ax = plt.subplots(figsize=(10, 5))
table = ax.table(cellText=summary, colLabels=["Metric", "Value"], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title("DBSCAN Clustering Summary", fontsize=14, pad=20)
plt.show()

anomalies = data[labels == -1]
print("Anomalous Transactions (Outliers):")
print(anomalies)
