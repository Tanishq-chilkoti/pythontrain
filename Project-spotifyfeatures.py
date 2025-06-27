import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

df = pd.read_csv("SpotifyFeatures.csv")
print(df.head())

df.dropna(inplace=True)

features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Best K: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print("Silhouette Score (K-Means):", silhouette_score(X_scaled, df['KMeans_Cluster']))
print("Davies-Bouldin Index (K-Means):", davies_bouldin_score(X_scaled, df['KMeans_Cluster']))

dbscan = DBSCAN(eps=1.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)


mask = df['DBSCAN_Cluster'] != -1
if mask.sum() > 0:
    print("Silhouette Score (DBSCAN):", silhouette_score(X_scaled[mask], df.loc[mask, 'DBSCAN_Cluster']))
    print("Davies-Bouldin Index (DBSCAN):", davies_bouldin_score(X_scaled[mask], df.loc[mask, 'DBSCAN_Cluster']))
else:
    print("DBSCAN found only noise, no valid clusters to evaluate.")

def recommend_similar(song_index, method='KMeans'):
    if method == 'KMeans':
        cluster_label = df.loc[song_index, 'KMeans_Cluster']
        cluster_songs = df[df['KMeans_Cluster'] == cluster_label]
    elif method == 'DBSCAN':
        cluster_label = df.loc[song_index, 'DBSCAN_Cluster']
        if cluster_label == -1:
            return "Song is noise in DBSCAN clustering, no recommendations."
        cluster_songs = df[df['DBSCAN_Cluster'] == cluster_label]
    else:
        return "Invalid method."

    recommendations = cluster_songs.sample(5)
    return recommendations[['track_name', 'artist_name']]

print(recommend_similar(0, method='KMeans'))