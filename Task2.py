import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load data
credit_cards = pd.read_csv('./data/CC GENERAL.csv')

# Drop rows with missing values
credit_cards = credit_cards.dropna()

# Select features for clustering
clustering_data = credit_cards[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

# Apply Min-Max scaling
scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Perform hierarchical clustering
Z = linkage(clustering_data_scaled, method='ward', metric='euclidean')

# Plot dendrogram to determine number of clusters
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Determine clusters based on the dendrogram
max_d = 7  # Adjust this threshold based on dendrogram
clusters = fcluster(Z, max_d, criterion='distance')

# Assign clusters to original dataframe
credit_cards["CREDIT_CARD_SEGMENTS"] = clusters

# Plot clusters on a 3D graph using Plotly
fig = go.Figure()
for cluster_id in credit_cards["CREDIT_CARD_SEGMENTS"].unique():
    fig.add_trace(go.Scatter3d(
        x=credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == cluster_id]['BALANCE'],
        y=credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == cluster_id]['PURCHASES'],
        z=credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == cluster_id]['CREDIT_LIMIT'],
        mode='markers',
        marker=dict(size=6, line=dict(width=1)),
        name=f'Cluster {cluster_id}'
    ))

fig.update_layout(
    width=800, height=800, autosize=True, showlegend=True,
    scene=dict(
        xaxis=dict(title='BALANCE', titlefont_color='black'),
        yaxis=dict(title='PURCHASES', titlefont_color='black'),
        zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black')
    ),
    font=dict(family="Gilroy", color='black', size=12)
)

fig.show()
