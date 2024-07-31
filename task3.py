import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN

credit_cards = pd.read_csv('./data/CC GENERAL.csv')

credit_cards = credit_cards.dropna()

clustering_data = credit_cards[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

dbscan = DBSCAN(eps=0.3, min_samples=10)
clusters = dbscan.fit_predict(clustering_data_scaled)

credit_cards["CREDIT_CARD_SEGMENTS"] = clusters

plot = go.Figure()

for cluster_label in np.unique(clusters):
    if cluster_label == -1:
        cluster_name = "Noise" 
    else:
        cluster_name = f"Cluster {cluster_label + 1}"  
    
    plot.add_trace(go.Scatter3d(
        x=credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == cluster_label]['BALANCE'],
        y=credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == cluster_label]['PURCHASES'],
        z=credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == cluster_label]['CREDIT_LIMIT'],
        mode='markers',
        marker=dict(size=6, line=dict(width=1)),
        name=cluster_name
    ))

plot.update_layout(
    width=800, height=800, autosize=True, showlegend=True,
    scene=dict(
        xaxis=dict(title='BALANCE', titlefont_color='black'),
        yaxis=dict(title='PURCHASES', titlefont_color='black'),
        zaxis=dict(title='CREDIT_LIMIT', titlefont_color='black'),
    ),
    font=dict(family="Gilroy", color='black', size=12),
    hovermode='closest',
    title="DBSCAN Clustering of Credit Card Data"
)

plot.show()
