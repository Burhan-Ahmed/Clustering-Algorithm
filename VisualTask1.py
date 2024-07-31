import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

credit_cards=pd.read_csv('./data/CC GENERAL.csv')

#one liner to see number of rows and columns in a Pandas dataframe
rows, columns = credit_cards.shape

print("The number of rows in the DataFrame is: ", rows)
print("The number of columns in the DataFrame is: ", columns)

print(credit_cards.head())

# check for any null values
credit_cards.isnull().sum()

#drop the rows with missing values
credit_cards = credit_cards.dropna()

#Dataset After PreProcessing
rows, columns = credit_cards.shape

# create the clusters
clustering_data = credit_cards[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
for i in clustering_data.columns:
    MinMaxScaler(i)
    
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(clustering_data)
credit_cards["CREDIT_CARD_SEGMENTS"] = clusters

print(clustering_data)

X = clustering_data.values

# elbow method

def elbow_method(X, max_clusters):
    sse = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        sse.append(kmeans.inertia_)
        
    plt.plot(range(1, max_clusters+1), sse)
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Distances")
    plt.grid(True)
    plt.show()

elbow_method(X, 15)

# no significant sum of squared distances reduction past 4 clusters

clustering_data = credit_cards[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
for i in clustering_data.columns:
    MinMaxScaler(i)
    
kmeans = KMeans(n_clusters=4)
clusters = kmeans.fit_predict(clustering_data)
credit_cards["CREDIT_CARD_SEGMENTS"] = clusters

#transform names of clusters for easy interpretation

credit_cards["CREDIT_CARD_SEGMENTS"] = credit_cards["CREDIT_CARD_SEGMENTS"].map({0: "Cluster 1", 1: 
    "Cluster 2", 2: "Cluster 3", 3: "Cluster 4"})
print(credit_cards["CREDIT_CARD_SEGMENTS"].head(10))

#plot the clusters on a 3D graph using Plotly

PLOT = go.Figure()
for i in list(credit_cards["CREDIT_CARD_SEGMENTS"].unique()):
    

    PLOT.add_trace(go.Scatter3d(x = credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"]== i]['BALANCE'],
                                y = credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == i]['PURCHASES'],
                                z = credit_cards[credit_cards["CREDIT_CARD_SEGMENTS"] == i]['CREDIT_LIMIT'],                        
                                mode = 'markers',marker_size = 6, marker_line_width = 1,
                                name = str(i)))
PLOT.update_traces(hovertemplate='BALANCE: %{x} <br>PURCHASES %{y} <br>DCREDIT_LIMIT: %{z}')

    
PLOT.update_layout(width = 800, height = 800, autosize = True, showlegend = True,
                   scene = dict(xaxis=dict(title = 'BALANCE', titlefont_color = 'black'),
                                yaxis=dict(title = 'PURCHASES', titlefont_color = 'black'),
                                zaxis=dict(title = 'CREDIT_LIMIT', titlefont_color = 'black')),
                   font = dict(family = "Gilroy", color  = 'black', size = 12))