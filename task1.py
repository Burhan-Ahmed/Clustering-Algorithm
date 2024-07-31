import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
file_path = './data/CC GENERAL.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Original Data:")
print(data.head())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column].astype(str))

# Use SimpleImputer to fill missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Convert back to DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Display the first few rows of the imputed dataset
print("\nImputed Data:")
print(data_imputed.head())

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Convert back to DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Display the first few rows of the scaled dataset
print("\nScaled Data:")
print(data_scaled.head())

# Save the preprocessed dataset
preprocessed_file_path = './data/CC_GENERAL_preprocessed.csv'
data_scaled.to_csv(preprocessed_file_path, index=False)
print(f"\nPreprocessed data saved to {preprocessed_file_path}")

# K-means clustering and SSE calculation
sse = []
K_range = range(1, 16)

for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(data_scaled)
    sse.append(kmeans.inertia_)

# Plotting the SSE values
plt.figure(figsize=(10, 6))
plt.plot(K_range, sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method For Optimal K')
plt.xticks(K_range)
plt.grid(True)
plt.show()

# Run K-means with the optimal number of clusters (let's assume K=5 for this example)
optimal_K = 5
kmeans = KMeans(n_clusters=optimal_K, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Add cluster labels to the original and scaled datasets
data['Cluster'] = clusters
data_scaled['Cluster'] = clusters

# Display the first few rows of the dataset with clusters
print("\nData with Cluster Labels:")
print(data.head())

print("\nScaled Data with Cluster Labels:")
print(data_scaled.head())

# Save the dataset with cluster labels
clustered_file_path = './data/CC_GENERAL_with_clusters.csv'
data.to_csv(clustered_file_path, index=False)
print(f"\nDataset with clusters saved to {clustered_file_path}")
