#-------------------------------------------------------------------------
# AUTHOR: Alexander Rodriguez 
# FILENAME: clustering.py
# SPECIFICATION: Runs k-means multiple times and check which k value max the Silhouette coefficient
# FOR: CS 4210- Assignment #5
# TIME SPENT: 20min
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

# Read the training data from 'training_data.csv'
df = pd.read_csv('training_data.csv', sep=',', header=None)

# Assign your training data to X_training feature matrix
X_training = df.values

# Initialize variables to store the best k and silhouette coefficient
best_k = 0
best_silhouette = -1

# Lists to store k values and their corresponding silhouette coefficients
k_values = []
silhouette_coeffs = []

# Run k-means testing different k values from 2 until 20 clusters
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

    # Calculate the silhouette coefficient for the current k
    silhouette = silhouette_score(X_training, kmeans.labels_)

    # Store the k and its corresponding silhouette coefficient
    k_values.append(k)
    silhouette_coeffs.append(silhouette)

    # Check if the current k has a higher silhouette coefficient than the previous best
    if silhouette > best_silhouette:
        best_k = k
        best_silhouette = silhouette

# Plot the silhouette coefficient for each k value
plt.plot(k_values, silhouette_coeffs, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient for Different k Values')
plt.show()

# Read the test data (clusters) from 'testing_data.csv'
test_df = pd.read_csv('testing_data.csv', sep=',', header=None)

# Assign your data labels to vector labels
labels = np.array(test_df.values).reshape(1, -1)[0]

# Run k-means with the best k value on the training data
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(X_training)

# Calculate and print the Homogeneity of this k-means clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
