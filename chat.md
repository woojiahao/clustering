To determine which cluster a new document belongs to within its class, given that you have previously clustered documents using hierarchical clustering, you can use the following approach:

### Step-by-Step Guide

1. **Represent the Clusters**:
   - Compute a representative for each cluster. Common choices include the centroid (mean vector of all documents in the cluster) or medoid (the document within the cluster that minimizes the average distance to all other documents in the cluster).

2. **Feature Representation of the New Document**:
   - Ensure the new document is represented in the same feature space as the documents used for clustering. This might involve using the same TF-IDF vectorizer, word embeddings, or any other feature extraction method.

3. **Distance Calculation**:
   - Compute the distance between the new document and each cluster representative. Common distance metrics include Euclidean distance, cosine similarity, or other appropriate metrics based on your feature space.

4. **Assign to the Nearest Cluster**:
   - Assign the new document to the cluster with the smallest distance to the document.

### Detailed Steps

#### Step 1: Compute Cluster Representatives
If your documents are represented as vectors (e.g., using TF-IDF, word embeddings), you can compute the centroid of each cluster as follows:

```python
import numpy as np

def compute_centroid(cluster_vectors):
    return np.mean(cluster_vectors, axis=0)

# Example: Assuming you have clusters represented as lists of vectors
clusters = {
    'cluster_1': [vector1, vector2, vector3],
    'cluster_2': [vector4, vector5, vector6],
    # Add more clusters as needed
}

cluster_centroids = {cluster: compute_centroid(vectors) for cluster, vectors in clusters.items()}
```

#### Step 2: Feature Representation of the New Document
Make sure the new document is vectorized using the same method as the clustered documents.

```python
# Assuming you have a function to vectorize documents
new_document_vector = vectorize_document(new_document)
```

#### Step 3: Distance Calculation
Calculate the distance between the new document vector and each cluster centroid.

```python
from scipy.spatial.distance import euclidean, cosine

def find_closest_cluster(new_vector, cluster_centroids, distance_metric='euclidean'):
    min_distance = float('inf')
    closest_cluster = None

    for cluster, centroid in cluster_centroids.items():
        if distance_metric == 'euclidean':
            distance = euclidean(new_vector, centroid)
        elif distance_metric == 'cosine':
            distance = cosine(new_vector, centroid)
        else:
            raise ValueError("Unsupported distance metric")

        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster

    return closest_cluster

closest_cluster = find_closest_cluster(new_document_vector, cluster_centroids)
```

#### Step 4: Assign to the Nearest Cluster
The variable `closest_cluster` now holds the identifier of the cluster to which the new document is closest.

### Example Code
Here’s a complete example assuming you have the necessary functions and data:

```python
import numpy as np
from scipy.spatial.distance import euclidean, cosine

# Function to compute the centroid of a cluster
def compute_centroid(cluster_vectors):
    return np.mean(cluster_vectors, axis=0)

# Function to find the closest cluster
def find_closest_cluster(new_vector, cluster_centroids, distance_metric='euclidean'):
    min_distance = float('inf')
    closest_cluster = None

    for cluster, centroid in cluster_centroids.items():
        if distance_metric == 'euclidean':
            distance = euclidean(new_vector, centroid)
        elif distance_metric == 'cosine':
            distance = cosine(new_vector, centroid)
        else:
            raise ValueError("Unsupported distance metric")

        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster

    return closest_cluster

# Assuming you have clusters represented as lists of vectors
clusters = {
    'cluster_1': [vector1, vector2, vector3],
    'cluster_2': [vector4, vector5, vector6],
    # Add more clusters as needed
}

# Compute centroids for each cluster
cluster_centroids = {cluster: compute_centroid(vectors) for cluster, vectors in clusters.items()}

# Vectorize the new document
new_document_vector = vectorize_document(new_document)

# Find the closest cluster
closest_cluster = find_closest_cluster(new_document_vector, cluster_centroids)

print(f"The new document belongs to {closest_cluster}")
```

### Conclusion
By representing each cluster with a centroid or medoid and computing the distance between the new document and these representatives, you can efficiently assign the new document to the appropriate cluster. This method ensures consistency with your hierarchical clustering results and leverages the distances in the feature space to determine the closest cluster.