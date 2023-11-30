from umap import UMAP
from sklearn.metrics import accuracy_score
# Define your custom functions
def clust(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    predictions = kmeans.fit_predict(embeddings)
    return predictions

def dim_red(embeddings, n_components=20):
    umap_model = UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings
