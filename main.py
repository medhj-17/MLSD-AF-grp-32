from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups

# Define your custom functions
def clust(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    predictions = kmeans.fit_predict(embeddings)
    return predictions


def dim_red(embeddings, n_components):
    factor_analysis = FactorAnalysis(n_components=n_components, random_state=42)
    reduced_embeddings = factor_analysis.fit_transform(embeddings)
    return reduced_embeddings

