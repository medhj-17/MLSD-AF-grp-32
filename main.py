from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sentence_transformers import SentenceTransformer
from umap import UMAP
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def dim_red(mat, p, method):
    '''
    Perform dimensionality reduction

    Input:
    -----
        mat : NxM list 
        p : number of dimensions to keep 
    Output:
    ------
        red_mat : NxP list such that p<<m
    '''
    if method=='ACP':
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=p)
        red_mat = pca.fit_transform(mat)
        
        
    elif method=='UMAP':
        umap_model = UMAP(n_components=p, random_state=42)
        red_mat = umap_model.fit_transform(embeddings)

    elif method=='TSNE':
        tsne = TSNE(n_components=p)
        red_mat = tsne.fit_transform(mat)
    else:
        raise Exception("Please select one of the four methods : APC, AFC, UMAP, TSNE")
    return red_mat


def clust(mat, k):
    '''
    Perform clustering

    Input:
    -----
        mat : input list 
        k : number of cluster
    Output:
    ------
        pred : list of predicted labels
    '''

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(mat)
    pred = kmeans.labels_
    
    return pred

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# Perform dimensionality reduction and clustering for each method
methods = ['ACP', 'UMAP','TSNE']
for method in methods:
    # Perform dimensionality reduction
    if method=='TSNE':
        red_emb = dim_red(embeddings, 3, method)
    else:
         red_emb = dim_red(embeddings, 20, method)

    # Perform clustering
    pred = clust(red_emb, k)
  
    # Evaluate clustering results
    nmi_score = normalized_mutual_info_score(pred, labels)
    ari_score = adjusted_rand_score(pred, labels)
    # Print results
    print(f'Method: {method}\nNMI: {nmi_score:.2f} \nARI: {ari_score:.2f}\n')

