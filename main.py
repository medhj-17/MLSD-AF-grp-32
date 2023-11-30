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

# import data
ng20 = fetch_20newsgroups(subset='test')
corpus = ng20.data[:2000]
labels = ng20.target[:2000]
k = len(set(labels))

# embedding
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(corpus)

# perform dimentionality reduction
red_emb = dim_red(embeddings, 20)

# perform clustering
pred = clust(red_emb, k)

# evaluate clustering results
nmi_score = normalized_mutual_info_score(pred,labels)
ari_score = adjusted_rand_score(pred,labels)
accuracy = accuracy_score(labels, pred)

print(f'NMI: {nmi_score:.2f}\nAccuracy: {accuracy:.2f}\nARI: {ari_score:.2f}')

