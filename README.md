# Clustering avec réduction de dimension sur les données NG20

Ce projet consiste à tester diverses approches de la réduction de dimension et du clustring,nous avons travaillé avec trois méthodes de réduction de dimension à savoir : l'Analyse en Composantes Principales (ACP), le t-SNE et l'UMAP, ensuite nous avons fait la visualisations des résultats de ces méthodes en les paramétrant sur 2 composantes et la méthode de clustring choisie est celle des k-means. Concernantles données utilisées des données textuelles en entrée (en l'occurrence, les données NG20 limitées à 2000 documents). Pour la représentation vectorielle des données textuelles, le modèle de langage Sentence Transformer a été employé.

L'objectif est de voir quelle combinaison donne les meilleurs résultats :

## Paramètres :
|   Modèle Sentence Transformer |     paraphrase-MiniLM-L6-v2     |
|      Dimension réduite        |    20 (ACP, UMAP), 3 (t-SNE)    |

 
## Evaluation des méthodes : 

|     Méthode     |    NMI    |    ARI    |
|-----------------|-----------|-----------|
|   ACP + k-means |    0.42   |   0.23    |
| t-SNE + k-means |    0.10   |   0.02    |
|  UMAP + k-means |    0.44   |   0.25    |


## Visualisation des résultats :
![Plot du résultat de clustering avec ACP, t-SNE et UMAP](results.png)
En exécutant le conteneur, Cette visualisation est sauvegardée dans une image "results.png"
