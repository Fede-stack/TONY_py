import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from collections import defaultdict
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from pathlib import Path

class ClusterVisualizer:
    def __init__(self, n_clusters=50, embedding_model_name='FritzStack/all-MiniLM-L6-v2-TSDAE-RedditMentalHealth'):
        self.n_clusters = n_clusters
        self.embedding_model_name = embedding_model_name
        self.model = SentenceTransformer(embedding_model_name)
        
    def fit(self, texts, df=None):
        """Generate embeddings, UMAP, KMeans, keywords"""
        # Embeddings
        self.texts = texts
        self.embeddings = self.model.encode(texts)
        print(f'Embeddings shape: {self.embeddings.shape}')
        
        # UMAP 10D
        self.reducer = umap.UMAP(n_neighbors=15, n_components=10, metric='cosine', random_state=0, n_jobs=-1)
        self.X_umap = self.reducer.fit_transform(self.embeddings)
        
        # KMeans
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.labels = self.clusterer.fit_predict(self.X_umap)
        
        # df con cluster
        if df is not None:
            self.df_final = df.copy()
            self.df_final['cluster'] = self.labels
            self.df_final.to_csv('reddit_clustered.csv', index=False)
        
        # UMAP 2D
        self.reducer_2d = umap.UMAP(n_components=2, random_state=0)
        self.embedding_2d = self.reducer_2d.fit_transform(self.X_umap)
        
        
        self._extract_keywords()
        
        return self
    
    def _extract_keywords(self):
        """Estract top keywords per cluster """
        self.keys = {}
        vectorizer = KeyphraseCountVectorizer()
        
        for k in range(self.n_clusters):
            mask = self.labels == k
            cluster_texts = [self.texts[i] for i in np.where(mask)[0]]
            if cluster_texts:
                vectorizer.fit(cluster_texts)
                self.keys[k] = list(vectorizer.get_feature_names_out()[:15])
    
    def plot(self, title='UMAP Clusters con Keywords'):
        """Plotly figure"""
        cluster_info = defaultdict(dict)
        for cluster_id in range(self.n_clusters):
            mask = self.labels == cluster_id
            if mask.sum() > 0:
                info = {
                    'n_points': mask.sum(),
                    'top_keywords': '<br>'.join(self.keys.get(cluster_id, [])[:10])
                }
                cluster_info[cluster_id] = info
        
     
        n_colors = min(self.n_clusters, 20)
        colors = px.colors.qualitative.Set1[:n_colors//2] + px.colors.qualitative.Set2[:n_colors//2]
        
        fig = go.Figure()
        for i, cluster_id in enumerate(range(self.n_clusters)):
            mask = self.labels == cluster_id
            if mask.sum() > 0:
                info = cluster_info[cluster_id]
                hover_text = f"""<b>Cluster {cluster_id}
                </b>Keywords:<br>{info['top_keywords']}"""
                
                fig.add_trace(go.Scatter(
                    x=self.embedding_2d[mask, 0],
                    y=self.embedding_2d[mask, 1],
                    mode='markers',
                    marker=dict(size=4, opacity=0.6, color=colors[i % len(colors)]),
                    showlegend=False,
                    hovertemplate=hover_text + '<extra></extra>'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='UMAP 1', yaxis_title='UMAP 2',
            height=700, width=900,
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white'
        )
        return fig
