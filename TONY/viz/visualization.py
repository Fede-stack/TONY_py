from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import umap


class visualize_topics:
    """
    Pipeline for text clustering and visualization.

    Steps:
        1. Encode texts with a SentenceTransformer model
        2. Reduce dimensionality with UMAP (high-dim for clustering)
        3. Cluster with KMeans
        4. Extract keywords per cluster with KeyBERT
        5. Visualize with UMAP 2D + interactive Plotly plot
    """

    def __init__(
        self,
        n_clusters: int = 50,
        model_name: str = 'FritzStack/all-MiniLM-L6-v2-TSDAE-RedditMentalHealth',
        umap_params: dict = None,
        top_n_keywords: int = 15,
        device: str = None,
    ):
        """
        Args:
            n_clusters:       Number of KMeans clusters
            model_name:       SentenceTransformer model name or path
            umap_params:      Optional dict to override default UMAP parameters
            top_n_keywords:   Number of keywords to extract per cluster (KeyBERT)
            device:           Device for SentenceTransformer ('cpu', 'cuda', 'mps')
        """
        self.n_clusters = n_clusters
        self.model_name = model_name
        self.top_n_keywords = top_n_keywords

        self.umap_params = umap_params or {
            'n_neighbors': 30,
            'n_components': 10,
            'min_dist': 0.1,
            'metric': 'cosine',
            'random_state': 0,
            'n_jobs': -1,
        }

        # Internal state
        self.embeddings = None
        self.X_umap = None
        self.embedding_2d = None
        self.labels = None
        self.df_final = None
        self.df_clusters = None
        self.keywords = None
        self.keyword_strings = None

        print("Loading models...")
        self.encoder = SentenceTransformer(model_name, device=device)
        print(f"  ✓ SentenceTransformer: {model_name}")
        self.kw_model = KeyBERT()
        print(f"  ✓ KeyBERT ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts: list) -> "TextClusterer":
        """
        Run the full pipeline on a list of texts.

        Args:
            texts: List of raw text strings

        Returns:
            self (for method chaining)
        """
        self.texts = texts

        print(f"\n[1/5] Encoding {len(texts)} texts...")
        self._encode(texts)

        print("[2/5] UMAP dimensionality reduction (high-dim)...")
        self._reduce_umap()

        print("[3/5] KMeans clustering...")
        self._cluster()

        print("[4/5] Extracting keywords per cluster...")
        self._extract_keywords()

        print("[5/5] UMAP 2D projection for visualization...")
        self._reduce_umap_2d()

        print("\n✓ Pipeline complete.")
        return self

    def plot(self, title: str = "UMAP 2D Embedding") -> go.Figure:
        """
        Build and display the interactive Plotly scatter plot.

        Args:
            title: Plot title

        Returns:
            Plotly Figure object
        """
        self._check_fitted()
        fig = self._build_figure(title)
        fig.show()
        return fig

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return a DataFrame with cluster id, post count, and top keywords."""
        self._check_fitted()
        summary = self.df_clusters[['cluster', 'n_posts']].copy()
        summary['keywords'] = [
            ', '.join(kw for kw, _ in self.keywords[k])
            for k in range(self.n_clusters)
        ]
        return summary

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _encode(self, texts: list):
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

    def _reduce_umap(self):
        reducer = umap.UMAP(**self.umap_params)
        self.X_umap = reducer.fit_transform(self.embeddings)

    def _cluster(self):
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.labels = clusterer.fit_predict(self.X_umap)

        self.df_final = pd.DataFrame({
            'Text': self.texts,
            'cluster': self.labels,
        })

        # Build per-cluster merged text
        posts_by_cluster = defaultdict(list)
        for _, row in self.df_final.iterrows():
            if row['cluster'] != -1:
                posts_by_cluster[row['cluster']].append(row['Text'])

        cluster_list = [
            {
                'cluster': cluster_id,
                'n_posts': len(posts),
                'merged_text': '\n\n'.join(posts),
            }
            for cluster_id, posts in sorted(posts_by_cluster.items())
        ]
        self.df_clusters = pd.DataFrame(cluster_list)

    def _extract_keywords(self):
        self.keywords = [
            self.kw_model.extract_keywords(
                self.df_clusters.merged_text[k],
                top_n=self.top_n_keywords,
                stop_words='english',
            )
            for k in range(self.n_clusters)
        ]

        self.keyword_strings = {
            k: '<br>'.join(f'• {kw}' for kw, _ in self.keywords[k])
            for k in range(self.n_clusters)
        }

    def _reduce_umap_2d(self):
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        self.embedding_2d = reducer_2d.fit_transform(self.X_umap)

    def _build_figure(self, title: str) -> go.Figure:
        fig = go.Figure()
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            mask = self.labels == label
            pts = self.embedding_2d[mask]
            keyword_str = self.keyword_strings.get(int(label), 'No keywords available')

            fig.add_trace(go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode='markers',
                marker=dict(size=6, opacity=0.7, line=dict(width=0.8, color='black')),
                name=f'Cluster {label}',
                customdata=[[keyword_str]] * len(pts),
                hovertemplate=(
                    f"<b>Cluster {label}</b><br>"
                    "──────────────<br>"
                    "<b>Top Keywords:</b><br>%{customdata[0]}"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=title,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                orientation='v',
                x=1.01, xanchor='left',
                y=0.5, yanchor='middle',
            ),
            xaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#e0e0e0', zeroline=False),
        )
        fig.update_xaxes(title_text="UMAP Dim 1")
        fig.update_yaxes(title_text="UMAP Dim 2")
        fig.update_traces(cliponaxis=False)
        return fig

    def _check_fitted(self):
        if self.labels is None:
            raise RuntimeError("Call .fit(texts) before using this method.")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        status = "fitted" if self.labels is not None else "not fitted"
        return (
            f"TextClusterer(\n"
            f"  model='{self.model_name}',\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  top_n_keywords={self.top_n_keywords},\n"
            f"  status={status}\n"
            f")"
        )
