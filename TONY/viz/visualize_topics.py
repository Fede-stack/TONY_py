"""
visualize_topics.py
───────────────────
Extended version of the original TextClusterer pipeline.

Clustering backend (choose one):
  • KMeans (default)  — fast, needs explicit n_clusters
      encode → UMAP (high-dim) → KMeans → ...
  • AdaptiveSpectralClustering — data-driven k* neighbors + spectral embedding
      encode → ASC (ID estimation + similarity matrix + spectral) → ...

LLM labeling step (optional):
  A small quantized model refines each cluster's topic name and keywords
  using two focused micro-tasks (Task A: label, Task B: keywords).

Usage
─────
  from llm_labeler import MLXLabeler
  from AdaptiveSpectralClustering import AdaptiveSpectralClustering

  labeler = MLXLabeler("mlx-community/Qwen3.5-0.8B-OptiQ-4bit")

  # Default: UMAP + KMeans
  model = visualize_topics(n_clusters=5, llm_labeler=labeler)

  # Alternative: AdaptiveSpectralClustering (n_clusters still needed for final KMeans)
  model = visualize_topics(n_clusters=5, use_adaptive_spectral=True, llm_labeler=labeler)

  model.fit(texts)
  model.plot()
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _wrap_label(label: str, max_chars: int = 14) -> str:
    """
    Break a label into two lines for display inside a bubble.
    Splits at the last space before max_chars; falls back to hard truncation.
    Returns empty string if max_chars == 0.
    """
    if max_chars == 0 or not label:
        return ""
    if len(label) <= max_chars:
        return label
    # Try to split at a word boundary
    cut = label[:max_chars].rfind(" ")
    if cut <= 0:
        cut = max_chars
    return label[:cut] + "<br>" + label[cut:].strip()
import umap
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .llm_labeler import BaseLLMLabeler

# ──────────────────────────────────────────────────────────────────────────────
#  Colour palette (one per cluster, wraps around)
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#5C6BC0", "#26A69A", "#EF5350", "#FFA726", "#66BB6A",
    "#AB47BC", "#29B6F6", "#FF7043", "#8D6E63", "#78909C",
    "#EC407A", "#FFEE58", "#9CCC65", "#26C6DA", "#7E57C2",
    "#D4E157", "#42A5F5", "#FF5722", "#00ACC1", "#43A047",
]


def _get_color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


# ──────────────────────────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────────────────────────

class visualize_topics:
    """
    Pipeline: encode → [KMeans | AdaptiveUMAP] → KeyBERT → LLM label → UMAP 2D → Plotly.

    Clustering backends (clustering_backend=):
      'kmeans'         — encode → UMAP (high-dim) → KMeans   [default]
      'adaptive_umap'  — encode → AdaptiveUMAP (ID-estimated dim + k* neighbors
                         + adaptive fuzzy graph + simplicial embedding → KMeans)
                         No manual hyperparameters needed: d and k* come from the data.

    Args:
        n_clusters:         Number of KMeans clusters (both backends).
        model_name:         SentenceTransformer model name or path.
        umap_params:        UMAP params for the high-dim step
                            (ignored when clustering_backend='adaptive_umap').
        top_n_keywords:     KeyBERT keywords per cluster (pre-LLM pool).
        device:             SentenceTransformer device ('cpu', 'cuda', 'mps').
        llm_labeler:        Optional BaseLLMLabeler. If None, LLM step is skipped.
        n_samples_for_llm:  Max texts per cluster forwarded to the LLM.
        clustering_backend: 'kmeans' or 'adaptive_umap'.
        adaptive_umap_params: kwargs forwarded to AdaptiveUMAP
                            e.g. {"n_iter": 10, "metric": "euclidean", "n_epochs": 200}.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        model_name: str = "all-MiniLM-L6-v2",
        umap_params: dict | None = None,
        top_n_keywords: int = 15,
        device: str | None = None,
        llm_labeler: BaseLLMLabeler | None = None,
        n_samples_for_llm: int = 5,
        clustering_backend: str = "kmeans",
        adaptive_umap_params: dict | None = None,
    ):
        if clustering_backend not in ("kmeans", "adaptive_umap"):
            raise ValueError(
                f"clustering_backend must be 'kmeans' or 'adaptive_umap', "
                f"got '{clustering_backend}'"
            )

        self.n_clusters           = n_clusters
        self.model_name           = model_name
        self.top_n_keywords       = top_n_keywords
        self.llm_labeler          = llm_labeler
        self.n_samples_for_llm    = n_samples_for_llm
        self.clustering_backend   = clustering_backend
        self.adaptive_umap_params = adaptive_umap_params or {}

        self.umap_params = umap_params or {
            "n_neighbors": 15,
            "n_components": 10,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 0,
            "n_jobs": -1,
        }

        # Internal state
        self.texts: list[str] = []
        self.embeddings      = None
        self.X_umap          = None    # populated only in kmeans path
        self.embedding_2d    = None
        self.labels          = None
        self.df_final: pd.DataFrame | None = None
        self.df_clusters: pd.DataFrame | None = None
        self.keybert_keywords: list | None = None
        self.llm_labels: list[dict] | None = None
        self.hover_strings: dict[int, str] = {}

        print("Loading embedding models...")
        self.encoder = SentenceTransformer(model_name, device=device)
        print(f"  ✓ SentenceTransformer: {model_name}")
        self.kw_model = KeyBERT()
        print("  ✓ KeyBERT ready")
        if llm_labeler:
            print(f"  ✓ LLM labeler: {type(llm_labeler).__name__}")
        print(f"  ✓ Clustering backend: {clustering_backend}")

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "visualize_topics":
        self.texts = texts
        is_adaptive  = self.clustering_backend == "adaptive_umap"
        n_llm        = 1 if self.llm_labeler else 0
        # kmeans path:       encode, umap_high, cluster, kw, [llm], umap_2d  → 5 or 6
        # adaptive_umap path: encode, adaptive_umap, kw, [llm], umap_2d      → 4 or 5
        total_steps  = (4 if is_adaptive else 5) + n_llm
        step         = 0

        step += 1
        print(f"\n[{step}/{total_steps}] Encoding {len(texts)} texts...")
        self._encode(texts)

        if is_adaptive:
            step += 1
            print(f"[{step}/{total_steps}] AdaptiveUMAP "
                  "(ID estimation → k* graph → simplicial embedding → KMeans)...")
            self._adaptive_umap_cluster()
        else:
            step += 1
            print(f"[{step}/{total_steps}] UMAP dimensionality reduction (high-dim)...")
            self._reduce_umap()
            step += 1
            print(f"[{step}/{total_steps}] KMeans clustering...")
            self._cluster()

        step += 1
        print(f"[{step}/{total_steps}] Extracting keywords with KeyBERT...")
        self._extract_keywords()

        if self.llm_labeler:
            step += 1
            print(f"[{step}/{total_steps}] LLM cluster labeling "
                  f"(2 tasks × {self.n_clusters} clusters)...")
            self._llm_label()

        step += 1
        print(f"[{step}/{total_steps}] UMAP* 2D projection for visualization...")
        self._reduce_umap_2d()
        self._build_hover_strings()

        print("\n✓ Pipeline complete.")
        return self

    def plot(self, title: str = "Topic Map") -> tuple[go.Figure, go.Figure]:
        """
        Show two separate figures:
          1. Scatter plot  — individual posts in 2D UMAP space
          2. Treemap       — cluster sizes with labels and keywords
        Returns (fig_scatter, fig_treemap).
        """
        self._check_fitted()
        fig_scatter  = self._build_scatter(title)
        fig_treemap  = self._build_treemap(title)
        fig_scatter.show()
        fig_treemap.show()
        return fig_scatter, fig_treemap

    def _build_scatter(self, title: str) -> go.Figure:
        """
        Clean scatter plot: one point per post, colored by cluster.
        No overlapping annotations — cluster identity lives in the legend
        and in the hover tooltip.
        """
        fig = go.Figure()

        for k in range(self.n_clusters):
            mask      = self.labels == k
            pts       = self.embedding_2d[mask]
            raw_texts = self.df_final.loc[mask, "text"].fillna("").tolist()
            previews  = [t[:130] + ("…" if len(t) > 130 else "") for t in raw_texts]
            color     = _get_color(k)

            if self.llm_labels:
                legend_name = self.llm_labels[k]["label"]
            else:
                legend_name = f"Cluster {k}"

            fig.add_trace(go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="markers",
                marker=dict(
                    size=9,
                    color=color,
                    opacity=0.72,
                    line=dict(width=0.5, color="white"),
                ),
                name=legend_name,
                legendgroup=str(k),
                customdata=[[self.hover_strings[k], p] for p in previews],
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    "──────────────<br>"
                    "<i>%{customdata[1]}</i>"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b> — UMAP* 2D embedding",
                font=dict(size=20),
                x=0.5, xanchor="center",
            ),
            width=1200, height=780,
            plot_bgcolor="#f7f8fc",
            paper_bgcolor="white",
            xaxis=dict(
                title=dict(text="UMAP* dim 1", font=dict(size=14)),
                showgrid=True, gridcolor="#e2e4ed",
                zeroline=False, showline=True, linecolor="#c0c0c0",
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title=dict(text="UMAP* dim 2", font=dict(size=14)),
                showgrid=True, gridcolor="#e2e4ed",
                zeroline=False, showline=True, linecolor="#c0c0c0",
                tickfont=dict(size=12),
            ),
            legend=dict(
                title=dict(text="<b>Clusters</b>", font=dict(size=13)),
                orientation="v",
                x=1.01, xanchor="left",
                y=0.5,  yanchor="middle",
                font=dict(size=11),
                tracegroupgap=2,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#dddddd",
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#cccccc",
                font_size=13,
            ),
            margin=dict(l=60, r=300, t=80, b=60),
        )
        fig.update_traces(cliponaxis=False)
        return fig

    def _build_treemap(self, title: str) -> go.Figure:
        """
        Treemap: one rectangle per cluster, area ∝ n_obs.
        Inside each cell: cluster label (bold) + post count.
        Keywords appear only in the hover tooltip to avoid clutter.
        Font is sized generously so text fills the available space.
        """
        total = sum(int((self.labels == k).sum()) for k in range(self.n_clusters))

        ids_tm, labels_tm, parents, values = [], [], [], []
        text_tm, colors_tm, hover_tm       = [], [], []

        # invisible root
        ids_tm.append("__root__")
        labels_tm.append(title)
        parents.append("")
        values.append(0)
        text_tm.append("")
        colors_tm.append("#ffffff")
        hover_tm.append("")

        for k in range(self.n_clusters):
            n_obs = int((self.labels == k).sum())
            pct   = 100 * n_obs / total if total else 0

            if self.llm_labels:
                label = self.llm_labels[k]["label"]
                kws   = self.llm_labels[k]["keywords"]
            else:
                label = f"Cluster {k}"
                kws   = [w for w, _ in self.keybert_keywords[k]]

            kw_hover = "<br>".join(f"• {w}" for w in kws)

            # Inside cell: label + count (no keywords → bigger font fits)
            cell_text = f"<b>{label}</b><br>{n_obs} posts · {pct:.1f}%"

            hover_text = (
                f"<b>{label}</b><br>"
                f"<span style='color:#555'>{n_obs} posts — {pct:.1f}%</span><br>"
                "──────────────<br>"
                f"<b>Keywords:</b><br>{kw_hover}"
            )

            ids_tm.append(f"cluster_{k}")
            labels_tm.append(label)
            parents.append("__root__")
            values.append(n_obs)
            text_tm.append(cell_text)
            colors_tm.append(_get_color(k))
            hover_tm.append(hover_text)

        fig = go.Figure(go.Treemap(
            ids=ids_tm,
            labels=labels_tm,
            parents=parents,
            values=values,
            text=text_tm,
            customdata=hover_tm,
            textinfo="text",
            hovertemplate="%{customdata}<extra></extra>",
            marker=dict(
                colors=colors_tm,
                line=dict(width=3, color="white"),
                pad=dict(t=28, l=8, r=8, b=8),
            ),
            textfont=dict(
                size=17,
                color="white",
                family="Arial Black, Arial, sans-serif",
            ),
            tiling=dict(
                packing="squarify",
                pad=5,
            ),
            root_color="white",
        ))

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b> — cluster overview",
                font=dict(size=20),
                x=0.5, xanchor="center",
            ),
            width=1400, height=860,
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=70, b=20),
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="#cccccc",
                font_size=14,
            ),
        )
        return fig

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return a DataFrame: cluster | n_posts | label | keywords."""
        self._check_fitted()
        rows = []
        for k in range(self.n_clusters):
            if self.llm_labels:
                label = self.llm_labels[k]["label"]
                kws = ", ".join(self.llm_labels[k]["keywords"])
            else:
                label = f"Cluster {k}"
                kws = ", ".join(w for w, _ in self.keybert_keywords[k])
            n = int((self.labels == k).sum())
            rows.append({"cluster": k, "n_posts": n, "label": label, "keywords": kws})
        return pd.DataFrame(rows)

    # ── Private pipeline steps ──────────────────────────────────────────────

    def _encode(self, texts: list[str]):
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

    # ── Clustering backends ─────────────────────────────────────────────────

    def _reduce_umap(self):
        """High-dim UMAP reduction, used only in the KMeans path."""
        reducer = umap.UMAP(**self.umap_params)
        self.X_umap = reducer.fit_transform(self.embeddings)

    def _cluster(self):
        """KMeans on the UMAP-reduced embeddings."""
        clusterer = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto")
        self.labels = clusterer.fit_predict(self.X_umap)
        self._build_df_clusters()

    def _adaptive_umap_cluster(self):
        """
        AdaptiveUMAP backend — replaces both _reduce_umap() and _cluster().

        Steps (all parameter-free from the data):
          1. Estimate intrinsic dimension d and per-point k* (dadapy binomial)
          2. Build adaptive fuzzy graph W_sym  (locally-varying neighborhood)
          3. Simplicial set embedding → (n × d) layout
          4. KMeans on the layout → labels

        Exposes:
          self.adaptive_umap   : AdaptiveUMAP instance (for inspection)
          self.X_umap          : the high-dim embedding (n × d), reused by UMAP 2D
        """
        from .adaptive_umap import AdaptiveUMAP

        au = AdaptiveUMAP(**self.adaptive_umap_params)
        self.labels = au.fit(self.embeddings, n_clusters=self.n_clusters)

        # Store for downstream use and for the 2D UMAP step
        self.adaptive_umap = au
        self.X_umap        = au.embedding    # (n × d) — already reduced

        self._build_df_clusters()

    # ── Shared cluster bookkeeping ──────────────────────────────────────────

    def _build_df_clusters(self):
        """Build df_final and df_clusters from self.texts and self.labels."""
        self.df_final = pd.DataFrame({"text": self.texts, "cluster": self.labels})

        posts_by_cluster: dict[int, list[str]] = defaultdict(list)
        for _, row in self.df_final.iterrows():
            posts_by_cluster[int(row["cluster"])].append(row["text"])

        self.df_clusters = pd.DataFrame([
            {
                "cluster": cid,
                "n_posts": len(posts),
                "merged_text": "\n\n".join(posts),
                "sample_texts": posts[: self.n_samples_for_llm],
            }
            for cid, posts in sorted(posts_by_cluster.items())
        ])

    def _extract_keywords(self):
        self.keybert_keywords = [
            self.kw_model.extract_keywords(
                self.df_clusters.loc[k, "merged_text"],
                top_n=self.top_n_keywords,
                stop_words="english",
            )
            for k in range(self.n_clusters)
        ]

    def _llm_label(self):
        """
        Send each cluster to the LLM labeler as two micro-tasks:
          Task A  →  topic label
          Task B  →  refined keywords
        Results stored in self.llm_labels.
        """
        cluster_inputs = [
            {
                "keywords": [w for w, _ in self.keybert_keywords[k]],
                "texts": self.df_clusters.loc[k, "sample_texts"],
            }
            for k in range(self.n_clusters)
        ]
        self.llm_labels = self.llm_labeler.label_batch(cluster_inputs, verbose=True)

    def _reduce_umap_2d(self):
        """
        2D layout for visualization — always uses the adaptive fuzzy graph.

        - Starts from self.embeddings (original sentence vectors, not X_umap)
        - n_components fixed to 2
        - k* reused from AdaptiveUMAP if already computed (adaptive_umap backend),
          otherwise estimated fresh from the embeddings
        """
        from .adaptive_umap import (
            _return_ids_kstar_binomial,
            _adaptive_fuzzy_graph,
            _simplicial_set_embedding,
        )

        # ── k* source ────────────────────────────────────────────────────────
        if hasattr(self, "adaptive_umap") and self.adaptive_umap.k_star is not None:
            # Already estimated during clustering — reuse, no extra cost
            k_star = self.adaptive_umap.k_star
            print("    → reusing k* from AdaptiveUMAP clustering step")
        else:
            # kmeans path: estimate k* now from the raw embeddings
            print("    → estimating k* from embeddings for 2D layout...")
            _, k_star = _return_ids_kstar_binomial(
                self.embeddings,
                n_iter=10,
                verbose=False,
            )

        # ── adaptive fuzzy graph on raw embeddings ────────────────────────────
        W_sym, _ = _adaptive_fuzzy_graph(self.embeddings, k_star)

        # ── 2D simplicial set embedding ───────────────────────────────────────
        self.embedding_2d = _simplicial_set_embedding(
            self.embeddings,
            W_sym,
            n_components=2,
            random_state=42,
        )

    def _build_hover_strings(self):
        """Build HTML hover text for each cluster, using LLM labels if available."""
        for k in range(self.n_clusters):
            if self.llm_labels:
                label = self.llm_labels[k]["label"]
                kws = self.llm_labels[k]["keywords"]
                kw_str = "<br>".join(f"• {w}" for w in kws)
                source_tag = "<i style='font-size:10px'>via LLM</i>"
            else:
                label = f"Cluster {k}"
                kws = self.keybert_keywords[k]
                kw_str = "<br>".join(f"• {w}" for w, _ in kws)
                source_tag = "<i style='font-size:10px'>via KeyBERT</i>"

            n = int((self.labels == k).sum())
            self.hover_strings[k] = (
                f"<b>{label}</b><br>"
                f"<span style='color:#888'>{n} posts</span><br>"
                "──────────────<br>"
                f"<b>Keywords:</b><br>{kw_str}<br>"
                f"{source_tag}"
            )

    def _check_fitted(self):
        if self.labels is None:
            raise RuntimeError("Call .fit(texts) before using this method.")

    # ── Dunder ──────────────────────────────────────────────────────────────

    def __repr__(self):
        status = "fitted" if self.labels is not None else "not fitted"
        llm    = type(self.llm_labeler).__name__ if self.llm_labeler else "None"
        extra  = ""
        if self.clustering_backend == "adaptive_umap" and hasattr(self, "adaptive_umap"):
            au = self.adaptive_umap
            extra = f", ID={au.intrinsic_dim:.2f}, mean_k*={au.k_star.mean():.1f}"
        return (
            f"visualize_topics(\n"
            f"  model='{self.model_name}',\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  clustering_backend='{self.clustering_backend}'{extra},\n"
            f"  llm_labeler={llm},\n"
            f"  status={status}\n"
            f")"
        )
