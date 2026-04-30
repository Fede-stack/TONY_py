"""
visualize_topics.py
───────────────────
Pipeline: encode → [KMeans | AdaptiveUMAP] → keywords → LLM label → UMAP 2D → Plotly.

Clustering backends (clustering_backend=):
  'kmeans'         — encode → UMAP (high-dim) → KMeans   [default]
  'adaptive_umap'  — encode → AdaptiveUMAP (ID estimation + k* neighbors
                     + adaptive fuzzy graph + simplicial embedding → KMeans)

Keyword backends (keyword_backend=):
  'keybert'  [default]
      For each cluster:
        1. Selects the nr_repr_docs documents closest to the cluster centroid
           (cosine similarity on sentence embeddings) — no text concatenation.
        2. Extracts keyword candidates from each representative document
           individually via KeyBERT, preserving syntactic structure.
        3. Embeds all candidate words and scores them via cosine similarity
           against the mean embedding of the representative documents.
        4. Returns the top_n_keywords most semantically coherent with the cluster.

  'ctfidf'
      BERTopic-style class-based TF-IDF on merged cluster documents.
      Controlled via vectorizer_params (min_df, max_df, max_features, …).

In both cases keywords are optionally refined by an LLM labeler that also
assigns a human-readable topic name to each cluster.

Usage
─────
  from llm_labeler import MLXLabeler
  from AdaptiveSpectralClustering import AdaptiveSpectralClustering

  labeler = MLXLabeler("mlx-community/Qwen3.5-0.8B-OptiQ-4bit")

  # Default: KeyBERT + KMeans
  model = visualize_topics(n_clusters=5, llm_labeler=labeler)

  # c-TF-IDF keywords + AdaptiveUMAP clustering
  model = visualize_topics(
      n_clusters=5,
      clustering_backend="adaptive_umap",
      keyword_backend="ctfidf",
      vectorizer_params={"min_df": 2, "max_df": 0.9, "max_features": 20_000},
      llm_labeler=labeler,
  )

  model.fit(texts)
  model.plot()
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from .llm_labeler import BaseLLMLabeler


# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

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
    cut = label[:max_chars].rfind(" ")
    if cut <= 0:
        cut = max_chars
    return label[:cut] + "<br>" + label[cut:].strip()


# ──────────────────────────────────────────────────────────────────────────────
#  c-TF-IDF  (BERTopic style)
# ──────────────────────────────────────────────────────────────────────────────

def _ctfidf(
    cluster_docs: list[str],
    top_n: int = 15,
    vectorizer_params: dict | None = None,
) -> list[list[tuple[str, float]]]:
    """
    BERTopic-style class-based TF-IDF.

    Parameters
    ----------
    cluster_docs : list[str]
        One merged string per cluster (all texts in the cluster concatenated).
    top_n : int
        Number of top keywords to return per cluster.
    vectorizer_params : dict | None
        Kwargs forwarded to CountVectorizer (e.g. min_df, max_df, max_features,
        token_pattern). stop_words defaults to "english" if not provided.

    Returns
    -------
    list[list[tuple[str, float]]]
        For each cluster: [(word, score), ...] sorted by score descending.

    Formula
    -------
        tf_c   = count(t, c) / |words in c|
        idf_c  = log(1 + m / sum_c count(t, c))
        score  = tf_c * idf_c

    where m = number of clusters.
    """
    m = len(cluster_docs)
    params = {"stop_words": "english", **(vectorizer_params or {})}

    # ── 1. raw term counts per cluster ────────────────────────────────────────
    vectorizer = CountVectorizer(**params)
    X = vectorizer.fit_transform(cluster_docs)      # (m × V) sparse int matrix
    vocab = np.array(vectorizer.get_feature_names_out())

    # ── 2. TF  (term freq normalised by cluster length) ───────────────────────
    words_per_cluster = np.asarray(X.sum(axis=1), dtype=float).flatten()   # (m,)
    words_per_cluster = np.where(words_per_cluster == 0, 1, words_per_cluster)
    tf = X.toarray() / words_per_cluster[:, None]                           # (m × V)

    # ── 3. IDF  (class-level, BERTopic formula) ───────────────────────────────
    df  = np.asarray((X > 0).sum(axis=0), dtype=float).flatten()            # (V,)
    idf = np.log(1 + m / (df + 1e-9))                                       # (V,)

    # ── 4. c-TF-IDF scores ───────────────────────────────────────────────────
    scores = tf * idf[None, :]                                              # (m × V)

    # ── 5. top-n per cluster ─────────────────────────────────────────────────
    results: list[list[tuple[str, float]]] = []
    for c in range(m):
        row     = scores[c]
        top_idx = np.argpartition(row, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        results.append([(vocab[i], float(row[i])) for i in top_idx])

    return results


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
    Pipeline: encode → [KMeans | AdaptiveUMAP] → keywords → LLM label → UMAP 2D → Plotly.

    Args
    ────
    n_clusters : int
        Number of KMeans clusters (both clustering backends).
    model_name : str
        SentenceTransformer model name or path.
    umap_params : dict | None
        UMAP kwargs for the high-dim reduction step (KMeans path only).
    top_n_keywords : int
        Keywords to return per cluster (both keyword backends).
    device : str | None
        SentenceTransformer device ('cpu', 'cuda', 'mps').
    llm_labeler : BaseLLMLabeler | None
        Optional LLM labeler. If None the LLM step is skipped and raw keywords
        are used as-is for labels and hover text.
    n_samples_for_llm : int
        Max representative texts per cluster sent to the LLM.
    clustering_backend : str
        'kmeans' (default) or 'adaptive_umap'.
    adaptive_umap_params : dict | None
        Kwargs forwarded to AdaptiveUMAP.
    keyword_backend : str
        'keybert' (default) or 'ctfidf'.
    nr_repr_docs : int
        Number of representative documents per cluster used by the KeyBERT
        backend (selected by cosine similarity to the cluster centroid).
    vectorizer_params : dict | None
        Kwargs forwarded to CountVectorizer (c-TF-IDF backend only).
        Defaults to {"stop_words": "english"}.
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
        keyword_backend: str = "keybert",
        nr_repr_docs: int = 5,
        vectorizer_params: dict | None = None,
    ):
        if clustering_backend not in ("kmeans", "adaptive_umap"):
            raise ValueError(
                f"clustering_backend must be 'kmeans' or 'adaptive_umap', "
                f"got '{clustering_backend}'"
            )
        if keyword_backend not in ("keybert", "ctfidf"):
            raise ValueError(
                f"keyword_backend must be 'keybert' or 'ctfidf', "
                f"got '{keyword_backend}'"
            )

        self.n_clusters           = n_clusters
        self.model_name           = model_name
        self.top_n_keywords       = top_n_keywords
        self.llm_labeler          = llm_labeler
        self.n_samples_for_llm    = n_samples_for_llm
        self.clustering_backend   = clustering_backend
        self.adaptive_umap_params = adaptive_umap_params or {}
        self.keyword_backend      = keyword_backend
        self.nr_repr_docs         = nr_repr_docs
        self.vectorizer_params    = {"stop_words": "english", **(vectorizer_params or {})}

        self.umap_params = umap_params or {
            "n_neighbors": 15,
            "n_components": 10,
            "min_dist": 0.1,
            "metric": "cosine",
            "random_state": 0,
            "n_jobs": -1,
        }

        # Internal state
        self.texts: list[str]                 = []
        self.embeddings                       = None
        self.X_umap                           = None
        self.embedding_2d                     = None
        self.labels                           = None
        self.df_final: pd.DataFrame | None    = None
        self.df_clusters: pd.DataFrame | None = None
        self.cluster_keywords: list | None    = None
        self.llm_labels: list[dict] | None    = None
        self.hover_strings: dict[int, str]    = {}

        print("Loading models...")
        self.encoder = SentenceTransformer(model_name, device=device)
        print(f"  ✓ SentenceTransformer : {model_name}")

        if keyword_backend == "keybert":
            self.kw_model = KeyBERT()
            print( "  ✓ Keyword extractor   : KeyBERT (repr-doc cosine)")
        else:
            self.kw_model = None
            print( "  ✓ Keyword extractor   : c-TF-IDF (BERTopic style)")

        if llm_labeler:
            print(f"  ✓ LLM labeler         : {type(llm_labeler).__name__}")

        print(f"  ✓ Clustering backend  : {clustering_backend}")

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, texts: list[str]) -> "visualize_topics":
        self.texts   = texts
        is_adaptive  = self.clustering_backend == "adaptive_umap"
        n_llm        = 1 if self.llm_labeler else 0
        # kmeans path:        encode, umap_high, cluster, kw, [llm], umap_2d → 5 or 6
        # adaptive_umap path: encode, adaptive,           kw, [llm], umap_2d → 4 or 5
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
        kw_label = "KeyBERT" if self.keyword_backend == "keybert" else "c-TF-IDF"
        print(f"[{step}/{total_steps}] Extracting keywords with {kw_label}...")
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
        fig_scatter = self._build_scatter(title)
        fig_treemap = self._build_treemap(title)
        fig_scatter.show()
        fig_treemap.show()
        return fig_scatter, fig_treemap

    def get_cluster_summary(self) -> pd.DataFrame:
        """Return a DataFrame: cluster | n_posts | label | keywords."""
        self._check_fitted()
        rows = []
        for k in range(self.n_clusters):
            if self.llm_labels:
                label = self.llm_labels[k]["label"]
                kws   = ", ".join(self.llm_labels[k]["keywords"])
            else:
                label = f"Cluster {k}"
                kws   = ", ".join(w for w, _ in self.cluster_keywords[k])
            n = int((self.labels == k).sum())
            rows.append({"cluster": k, "n_posts": n, "label": label, "keywords": kws})
        return pd.DataFrame(rows)

    # ── Private — visualisation ─────────────────────────────────────────────

    def _build_scatter(self, title: str) -> go.Figure:
        """
        Clean scatter plot: one point per post, colored by cluster.
        Cluster identity lives in the legend and in the hover tooltip.
        """
        fig = go.Figure()

        for k in range(self.n_clusters):
            mask        = self.labels == k
            pts         = self.embedding_2d[mask]
            raw_texts   = self.df_final.loc[mask, "text"].fillna("").tolist()
            previews    = [t[:130] + ("…" if len(t) > 130 else "") for t in raw_texts]
            color       = _get_color(k)
            legend_name = self.llm_labels[k]["label"] if self.llm_labels else f"Cluster {k}"

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
        Keywords appear only in the hover tooltip.
        """
        total = sum(int((self.labels == k).sum()) for k in range(self.n_clusters))

        ids_tm, labels_tm, parents, values = [], [], [], []
        text_tm, colors_tm, hover_tm       = [], [], []

        # invisible root
        ids_tm.append("__root__"); labels_tm.append(title)
        parents.append(""); values.append(0)
        text_tm.append(""); colors_tm.append("#ffffff"); hover_tm.append("")

        for k in range(self.n_clusters):
            n_obs = int((self.labels == k).sum())
            pct   = 100 * n_obs / total if total else 0

            if self.llm_labels:
                label = self.llm_labels[k]["label"]
                kws   = self.llm_labels[k]["keywords"]
            else:
                label = f"Cluster {k}"
                kws   = [w for w, _ in self.cluster_keywords[k]]

            kw_hover   = "<br>".join(f"• {w}" for w in kws)
            cell_text  = f"<b>{label}</b><br>{n_obs} posts · {pct:.1f}%"
            hover_text = (
                f"<b>{label}</b><br>"
                f"<span style='color:#555'>{n_obs} posts — {pct:.1f}%</span><br>"
                f"──────────────<br>"
                f"<b>Keywords:</b><br>{kw_hover}"
            )

            ids_tm.append(f"cluster_{k}"); labels_tm.append(label)
            parents.append("__root__");    values.append(n_obs)
            text_tm.append(cell_text);     colors_tm.append(_get_color(k))
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
            tiling=dict(packing="squarify", pad=5),
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
            hoverlabel=dict(bgcolor="white", bordercolor="#cccccc", font_size=14),
        )
        return fig

    # ── Private — pipeline ──────────────────────────────────────────────────

    def _encode(self, texts: list[str]):
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)

    # ── Clustering backends ─────────────────────────────────────────────────

    def _reduce_umap(self):
        """High-dim UMAP reduction — KMeans path only."""
        reducer     = umap.UMAP(**self.umap_params)
        self.X_umap = reducer.fit_transform(self.embeddings)

    def _cluster(self):
        """KMeans on UMAP-reduced embeddings."""
        clusterer   = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto")
        self.labels = clusterer.fit_predict(self.X_umap)
        self._build_df_clusters()

    def _adaptive_umap_cluster(self):
        """
        AdaptiveUMAP backend — replaces both _reduce_umap() and _cluster().

        Steps (all parameter-free from the data):
          1. Estimate intrinsic dimension d and per-point k* (dadapy binomial).
          2. Build adaptive fuzzy graph W_sym (locally-varying neighborhood).
          3. Simplicial set embedding → (n × d) layout.
          4. KMeans on the layout → labels.
        """
        from .adaptive_umap import AdaptiveUMAP

        au          = AdaptiveUMAP(**self.adaptive_umap_params)
        self.labels = au.fit(self.embeddings, n_clusters=self.n_clusters)

        self.adaptive_umap = au
        self.X_umap        = au.embedding   # (n × d), reused by UMAP 2D step

        self._build_df_clusters()

    # ── Cluster bookkeeping ─────────────────────────────────────────────────

    def _build_df_clusters(self):
        """Build df_final and df_clusters from self.texts and self.labels."""
        self.df_final = pd.DataFrame({"text": self.texts, "cluster": self.labels})

        posts_by_cluster: dict[int, list[str]] = defaultdict(list)
        for _, row in self.df_final.iterrows():
            posts_by_cluster[int(row["cluster"])].append(row["text"])

        self.df_clusters = pd.DataFrame([
            {
                "cluster":      cid,
                "n_posts":      len(posts),
                "merged_text":  "\n\n".join(posts),
                "sample_texts": posts[: self.n_samples_for_llm],
            }
            for cid, posts in sorted(posts_by_cluster.items())
        ])

    # ── Keyword backends ────────────────────────────────────────────────────

    def _extract_keywords(self):
        if self.keyword_backend == "keybert":
            self._extract_keywords_keybert()
        else:
            self._extract_keywords_ctfidf()

    def _extract_keywords_keybert(self):
        """
        KeyBERT backend (primary).

        For each cluster:
          1. Compute the cluster centroid from sentence embeddings.
          2. Select the nr_repr_docs documents closest to the centroid
             (cosine similarity) — no text concatenation.
          3. Extract keyword candidates from each representative document
             individually via KeyBERT, preserving syntactic structure.
          4. Embed all unique candidates and score them via cosine similarity
             against the mean embedding of the representative documents.
          5. Return the top_n_keywords most semantically coherent candidates.
        """
        results: list[list[tuple[str, float]]] = []

        for k in range(self.n_clusters):
            mask          = self.labels == k
            cluster_embs  = self.embeddings[mask]
            cluster_texts = [t for t, m in zip(self.texts, mask) if m]

            # ── 1. centroid ────────────────────────────────────────────────
            centroid      = cluster_embs.mean(axis=0)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

            # ── 2. cosine similarity to centroid ───────────────────────────
            norms   = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
            normed  = cluster_embs / np.where(norms == 0, 1, norms)
            sims    = normed @ centroid_norm

            n_repr  = min(self.nr_repr_docs, len(cluster_texts))
            top_idx = np.argsort(sims)[-n_repr:][::-1]

            repr_docs = [cluster_texts[i] for i in top_idx]
            repr_embs = cluster_embs[top_idx]

            # ── 3. keyword candidates (one doc at a time) ──────────────────
            candidates: list[str] = []
            for doc in repr_docs:
                kws = self.kw_model.extract_keywords(
                    doc,
                    top_n=self.top_n_keywords,
                    stop_words="english",
                )
                candidates.extend(w for w, _ in kws)

            # deduplicate preserving order
            candidates = list(dict.fromkeys(candidates))

            if not candidates:
                results.append([])
                continue

            # ── 4. embed candidates ────────────────────────────────────────
            cand_embs = self.encoder.encode(candidates, show_progress_bar=False)

            # ── 5. cosine vs mean of representative embeddings ─────────────
            repr_mean      = repr_embs.mean(axis=0)
            repr_mean_norm = repr_mean / (np.linalg.norm(repr_mean) + 1e-9)

            cand_norms  = np.linalg.norm(cand_embs, axis=1, keepdims=True)
            cand_normed = cand_embs / np.where(cand_norms == 0, 1, cand_norms)
            scores      = cand_normed @ repr_mean_norm

            # ── 6. top_n_keywords ──────────────────────────────────────────
            top_n    = min(self.top_n_keywords, len(candidates))
            best_idx = np.argsort(scores)[-top_n:][::-1]
            results.append([(candidates[i], float(scores[i])) for i in best_idx])

        self.cluster_keywords = results

    def _extract_keywords_ctfidf(self):
        """c-TF-IDF backend (alternative): merged cluster docs → BERTopic scoring."""
        merged_docs = self.df_clusters["merged_text"].tolist()
        self.cluster_keywords = _ctfidf(
            merged_docs,
            top_n=self.top_n_keywords,
            vectorizer_params=self.vectorizer_params,
        )

    # ── LLM labeling ────────────────────────────────────────────────────────

    def _llm_label(self):
        """
        Send each cluster to the LLM labeler as two micro-tasks:
          Task A → topic label
          Task B → refined keywords
        Results stored in self.llm_labels.
        """
        cluster_inputs = [
            {
                "keywords": [w for w, _ in self.cluster_keywords[k]],
                "texts":    self.df_clusters.loc[k, "sample_texts"],
            }
            for k in range(self.n_clusters)
        ]
        self.llm_labels = self.llm_labeler.label_batch(cluster_inputs, verbose=True)

    # ── 2D layout ───────────────────────────────────────────────────────────

    def _reduce_umap_2d(self):
        """
        2D layout for visualization using the adaptive fuzzy graph.

        - Operates on self.embeddings (original sentence vectors).
        - k* is reused from AdaptiveUMAP if already computed, otherwise
          estimated fresh from the raw embeddings (KMeans path).
        """
        from .adaptive_umap import (
            _adaptive_fuzzy_graph,
            _return_ids_kstar_binomial,
            _simplicial_set_embedding,
        )

        if hasattr(self, "adaptive_umap") and self.adaptive_umap.k_star is not None:
            k_star = self.adaptive_umap.k_star
            print("    → reusing k* from AdaptiveUMAP clustering step")
        else:
            print("    → estimating k* from embeddings for 2D layout...")
            _, k_star = _return_ids_kstar_binomial(
                self.embeddings, n_iter=10, verbose=False,
            )

        W_sym, _          = _adaptive_fuzzy_graph(self.embeddings, k_star)
        self.embedding_2d = _simplicial_set_embedding(
            self.embeddings, W_sym, n_components=2, random_state=0,
        )

    # ── Hover strings ───────────────────────────────────────────────────────

    def _build_hover_strings(self):
        """Build HTML hover text for each cluster."""
        if self.keyword_backend == "keybert":
            source_tag = "<i style='font-size:10px'>via KeyBERT</i>"
        else:
            source_tag = "<i style='font-size:10px'>via c-TF-IDF</i>"

        for k in range(self.n_clusters):
            if self.llm_labels:
                label      = self.llm_labels[k]["label"]
                kws        = self.llm_labels[k]["keywords"]
                kw_str     = "<br>".join(f"• {w}" for w in kws)
                source_tag = "<i style='font-size:10px'>via LLM</i>"
            else:
                label  = f"Cluster {k}"
                kw_str = "<br>".join(f"• {w}" for w, _ in self.cluster_keywords[k])

            n = int((self.labels == k).sum())
            self.hover_strings[k] = (
                f"<b>{label}</b><br>"
                f"<span style='color:#888'>{n} posts</span><br>"
                f"──────────────<br>"
                f"<b>Keywords:</b><br>{kw_str}<br>"
                f"{source_tag}"
            )

    # ── Guards & dunder ─────────────────────────────────────────────────────

    def _check_fitted(self):
        if self.labels is None:
            raise RuntimeError("Call .fit(texts) before using this method.")

    def __repr__(self):
        status = "fitted" if self.labels is not None else "not fitted"
        llm    = type(self.llm_labeler).__name__ if self.llm_labeler else "None"
        extra  = ""
        if self.clustering_backend == "adaptive_umap" and hasattr(self, "adaptive_umap"):
            au    = self.adaptive_umap
            extra = f", ID={au.intrinsic_dim:.2f}, mean_k*={au.k_star.mean():.1f}"
        return (
            f"visualize_topics(\n"
            f"  model='{self.model_name}',\n"
            f"  n_clusters={self.n_clusters},\n"
            f"  clustering_backend='{self.clustering_backend}'{extra},\n"
            f"  keyword_backend='{self.keyword_backend}',\n"
            f"  llm_labeler={llm},\n"
            f"  status={status}\n"
            f")"
        )
