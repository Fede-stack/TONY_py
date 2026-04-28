"""
adaptive_umap.py
────────────────
AdaptiveUMAP: a parameter-free UMAP variant that estimates both the
neighborhood size k* and the embedding dimensionality d from the data.

Pipeline
────────
  1. ID + k* estimation   (dadapy binomial estimator, iterative)
       → intrinsic dimension d  (used as n_components)
       → per-point k*_i         (used as local neighborhood size)

  2. Adaptive fuzzy graph  (UMAP* graph with locally-varying k*)
       → symmetrised sparse weight matrix W_sym

  3. Simplicial set embedding  (UMAP layout on W_sym)
       → high-dim embedding  (shape: n × d)

  4. KMeans on the embedding
       → cluster labels

No manual hyperparameters needed: d and k* come entirely from the data.

Usage
─────
  from adaptive_umap import AdaptiveUMAP

  au = AdaptiveUMAP(n_iter=10)
  labels = au.fit(X, n_clusters=5)

  # Inspect internals
  print(au.intrinsic_dim)   # estimated ID
  print(au.k_star)          # per-point k* (shape: n,)
  print(au.embedding.shape) # (n, intrinsic_dim)
  print(au.W_sym)           # sparse adjacency (n × n)
"""

from __future__ import annotations

import warnings
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# ══════════════════════════════════════════════════════════════════════════════
#  1. ID + k* estimation
# ══════════════════════════════════════════════════════════════════════════════

def _return_ids_kstar_binomial(
    X: np.ndarray,
    n_iter: int = 10,
    initial_id: float | None = None,
    Dthr: float = 6.67,
    r: float | str = "opt",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate intrinsic dimension (ID) and per-point k* via the dadapy
    binomial estimator coupled with iterative kstar scaling.

    Args:
        X:          Input data matrix (n_samples × n_features).
        n_iter:     Number of ID estimation iterations.
        initial_id: Seed ID. If None, uses the 2NN estimator.
        Dthr:       kstar threshold (default 6.67 ≈ chi² p=0.01).
        r:          Binomial ratio. 'opt' computes the optimal value
                    r = min(0.95, 0.2032^(1/d)) at each iteration.
        verbose:    Print per-iteration diagnostics.

    Returns:
        ids    : (n_iter,) array — ID estimate at each iteration.
        kstars : (n,) int array — final per-point k* values.
    """
    from dadapy import Data

    data = Data(X)
    if initial_id is None:
        data.compute_id_2NN(algorithm="base")
    else:
        data.compute_distances()
        data.set_id(initial_id)

    ids    = np.zeros(n_iter)
    kstars = np.zeros((n_iter, data.N), dtype=int)

    for i in range(n_iter):
        data.compute_kstar(Dthr)
        if verbose:
            print(f"  iter {i:>2d} | ID = {data.intrinsic_dim:.4f} | "
                  f"mean k* = {data.kstar.mean():.1f}")

        r_eff = min(0.95, 0.2032 ** (1.0 / data.intrinsic_dim)) if r == "opt" else r
        rk = np.array([dd[data.kstar[j]] for j, dd in enumerate(data.distances)])
        rn = rk * r_eff
        n  = np.sum([dd < rn[j] for j, dd in enumerate(data.distances)], axis=1)

        id_new = np.log((n.mean() - 1) / (data.kstar.mean() - 1)) / np.log(r_eff)
        data.set_id(id_new)

        ids[i]    = id_new
        kstars[i] = data.kstar

    return ids, kstars[-1]


# ══════════════════════════════════════════════════════════════════════════════
#  2. Adaptive fuzzy graph (UMAP* with locally-varying k*)
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_fuzzy_graph(
    X: np.ndarray,
    k_star: np.ndarray,
    metric: str = "euclidean",
    n_jobs: int = 1,
    tol: float = 1e-5,
    max_iters: int = 64,
) -> tuple[sparse.csr_matrix, dict]:
    """
    Build an adaptive fuzzy simplicial set (UMAP*) with locally-varying k*_i.

    For each point i:
      - ρ_i   = distance to nearest non-zero neighbor (local connectivity)
      - σ_i   = bandwidth solved so Σ_j exp(-max(0, d_ij − ρ_i)/σ_i) = log2(k*_i)
      - w_ij  = exp(-max(0, d_ij − ρ_i)/σ_i)  for j in k*_i nearest neighbors

    The directed graph is symmetrised as:
      W_sym = W + Wᵀ − W ⊙ Wᵀ   (fuzzy union)

    Args:
        X:         Input data matrix (n × d).
        k_star:    Per-point neighborhood sizes (n,).
        metric:    sklearn-compatible distance metric.
        n_jobs:    Parallel jobs for NearestNeighbors.
        tol:       Convergence tolerance for σ binary search.
        max_iters: Max iterations for σ binary search.

    Returns:
        W_sym : (n × n) sparse CSR matrix — symmetrised fuzzy weights.
        info  : dict with keys 'rho' (n,) and 'sigma' (n,).
    """
    n     = X.shape[0]
    k_star = np.asarray(k_star, dtype=int)
    k_max  = int(k_star.max())

    # ── 1. Find k_max nearest neighbors for every point ───────────────────────
    nbrs = NearestNeighbors(n_neighbors=k_max + 1, metric=metric, n_jobs=n_jobs)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances, indices = distances[:, 1:], indices[:, 1:]  # drop self

    # ── 2. ρ_i = distance to first non-zero neighbor ──────────────────────────
    rho = np.array([d[d > 0][0] if np.any(d > 0) else 0.0 for d in distances])

    # ── 3. Binary-search σ_i ─────────────────────────────────────────────────
    def _solve_sigma(d: np.ndarray, rho_i: float, k_i: int) -> float:
        target = np.log2(max(k_i, 2))
        lo, hi = 1e-5, 10.0
        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            s   = np.exp(-np.maximum(0.0, d[:k_i] - rho_i) / mid).sum()
            if abs(s - target) < tol:
                return mid
            lo, hi = (mid, hi) if s < target else (lo, mid)
        return 0.5 * (lo + hi)

    # ── 4. Directed weights w_ij ──────────────────────────────────────────────
    sigma = np.zeros(n)
    rows, cols, vals = [], [], []

    for i in range(n):
        k_i       = max(int(k_star[i]), 1)
        d_i       = distances[i, :k_i]
        idx_i     = indices[i, :k_i]
        sigma[i]  = _solve_sigma(d_i, rho[i], k_i)
        w_i       = np.exp(-np.maximum(0.0, d_i - rho[i]) / sigma[i])
        rows     += [i] * k_i
        cols     += idx_i.tolist()
        vals     += w_i.tolist()

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    # ── 5. Fuzzy union symmetrisation ─────────────────────────────────────────
    W_sym = W + W.T - W.multiply(W.T)
    W_sym.eliminate_zeros()

    return W_sym, {"rho": rho, "sigma": sigma}


# ══════════════════════════════════════════════════════════════════════════════
#  3. Simplicial set embedding wrapper
# ══════════════════════════════════════════════════════════════════════════════

def _simplicial_set_embedding(
    X: np.ndarray,
    W_sym: sparse.csr_matrix,
    n_components: int,
    random_state: int = 0,
    n_epochs: int = 200,
    initial_alpha: float = 1.0,
    a: float = 1.576,
    b: float = 0.8951,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Thin wrapper around umap.umap_.simplicial_set_embedding.

    Passes the pre-built W_sym graph directly, bypassing UMAP's own
    neighbor search and graph construction.

    Returns:
        embedding : (n × n_components) float array.
    """
    from umap.umap_ import simplicial_set_embedding as _sse

    rng = np.random.RandomState(random_state)

    result = _sse(
        data=X,
        graph=W_sym,
        n_components=n_components,
        initial_alpha=initial_alpha,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init="spectral",
        random_state=rng,
        metric=metric,
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
    )
    # simplicial_set_embedding returns (embedding,) or (embedding, aux)
    return result[0] if isinstance(result, tuple) else result


# ══════════════════════════════════════════════════════════════════════════════
#  4. AdaptiveUMAP — public class
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveUMAP:
    """
    Parameter-free UMAP clustering.

    Both the embedding dimensionality (intrinsic dimension d) and the
    per-point neighborhood size (k*) are estimated from the data, so no
    manual hyperparameter tuning is needed.

    Args:
        n_iter:            Iterations for the ID/k* estimation loop.
        initial_id:        Seed intrinsic dimension. None → 2NN auto-estimate.
        Dthr:              kstar chi² threshold (default 6.67 ≈ p=0.01).
        r:                 Binomial ratio. 'opt' = adaptive per iteration.
        metric:            Distance metric for neighbor search.
        n_epochs:          UMAP layout optimization epochs.
        random_state:      Seed for reproducibility.
        verbose_id:        Print ID/k* estimation progress.

    Attributes (populated after fit):
        intrinsic_dim  : float  — final estimated ID.
        k_star         : ndarray (n,) — per-point neighborhood size.
        W_sym          : sparse matrix (n × n) — fuzzy weight graph.
        embedding      : ndarray (n × d) — high-dim UMAP embedding.
        labels_        : ndarray (n,) — KMeans cluster assignments.
    """

    def __init__(
        self,
        n_iter: int = 10,
        initial_id: float | None = None,
        Dthr: float = 6.67,
        r: float | str = "opt",
        metric: str = "euclidean",
        n_epochs: int = 200,
        random_state: int = 0,
        verbose_id: bool = False,
    ):
        self.n_iter       = n_iter
        self.initial_id   = initial_id
        self.Dthr         = Dthr
        self.r            = r
        self.metric       = metric
        self.n_epochs     = n_epochs
        self.random_state = random_state
        self.verbose_id   = verbose_id

        # populated by fit()
        self.intrinsic_dim: float | None = None
        self.k_star:        np.ndarray | None = None
        self.W_sym:         sparse.csr_matrix | None = None
        self.embedding:     np.ndarray | None = None
        self.labels_:       np.ndarray | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Run the full pipeline and return cluster labels.

        Args:
            X:          Data matrix (n_samples × n_features).
            n_clusters: Number of KMeans clusters.

        Returns:
            labels_ : (n,) int array of cluster assignments.
        """
        X = np.asarray(X, dtype=np.float32)

        # Step 1 — ID + k* estimation
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            ids, self.k_star = _return_ids_kstar_binomial(
                X,
                n_iter=self.n_iter,
                initial_id=self.initial_id,
                Dthr=self.Dthr,
                r=self.r,
                verbose=self.verbose_id,
            )
        self.intrinsic_dim = float(ids[-1])
        n_components = max(2, int(round(self.intrinsic_dim)))

        print(f"    → estimated ID = {self.intrinsic_dim:.2f}  "
              f"| embedding dim = {n_components}  "
              f"| mean k* = {self.k_star.mean():.1f}")

        # Step 2 — adaptive fuzzy graph
        self.W_sym, self._graph_info = _adaptive_fuzzy_graph(
            X,
            self.k_star,
            metric=self.metric,
        )

        # Step 3 — simplicial set embedding
        self.embedding = _simplicial_set_embedding(
            X,
            self.W_sym,
            n_components=n_components,
            random_state=self.random_state,
            n_epochs=self.n_epochs,
            metric=self.metric,
        )

        # Step 4 — KMeans on the embedding
        km = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        self.labels_ = km.fit_predict(self.embedding)

        return self.labels_

    def __repr__(self) -> str:
        fitted = self.labels_ is not None
        return (
            f"AdaptiveUMAP("
            f"n_iter={self.n_iter}, "
            f"metric='{self.metric}', "
            f"fitted={fitted}"
            + (f", ID={self.intrinsic_dim:.2f}" if fitted else "")
            + ")"
        )
