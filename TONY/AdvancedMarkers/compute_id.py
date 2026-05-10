"""
intrinsic_dimension.py  –  AdvancedMarkers sub-module
======================================================
Estimates the intrinsic dimension of a list of representations
using the ABIDE algorithm (binomial estimator + kstar scale selection).

Supported encoding backends
----------------------------
* SentenceTransformers  (local, default)
* OpenRouter            (remote, any model that exposes /v1/embeddings)
* Pre-computed          (pass numpy arrays directly – no encoding step)

Typical usage
-------------
>>> from TONYpy.TONY.AdvancedMarkers.intrinsic_dimension import IntrinsicDimensionFinder

>>> # 1. from raw texts  ─  local model
>>> idf = IntrinsicDimensionFinder(backend="sentence_transformers")
>>> result = idf.fit(["doc1", "doc2", ...])
>>> print(result.id, result.kstars)

>>> # 2. from raw texts  ─  OpenRouter
>>> idf = IntrinsicDimensionFinder(
...     backend="openrouter",
...     model_name="openai/text-embedding-3-small",
...     api_key="sk-or-..."
... )
>>> result = idf.fit(["doc1", "doc2", ...])

>>> # 3. from pre-computed embeddings
>>> idf = IntrinsicDimensionFinder(backend="precomputed")
>>> result = idf.fit(embedding_matrix)   # np.ndarray  (N, D)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import requests

# ── lazy import: SentenceTransformers is optional ────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ── suppress HuggingFace verbosity ───────────────────────────────────────────
try:
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
except ImportError:
    pass

from ABIDE import ABIDE   


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntrinsicDimensionResult:
    """
    Container for the output of :meth:`IntrinsicDimensionFinder.fit`.

    Attributes
    ----------
    id : float
        Final intrinsic dimension estimate (last ABIDE iteration).
    id_history : np.ndarray, shape (n_iter,)
        ID estimate at each iteration.
    kstars : np.ndarray, shape (N,)
        Per-point kstar values at the last iteration.
    embeddings : np.ndarray, shape (N, D)
        The embedding matrix actually fed to ABIDE.
    model_name : str
        Name of the model used for encoding (empty for precomputed).
    backend : str
        Encoding backend used: ``"sentence_transformers"``,
        ``"openrouter"``, or ``"precomputed"``.
    """
    id: float
    id_history: np.ndarray
    kstars: np.ndarray
    embeddings: np.ndarray
    model_name: str
    backend: str
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"IntrinsicDimensionResult("
            f"id={self.id:.4f}, "
            f"n_points={len(self.kstars)}, "
            f"backend='{self.backend}', "
            f"model='{self.model_name}')"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class IntrinsicDimensionFinder:
    """
    Estimates the intrinsic dimension of a collection of representations.

    Parameters
    ----------
    backend : str
        Encoding backend to use.  One of:
        - ``"sentence_transformers"`` – local HuggingFace/ST model (default)
        - ``"openrouter"``            – remote call to OpenRouter /v1/embeddings
        - ``"precomputed"``           – skip encoding; pass raw embeddings to fit()
    model_name : str, optional
        Model identifier.
        • SentenceTransformers default : ``"sentence-transformers/all-MiniLM-L6-v2"``
        • OpenRouter examples          : ``"openai/text-embedding-3-small"``,
                                         ``"mistralai/mistral-embed"``
    api_key : str, optional
        OpenRouter API key.  Falls back to the ``OPENROUTER_API_KEY``
        environment variable when not provided explicitly.
    n_iter : int
        Number of ABIDE iterations (default 10).
    initial_id : float or None
        Starting ID for ABIDE.  ``None`` triggers the 2NN estimator.
    Dthr : float
        Threshold for the kstar test (default 6.67).
    batch_size : int
        Batch size for OpenRouter requests (default 64).  Has no effect
        on SentenceTransformers, which handles batching internally.
    verbose : bool
        Whether ABIDE should print per-iteration info (default False).
    """

    _BACKENDS = {"sentence_transformers", "openrouter", "precomputed"}

    def __init__(
        self,
        backend: str = "sentence_transformers",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        n_iter: int = 10,
        initial_id: Optional[float] = None,
        Dthr: float = 6.67,
        batch_size: int = 64,
        verbose: bool = False,
    ) -> None:
        if backend not in self._BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from {self._BACKENDS}."
            )
        self.backend = backend
        self.model_name = model_name or self._default_model(backend)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.n_iter = n_iter
        self.initial_id = initial_id
        self.Dthr = Dthr
        self.batch_size = batch_size
        self.verbose = verbose

        # cache for the local ST model (loaded once on first call)
        self._st_model: Optional["SentenceTransformer"] = None

    # ── public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        representations: Union[List[str], np.ndarray],
        Dthr: Optional[float] = None,
    ) -> IntrinsicDimensionResult:
        """
        Encode *representations* (if needed) and estimate the intrinsic dimension.

        Parameters
        ----------
        representations : list[str] or np.ndarray
            • list of strings   → encoded by the configured backend
            • np.ndarray (N, D) → used directly (requires backend="precomputed")
        Dthr : float, optional
            Threshold for the kstar test.  When provided, overrides the value
            set at construction time for this single call only.
            If omitted, the instance default (``self.Dthr``) is used.

        Returns
        -------
        IntrinsicDimensionResult
        """
        dthr_eff = Dthr if Dthr is not None else self.Dthr

        embeddings = self._encode(representations)

        abide = ABIDE(embeddings, self.initial_id, self.n_iter)
        id_history, kstars = abide.return_ids_kstar_binomial(
            Dthr=dthr_eff,
            verbose=self.verbose,
        )

        return IntrinsicDimensionResult(
            id=float(id_history[-1]),
            id_history=id_history,
            kstars=kstars,
            embeddings=embeddings,
            model_name=self.model_name,
            backend=self.backend,
            extra={"Dthr": dthr_eff},
        )

    def fit_multiple(
        self,
        representation_groups: List[Union[List[str], np.ndarray]],
        Dthr: Optional[float] = None,
    ) -> List[IntrinsicDimensionResult]:
        """
        Convenience wrapper: call :meth:`fit` on each group independently.

        Parameters
        ----------
        representation_groups : list of (list[str] or np.ndarray)
            Each element is a separate collection of representations.
        Dthr : float, optional
            Threshold forwarded to every :meth:`fit` call.
            Overrides the instance default for this call only.

        Returns
        -------
        list[IntrinsicDimensionResult]
        """
        return [self.fit(group, Dthr=Dthr) for group in representation_groups]

    # ── encoding ─────────────────────────────────────────────────────────────

    def _encode(
        self, representations: Union[List[str], np.ndarray]
    ) -> np.ndarray:
        """Route to the correct encoding backend."""
        if isinstance(representations, np.ndarray):
            if self.backend != "precomputed":
                warnings.warn(
                    "Received a numpy array but backend is not 'precomputed'. "
                    "Treating as pre-computed embeddings.",
                    UserWarning,
                    stacklevel=3,
                )
            return representations.astype(np.float64)

        # ── list of strings ───────────────────────────────────────────────────
        if self.backend == "precomputed":
            raise TypeError(
                "backend='precomputed' expects a numpy array, not a list of strings."
            )
        if self.backend == "sentence_transformers":
            return self._encode_sentence_transformers(representations)
        if self.backend == "openrouter":
            return self._encode_openrouter(representations)

        raise ValueError(f"Unknown backend: {self.backend}")

    def _encode_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
        if self._st_model is None:
            self._st_model = SentenceTransformer(self.model_name)
        embeddings = self._st_model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float64)

    def _encode_openrouter(self, texts: List[str]) -> np.ndarray:
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Pass api_key= or set "
                "the OPENROUTER_API_KEY environment variable."
            )
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model_name, "input": batch},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            batch_embeddings = np.array(
                [item["embedding"] for item in data["data"]], dtype=np.float64
            )
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_model(backend: str) -> str:
        defaults = {
            "sentence_transformers": "sentence-transformers/all-MiniLM-L6-v2",
            "openrouter": "openai/text-embedding-3-small",
            "precomputed": "",
        }
        return defaults[backend]

    def __repr__(self) -> str:
        return (
            f"IntrinsicDimensionFinder("
            f"backend='{self.backend}', "
            f"model='{self.model_name}', "
            f"n_iter={self.n_iter})"
        )
