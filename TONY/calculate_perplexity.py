from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Optional heavy imports – resolved lazily inside __init__ so the file can be
# imported on machines that have only one of the two backends installed.
# ---------------------------------------------------------------------------


def _is_mlx_model(model_name: str, backend: Optional[str]) -> bool:
    """
    Heuristic to decide which backend to use.

    Priority:
    1. Explicit ``backend`` argument ('mlx' | 'hf' | 'auto').
    2. Model name / path contains 'mlx' (common convention on HF Hub,
       e.g. "mlx-community/Mistral-7B-Instruct-v0.3-4bit").
    3. Fall back to HuggingFace.
    """
    if backend is not None:
        backend = backend.lower()
        if backend == "mlx":
            return True
        if backend in ("hf", "huggingface", "transformers"):
            return False
        # backend == 'auto' → fall through to heuristic
    return "mlx" in model_name.lower()


class PerplexityExtractor:
    """
    Extract perplexity from a causal language model.

    Supports two backends:
    - **HuggingFace Transformers** (default for most models)
    - **MLX / mlx-lm** (Apple Silicon; selected automatically when the model
      name contains ``"mlx"`` or when ``backend="mlx"`` is passed explicitly)

    Parameters
    ----------
    model_name : str
        HuggingFace model name / local path.
    device : str, optional
        ``"cuda"`` or ``"cpu"`` (ignored for MLX, which always runs on Apple
        Silicon via the Metal backend).
    backend : str, optional
        ``"mlx"``, ``"hf"`` (or ``"huggingface"`` / ``"transformers"``), or
        ``"auto"`` (default).  When ``"auto"``, the class inspects
        ``model_name`` for the substring ``"mlx"`` to decide.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        backend: Optional[str] = "auto",
    ):
        self.model_name = model_name
        self.backend = "mlx" if _is_mlx_model(model_name, backend) else "hf"

        if self.backend == "mlx":
            self._init_mlx()
        else:
            self._init_hf(device)

    # ── HuggingFace ────────────────────────────────────────────────────

    def _init_hf(self, device: Optional[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── MLX ────────────────────────────────────────────────────────────

    def _init_mlx(self) -> None:
        try:
            from mlx_lm import load as mlx_load
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for MLX models. "
                "Install it with:  pip install mlx-lm"
            ) from exc

        # mlx_lm.load returns (model, tokenizer)
        self.model, self.tokenizer = mlx_load(self.model_name)
        self.device = "mlx"  # informational only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity for a single text.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        float
            Perplexity score.
        """
        if self.backend == "mlx":
            return self._perplexity_mlx(str(text))
        return self._perplexity_hf(str(text))

    def compute_perplexity_batch(
        self, texts: list[str], batch_size: int = 8
    ) -> list[float]:
        """
        Compute perplexity for a list of texts.

        Parameters
        ----------
        texts : list of str
            Input texts.
        batch_size : int
            Number of texts per mini-batch (used only for the HuggingFace
            backend; MLX processes each text individually).

        Returns
        -------
        list of float
            Perplexity score for each text.
        """
        perplexities: list[float] = []
        for i in range(0, len(texts), batch_size):
            for text in texts[i : i + batch_size]:
                perplexities.append(self.compute_perplexity(text))
        return perplexities

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # ── HuggingFace ────────────────────────────────────────────────────

    def _perplexity_hf(self, text: str) -> float:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        return math.exp(loss.item())

    # ── MLX ────────────────────────────────────────────────────────────

    def _perplexity_mlx(self, text: str) -> float:
        """
        Compute perplexity using an MLX causal LM.

        The model is called once with the token sequence; the per-token
        cross-entropy loss is averaged over *all* positions (teacher-forcing
        / next-token-prediction objective, identical to the HF approach).
        """
        import mlx.core as mx
        import mlx.nn as nn

        # ── Tokenise ───────────────────────────────────────────────────
        # mlx-lm tokenisers follow the HuggingFace PreTrainedTokenizer API.
        token_ids: list[int] = self.tokenizer.encode(text)
        if len(token_ids) < 2:
            # Need at least one input and one target token.
            return float("inf")

        tokens = mx.array(token_ids)[None]  # shape (1, seq_len)

        # ── Forward pass ───────────────────────────────────────────────
        # MLX causal LM models expose __call__(tokens, cache=None) → logits
        # with shape (batch, seq_len, vocab_size).
        logits = self.model(tokens)          # (1, seq_len, vocab_size)
        mx.eval(logits)                       # materialise before slicing

        # ── Shift for next-token prediction ────────────────────────────
        # Input  tokens: [t0, t1, …, t_{n-1}]
        # Target tokens: [t1, t2, …, t_n    ]
        shift_logits = logits[:, :-1, :]     # (1, seq_len-1, vocab_size)
        shift_labels = tokens[:, 1:]         # (1, seq_len-1)

        # Flatten to (seq_len-1, vocab_size) and (seq_len-1,)
        flat_logits = shift_logits.reshape(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.reshape(-1)

        # ── Cross-entropy loss ─────────────────────────────────────────
        # nn.losses.cross_entropy expects (N, C) logits + (N,) integer labels
        per_token_loss = nn.losses.cross_entropy(
            flat_logits, flat_labels, reduction="none"
        )
        mean_loss = mx.mean(per_token_loss)
        mx.eval(mean_loss)

        return math.exp(mean_loss.item())

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PerplexityExtractor("
            f"model='{self.model_name}', "
            f"backend='{self.backend}', "
            f"device='{self.device}')"
        )
