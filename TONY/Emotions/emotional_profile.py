import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional


class EmotionProfiler:
    """
    Computes an aggregated emotional profile over a list of documents
    using the RACLETTE causal language model.

    Each document is scored against a set of emotion labels via
    joint log-probability estimation; probabilities are normalised
    with softmax and then averaged across all documents.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: 'FritzStack/RACLETTE-fp16').
    emotions : list[str] | None
        Emotion labels to score. Falls back to the 32-emotion default list.
    load_in_4bit : bool
        Whether to apply 4-bit NF4 quantisation (default: True).
    device : str | None
        Target device. If None, auto-selects CUDA when available.
    prompt_template : str
        Template used to wrap each sentence.
        Must contain exactly one ``{text}`` placeholder.
    """

    MODEL_NAME = "FritzStack/RACLETTE-fp16"
    PROMPT_TEMPLATE = "<|prompter|>{text}<|endoftext|><|emotion|>"
    DEFAULT_EMOTIONS = [
        "surprised", "excited", "angry", "proud", "sad", "annoyed", "grateful", "lonely",
        "afraid", "terrified", "guilty", "impressed", "disgusted", "hopeful", "confident",
        "furious", "anxious", "anticipating", "joyful", "nostalgic", "disappointed",
        "prepared", "jealous", "content", "devastated", "embarrassed", "caring",
        "sentimental", "trusting", "ashamed", "apprehensive", "faithful"
    ]

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        emotions: Optional[list[str]] = None,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
        prompt_template: str = PROMPT_TEMPLATE,
    ):
        self.emotions = emotions if emotions is not None else self.DEFAULT_EMOTIONS
        self.prompt_template = prompt_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.tokenizer = self._load_model(model_name, load_in_4bit)
        self.emotion_token_ids = self._build_token_id_map()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(
        self, model_name: str, load_in_4bit: bool
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load model and tokenizer, optionally with 4-bit quantisation."""
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def _build_token_id_map(self) -> dict[str, list[int]]:
        """Map each emotion label to its (possibly multi-token) token IDs."""
        return {
            emotion: self.tokenizer.encode(emotion, add_special_tokens=False)
            for emotion in self.emotions
        }

    def _emotion_log_prob(self, prompt: str, token_ids: list[int]) -> float:
        """
        Compute the joint log-probability of a token sequence given a prompt:

            log P(t1, t2, …, tn | prompt) = Σ_i log P(ti | prompt, t1…t_{i-1})
        """
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = prompt_ids.shape[1]

        full_ids = torch.cat(
            [prompt_ids, torch.tensor([token_ids], device=self.device)], dim=1
        )

        with torch.no_grad():
            outputs = self.model(full_ids)

        logits = outputs.logits[0]  # (seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        total_log_prob = 0.0
        for i, tid in enumerate(token_ids):
            pos = prompt_len - 1 + i  # position predicting current token
            total_log_prob += log_probs[pos, tid].item()

        return total_log_prob

    def _predict_probs(self, text: str) -> dict[str, float]:
        """
        Return softmax-normalised emotion probabilities for a single text.
        """
        prompt = self.prompt_template.format(text=text)

        log_probs = {
            emotion: self._emotion_log_prob(prompt, tids)
            for emotion, tids in self.emotion_token_ids.items()
        }

        emotions = list(log_probs.keys())
        lp_tensor = torch.tensor([log_probs[e] for e in emotions])
        probs_tensor = F.softmax(lp_tensor, dim=0)

        return {e: probs_tensor[i].item() for i, e in enumerate(emotions)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_emotional_profile(self, documents: list[str]) -> dict[str, float]:
        """
        Compute the average emotional profile over a list of documents.

        Each document is scored independently; the resulting probability
        distributions are averaged to produce the final profile.

        Parameters
        ----------
        documents : list[str]
            Input texts (sentences, paragraphs, full documents, etc.).

        Returns
        -------
        dict[str, float]
            Mapping from emotion label to mean probability (sums to ~1).

        Raises
        ------
        ValueError
            If ``documents`` is empty.
        """
        if not documents:
            raise ValueError("`documents` must contain at least one entry.")

        profile: dict[str, float] = {e: 0.0 for e in self.emotions}

        for doc in documents:
            probs = self._predict_probs(doc)
            for emotion, prob in probs.items():
                profile[emotion] += prob

        n = len(documents)
        return {e: v / n for e, v in profile.items()}

    def top_emotions(
        self, documents: list[str], k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Return the ``k`` most probable emotions for the given documents.

        Parameters
        ----------
        documents : list[str]
            Input texts.
        k : int
            Number of top emotions to return (default: 5).

        Returns
        -------
        list[tuple[str, float]]
            Sorted list of (emotion, probability) pairs, descending by probability.
        """
        profile = self.build_emotional_profile(documents)
        sorted_emotions = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:k]
