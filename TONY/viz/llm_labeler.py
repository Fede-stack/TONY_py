"""
llm_labeler.py
──────────────
Two LLM backends for cluster labeling with small quantized models.

Each backend splits the work into two focused micro-tasks so that
small models (≤1B params) can handle them reliably:

  Task A  →  topic definition  (3-6 word thematic label)
  Task B  →  keyword selection (5 representative keywords)

Backends
────────
  TransformersLabeler  – HuggingFace transformers + bitsandbytes (4-bit)
  MLXLabeler           – mlx-lm (Apple Silicon / MLX), Qwen3-aware

Usage
─────
  from llm_labeler import TransformersLabeler, MLXLabeler

  # Normal (CUDA / CPU)
  labeler = TransformersLabeler(model_id="Qwen/Qwen2.5-0.5B-Instruct")
  result  = labeler.label(keywords=["boss", "salary", "promotion"], texts=[...])
  # → {"label": "Workplace frustration and unfair treatment",
  #    "keywords": ["boss", "salary", "overwork", "promotion", "credit"]}

  # Apple Silicon (Qwen3 supported)
  labeler = MLXLabeler(model_id="mlx-community/Qwen3.5-0.8B-OptiQ-4bit")
  result  = labeler.label(keywords=[...], texts=[...])
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod


# ══════════════════════════════════════════════════════════════════════════════
#  1. OUTPUT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _clean_output(raw: str) -> str:
    """
    Remove all artifacts that small chat models leak into their output.

    Handles:
      - Chat template tokens : <|im_end|>  <|endoftext|>  <|eot_id|>
      - Angle-bracket tokens : <eos>  </s>  <pad>  <unk>
      - LLaMA markers        : [INST]  [/INST]  <<SYS>>
      - Markdown / quotes    : " ' ` * #
    """
    raw = re.sub(r"<\|[^|>]*\|>", "", raw)
    raw = re.sub(r"</?[a-zA-Z_][a-zA-Z_0-9]*>", "", raw)
    raw = re.sub(r"\[/?(?:INST|SYS)\]|<</?SYS>>", "", raw)
    raw = re.sub(r"[\"'`*#]", "", raw)
    return raw.strip()


def _strip_think_block(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by Qwen3 before its actual answer.

    Two cases:
      - Closed block  <think>...</think>  -> drop block, keep what follows
      - Unclosed block <think>... (EOF)   -> drop everything from <think> onward
        (budget ran out mid-thought)
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  2. KEYWORD DEDUPLICATION  (singular / plural / inflections)
# ══════════════════════════════════════════════════════════════════════════════

def _stem(word: str) -> str:
    """
    Minimal suffix-stripping stemmer (no external deps).
    Reduces common English inflections to a shared root so that
    'office' / 'offices', 'work' / 'working' are treated as duplicates.
    """
    w = word.lower().strip()
    rules = [
        ("ies",  3, "y"),
        ("ves",  3, "f"),
        ("ness", 4, ""),
        ("tion", 4, ""),
        ("ment", 4, ""),
        ("ing",  4, ""),
        ("ed",   3, ""),
        ("er",   3, ""),
        ("est",  3, ""),
        ("es",   3, ""),
        ("s",    3, ""),
    ]
    for suffix, min_root, replacement in rules:
        if w.endswith(suffix) and len(w) - len(suffix) >= min_root:
            return w[: len(w) - len(suffix)] + replacement
    return w


def _deduplicate_keywords(keywords: list[str]) -> list[str]:
    """
    Remove keywords whose stem already appears earlier in the list.
    Preserves the original form of the first occurrence.

    Example: ['office', 'offices', 'work', 'working'] -> ['office', 'work']
    """
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        s = _stem(kw)
        if s not in seen:
            seen.add(s)
            result.append(kw)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  3. PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_LABEL = """\
Your job: read some texts and write a SHORT TOPIC LABEL (3 to 6 words) \
that captures their general theme. Do NOT copy a single word from the texts. \
Write a descriptive phrase.

Example
Texts:
- I got passed over for promotion again despite great reviews
- My manager takes credit for all my ideas in meetings
- They announced record profits then froze our salaries
Topic label: workplace frustration and unfair treatment

Now do the same for these texts:
{samples}
Relevant keywords for context: {keywords}
Topic label:"""

_PROMPT_KEYWORDS = """\
Topic: {label}

Keyword list: {keywords}

From the list above pick the 5 keywords that best describe the topic. \
Write them comma-separated on a single line. Nothing else.
Keywords:"""


def _build_label_prompt(keywords: list[str], texts: list[str]) -> str:
    samples = "\n".join(f"- {t[:140]}" for t in texts[:4])
    kw_hint = ", ".join(keywords[:8])
    return _PROMPT_LABEL.format(samples=samples or "(no samples)", keywords=kw_hint)


def _build_keyword_prompt(keywords: list[str], label: str) -> str:
    return _PROMPT_KEYWORDS.format(label=label, keywords=", ".join(keywords))


# ══════════════════════════════════════════════════════════════════════════════
#  4. OUTPUT PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_label(raw: str) -> str:
    """
    Extract a clean topic label from the model reply.
    Handles echoed prompt markers, think blocks, and multi-line outputs.
    """
    raw = _strip_think_block(raw)
    raw = _clean_output(raw)

    for marker in ("topic label:", "label:", "topic:"):
        if raw.lower().startswith(marker):
            raw = raw[len(marker):].strip()
            break

    for line in raw.split("\n"):
        line = line.strip(" .-:")
        if line:
            return line[:80].capitalize()

    return "Unknown topic"


def _parse_keywords(raw: str, original: list[str]) -> list[str]:
    """
    Parse comma-separated keywords from model reply, then deduplicate
    singular/plural and inflected forms.
    Falls back to the first 5 deduplicated original keywords if parsing fails.
    """
    raw = _strip_think_block(raw)
    raw = _clean_output(raw)

    for marker in ("keywords:", "keyword:"):
        if raw.lower().startswith(marker):
            raw = raw[len(marker):].strip()
            break

    raw = raw.split("\n")[0]

    candidates = [k.strip().lower() for k in raw.split(",") if k.strip()]
    orig_lower = [o.lower() for o in original]
    kept = [c for c in candidates if any(c in o or o in c for o in orig_lower)]
    kept = _deduplicate_keywords(kept)

    if len(kept) < 5:
        extras = _deduplicate_keywords([o.lower() for o in original])
        for e in extras:
            if e not in kept:
                kept.append(e)
            if len(kept) == 5:
                break

    return kept[:5]


# ══════════════════════════════════════════════════════════════════════════════
#  5. ABSTRACT BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class BaseLLMLabeler(ABC):
    """
    Abstract base class for cluster labelers.
    Subclasses must implement _generate(prompt, max_new_tokens).
    """

    def __init__(self, max_new_tokens_label: int = 512, max_new_tokens_kw: int = 256):
        self.max_new_tokens_label = max_new_tokens_label
        self.max_new_tokens_kw = max_new_tokens_kw
        self.debug = False  # set True to print raw model output

    @abstractmethod
    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        """Run inference and return the raw generated text (new tokens only)."""
        ...

    def label(self, keywords: list[str], texts: list[str]) -> dict:
        """
        Two micro-tasks -> return {"label": str, "keywords": list[str]}.
        """
        # Task A: topic definition
        prompt_a = _build_label_prompt(keywords, texts)
        raw_label = self._generate(prompt_a, self.max_new_tokens_label)
        if self.debug:
            print(f"\n[DEBUG Task A raw] {repr(raw_label)}")
        label = _parse_label(raw_label)

        # Task B: keyword selection (uses the label from Task A as context)
        prompt_b = _build_keyword_prompt(keywords, label)
        raw_kw = self._generate(prompt_b, self.max_new_tokens_kw)
        if self.debug:
            print(f"[DEBUG Task B raw] {repr(raw_kw)}")
        refined_kw = _parse_keywords(raw_kw, keywords)

        return {"label": label, "keywords": refined_kw}

    def label_batch(self, clusters: list[dict], verbose: bool = True) -> list[dict]:
        """
        Label a list of clusters sequentially.

        Args:
            clusters: list of dicts with keys 'keywords' (list[str])
                      and 'texts' (list[str]).
        """
        results = []
        n = len(clusters)
        for i, c in enumerate(clusters):
            if verbose:
                print(f"  Labeling cluster {i+1}/{n}...", end="\r", flush=True)
            t0 = time.time()
            result = self.label(keywords=c["keywords"], texts=c.get("texts", []))
            elapsed = time.time() - t0
            if verbose:
                print(f"  [{i+1:>3}/{n}] '{result['label']}'  ({elapsed:.1f}s)")
            results.append(result)
        if verbose:
            print()
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  6. BACKEND — HuggingFace Transformers  (CUDA / CPU)
# ══════════════════════════════════════════════════════════════════════════════

class TransformersLabeler(BaseLLMLabeler):
    """
    Uses HuggingFace transformers with optional 4-bit quantization.

    Requirements:
      pip install transformers accelerate bitsandbytes
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        load_in_4bit: bool = False,
        device_map: str = "auto",
        max_new_tokens_label: int = 512,
        max_new_tokens_kw: int = 256,
    ):
        super().__init__(max_new_tokens_label, max_new_tokens_kw)
        self.model_id = model_id

        print(f"[TransformersLabeler] Loading '{model_id}' ...")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        quant_cfg = None
        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map=device_map,
            torch_dtype="auto",
        )
        self._model.eval()
        print("[TransformersLabeler] ✓ Ready")

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            encoded = self._tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = encoded.to(self._model.device)
        input_len = input_ids.shape[-1]

        stop_ids: list[int] = []
        if self._tokenizer.eos_token_id is not None:
            eos = self._tokenizer.eos_token_id
            stop_ids += eos if isinstance(eos, list) else [eos]
        for tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
            tid = self._tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self._tokenizer.unk_token_id:
                stop_ids.append(tid)
        stop_ids = list(set(stop_ids))

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=stop_ids if stop_ids else None,
                pad_token_id=stop_ids[0] if stop_ids else 0,
            )

        new_tokens = output_ids[0][input_len:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = _strip_think_block(raw)
        return _clean_output(raw)


# ══════════════════════════════════════════════════════════════════════════════
#  7. BACKEND — MLX  (Apple Silicon, Qwen3-aware)
# ══════════════════════════════════════════════════════════════════════════════

def _is_qwen3(model_id: str) -> bool:
    mid = model_id.lower()
    return "qwen3" in mid or "qwen-3" in mid


class MLXLabeler(BaseLLMLabeler):
    """
    Uses mlx-lm for inference on Apple Silicon (M1/M2/M3/M4).

    Automatically handles Qwen3's thinking mode:
      - Disables it via enable_thinking=False in apply_chat_template
        when mlx-lm >= 0.21 is installed.
      - Falls back to /no_think suffix for older versions.
      - Always strips residual <think>...</think> blocks from the output.

    Requirements:
      pip install mlx-lm
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        max_new_tokens_label: int = 512,
        max_new_tokens_kw: int = 256,
    ):
        super().__init__(max_new_tokens_label, max_new_tokens_kw)
        self.model_id = model_id
        self._qwen3 = _is_qwen3(model_id)

        print(f"[MLXLabeler] Loading '{model_id}' ...")
        if self._qwen3:
            print("[MLXLabeler] Qwen3 detected — thinking mode will be disabled.")

        from mlx_lm import load
        self._model, self._tokenizer = load(model_id)
        print("[MLXLabeler] ✓ Ready")

    def _apply_template(self, prompt: str) -> str:
        """
        Build the formatted prompt string.
        For Qwen3: tries enable_thinking=False first, then /no_think fallback.
        """
        messages = [{"role": "user", "content": prompt}]

        if self._qwen3:
            # Attempt 1: mlx-lm >= 0.21 native API
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                pass

            # Attempt 2: /no_think prompt suffix (honoured by Qwen3)
            messages = [{"role": "user", "content": prompt + " /no_think"}]

        try:
            return self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return prompt

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        from mlx_lm import generate

        formatted = self._apply_template(prompt)

        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_new_tokens,
            verbose=False,
        )

        # Strip think blocks (no-op for non-Qwen3 or when thinking is disabled)
        response = _strip_think_block(response)
        return _clean_output(response)