import json, time, torch, urllib.request, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union


# ════════════════════════════════════════════════════════════
#  SAE MODEL
# ════════════════════════════════════════════════════════════

class TopKSAE(nn.Module):
    def __init__(self, d_input, n_latents, k, k_aux, aux_alpha=1/32):
        super().__init__()
        self.d_input, self.n_latents = d_input, n_latents
        self.k, self.k_aux, self.aux_alpha = k, k_aux, aux_alpha
        self.W_e = nn.Linear(d_input,   n_latents, bias=True)
        self.W_d = nn.Linear(n_latents, d_input,   bias=True)
        self.register_buffer('steps_since_fired',
                             torch.zeros(n_latents, dtype=torch.long))
        self._total = 0

    @staticmethod
    def _topk(x, k):
        vals, idx = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, idx, vals)
        return F.relu(out)

    def encode(self, x):
        return self._topk(self.W_e(x), self.k)

    def forward(self, x):
        h = self.encode(x)
        return self.W_d(h), h


# ════════════════════════════════════════════════════════════
#  MAIN CLASS
# ════════════════════════════════════════════════════════════

class SAEInterpreter:
    """
    Interprets texts using a SAE trained on Qwen3 embeddings.

    Parameters
    ----------
    hf_repo_id : str
        HuggingFace repository to download artifacts from.
        Default: 'FritzStack/SAE-mental-health'
        Files are downloaded once and cached in artifacts_dir.
    hf_token : str | None
        HuggingFace token (required for private repositories).
    artifacts_dir : str
        Local directory where artifacts are saved/loaded.
        Default: './sae_artifacts'
    openrouter_key : str | None
        OpenRouter API key for embeddings.
    embedding_model : str
        Embedding model. Default: 'qwen/qwen3-embedding-4b' (OpenRouter)
    use_huggingface : bool
        If True, uses local HuggingFace model for embeddings (requires GPU).
    max_freq : float
        Excludes features with freq > max_freq (too generic). Default 0.15.
    device : str
        'cpu' | 'cuda' | 'mps'. Default: auto-detect.

    Example
    -------
    interpreter = SAEInterpreter(
        hf_token       = 'hf_...',
        openrouter_key = 'sk-or-...',
    )
    result  = interpreter("I haven't left my bed in three days")
    results = interpreter(["text 1", "text 2"], top_k=5)
    """

    OPENROUTER_EMBED_URL = 'https://openrouter.ai/api/v1/embeddings'

    HF_FILES = [
        'SAE32_weights.pt',
        'sae32_config.json',
        'emb_mean.npy',
        'emb_std.npy',
        'interpretable_features.json',
    ]

    def __init__(
        self,
        hf_repo_id:      str   = 'FritzStack/SAE-mental-health',
        hf_token:        str   = None,
        artifacts_dir:   str   = './sae_artifacts',
        openrouter_key:  str   = None,
        embedding_model: str   = 'qwen/qwen3-embedding-4b',
        use_huggingface: bool  = False,
        max_freq:        float = 0.15,
        device:          str   = None,
    ):
        self.hf_repo_id      = hf_repo_id
        self.hf_token        = hf_token
        self.artifacts_dir   = Path(artifacts_dir)
        self.openrouter_key  = openrouter_key
        self.embedding_model = embedding_model
        self.use_huggingface = use_huggingface
        self.max_freq        = max_freq
        self.device          = device or self._detect_device()

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self._download_artifacts()
        self._load_artifacts()

        if use_huggingface:
            self._load_hf_embedding_model()

    # ── Setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_device():
        if torch.cuda.is_available():         return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'

    def _download_artifacts(self):
        """
        Downloads files from HuggingFace if not already cached locally.
        Already cached files are skipped.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                'huggingface_hub not installed: pip install huggingface_hub'
            )

        print(f'Downloading artifacts from {self.hf_repo_id}...')
        for fname in self.HF_FILES:
            dest = self.artifacts_dir / fname
            if dest.exists():
                print(f'  (cached) {fname}')
                continue
            print(f'  ↓ {fname}')
            hf_hub_download(
                repo_id   = self.hf_repo_id,
                filename  = fname,
                repo_type = 'model',
                token     = self.hf_token,
                local_dir = str(self.artifacts_dir),
            )
        print('  ✓ Artifacts ready\n')

    def _load_artifacts(self):
        d = self.artifacts_dir

        # Config + SAE weights
        with open(d / 'sae32_config.json') as f:
            cfg = json.load(f)

        self.sae = TopKSAE(
            cfg['d_input'], cfg['n_latents'], cfg['k'], cfg['k_aux']
        )
        self.sae.load_state_dict(
            torch.load(d / 'SAE32_weights.pt', map_location='cpu')
        )
        self.sae.eval()
        self.sae.to(self.device)

        # Normalisation statistics
        self.emb_mean = np.load(d / 'emb_mean.npy')
        self.emb_std  = np.load(d / 'emb_std.npy')

        # Interpretable features (filtered by max_freq)
        with open(d / 'interpretable_features.json', encoding='utf-8') as f:
            features = json.load(f)

        self._features = {
            feat['feature_id']: feat
            for feat in features
            if feat.get('freq', 0) <= self.max_freq
        }

        print('SAEInterpreter ready')
        print(f'  device    : {self.device}')
        print(f'  embedding : '
              f'{"HuggingFace" if self.use_huggingface else "OpenRouter"}'
              f' — {self.embedding_model}')

    def _load_hf_embedding_model(self):
        """Loads the embedding model locally from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                'transformers not installed: pip install transformers'
            )
        print(f'Loading embedding model {self.embedding_model}...')
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model, trust_remote_code=True
        )
        self._hf_model = AutoModel.from_pretrained(
            self.embedding_model,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self._hf_model.eval()
        print('  ✓ Embedding model loaded')

    # ── Embedding ─────────────────────────────────────────────────────────

    def _embed_openrouter(self, texts: list, batch_size: int) -> np.ndarray:
        if not self.openrouter_key:
            raise ValueError('openrouter_key not provided')

        headers = {
            'Authorization': f'Bearer {self.openrouter_key}',
            'Content-Type':  'application/json',
            'HTTP-Referer':  'https://github.com/sae-mh',
            'X-Title':       'SAEInterpreter',
        }
        all_embs = []
        for batch in [texts[i:i+batch_size]
                      for i in range(0, len(texts), batch_size)]:
            payload  = json.dumps({
                'model': self.embedding_model, 'input': batch
            }).encode()
            last_err = None
            for attempt in range(5):
                try:
                    req = urllib.request.Request(
                        self.OPENROUTER_EMBED_URL,
                        data=payload, headers=headers, method='POST'
                    )
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = json.loads(r.read().decode())
                    if 'error' in data:
                        raise ValueError(f"API error: {data['error']}")
                    if 'data' in data:
                        all_embs.extend(
                            [x['embedding'] for x in data['data']]
                        )
                    elif 'embeddings' in data:
                        all_embs.extend(data['embeddings'])
                    else:
                        raise ValueError(
                            f"Unexpected response structure: {list(data.keys())}"
                        )
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f'Embedding failed: {last_err}')
            time.sleep(0.1)
        return np.array(all_embs, dtype=np.float32)

    def _embed_huggingface(self, texts: list, batch_size: int) -> np.ndarray:
        all_embs = []
        with torch.no_grad():
            for batch in [texts[i:i+batch_size]
                          for i in range(0, len(texts), batch_size)]:
                enc  = self._hf_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors='pt'
                ).to(self.device)
                out  = self._hf_model(**enc)
                # Mean pooling over sequence dimension
                mask = enc['attention_mask'].unsqueeze(-1).float()
                embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    def _embed(self, texts: list, batch_size: int) -> np.ndarray:
        if self.use_huggingface:
            return self._embed_huggingface(texts, batch_size)
        return self._embed_openrouter(texts, batch_size)

    # ── SAE Inference ─────────────────────────────────────────────────────

    def _run_sae(self, embs: np.ndarray) -> np.ndarray:
        X   = (embs - self.emb_mean) / (self.emb_std + 1e-8)
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            _, Z = self.sae(X_t)
        return Z.cpu().numpy()   # (N, n_latents)

    # ── Public API ────────────────────────────────────────────────────────

    def __call__(
        self,
        texts:      Union[str, list],
        top_k:      int = 3,
        batch_size: int = 32,
    ) -> Union[dict, list]:
        """
        Parameters
        ----------
        texts      : str or list[str]
        top_k      : number of features to return per text (default 3)
        batch_size : batch size for the embedding API

        Returns
        -------
        dict           if input is str
        list[dict]     if input is list

        Dict structure:
        {
            'text': str,
            'features': [
                {'rank': 1, 'score': float, 'feature_id': int, 'label': str},
                ...
            ]
        }
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        embs = self._embed(texts, batch_size)
        Z    = self._run_sae(embs)

        results = []
        for i, text in enumerate(texts):
            scores = Z[i]
            active = [
                {
                    'feature_id': fid,
                    'score':      round(float(scores[fid]), 4),
                    'label':      self._features[fid]['label'],
                }
                for fid in self._features
                if scores[fid] > 0
            ]
            active.sort(key=lambda x: x['score'], reverse=True)
            for rank, feat in enumerate(active[:top_k], 1):
                feat['rank'] = rank
            results.append({'text': text, 'features': active[:top_k]})

        return results[0] if single else results