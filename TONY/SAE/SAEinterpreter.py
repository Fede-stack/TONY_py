import json, time, torch, urllib.request, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Literal


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

SAEName = Literal['SAE16', 'SAE32', 'SAE64']

class SAEInterpreter:
    """
    Interprets texts using one or all SAEs trained on Qwen3 embeddings.

    Parameters
    ----------
    hf_repo_id : str
        HuggingFace repository to download artifacts from.
        Default: 'FritzStack/SAE-mental-health'
    hf_token : str | None
        HuggingFace token (required for private repositories).
    artifacts_dir : str
        Local directory where artifacts are cached.
        Default: './sae_artifacts'
    openrouter_key : str | None
        OpenRouter API key for embeddings.
    embedding_model : str
        Embedding model. Default: 'qwen/qwen3-embedding-4b'
    use_huggingface : bool
        If True, uses local HuggingFace model for embeddings.
    max_freq : float
        Excludes features with freq > max_freq. Default 0.15.
    device : str
        'cpu' | 'cuda' | 'mps'. Default: auto-detect.

    Example
    -------
    interpreter = SAEInterpreter(
        hf_token       = 'hf_...',
        openrouter_key = 'sk-or-...',
    )

    # Single text, default SAE (SAE64)
    result = interpreter("I haven't left my bed in three days", sae='SAE64')

    # Specify SAE
    result = interpreter("I haven't left my bed", sae='SAE64')

    # All 3 SAEs
    result = interpreter("I haven't left my bed", sae='all')

    # Batch
    results = interpreter(["text 1", "text 2"], top_k=5)
    """

    OPENROUTER_EMBED_URL = 'https://openrouter.ai/api/v1/embeddings'
    SAE_NAMES            = ['SAE16', 'SAE32', 'SAE64']

    HF_FILES = (
        ['emb_mean.npy', 'emb_std.npy', 'global_config.json'] +
        [f'{name}_weights.pt'                  for name in ['SAE16', 'SAE32', 'SAE64']] +
        [f'{name}_config.json'                 for name in ['SAE16', 'SAE32', 'SAE64']] +
        [f'interpretable_features_{name}.json' for name in ['SAE16', 'SAE32', 'SAE64']]
    )

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
        self._load_all()

        if use_huggingface:
            self._load_hf_embedding_model()

    # ── Setup ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_device():
        if torch.cuda.is_available():         return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'

    def _download_artifacts(self):
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

    def _load_all(self):
        d = self.artifacts_dir

        # Normalisation statistics (shared across all SAEs)
        self.emb_mean = np.load(d / 'emb_mean.npy')
        self.emb_std  = np.load(d / 'emb_std.npy')

        # Load each SAE
        self._saes     = {}   # name → TopKSAE
        self._features = {}   # name → {feature_id: feat_dict}

        for name in self.SAE_NAMES:
            # Config + weights
            with open(d / f'{name}_config.json') as f:
                cfg = json.load(f)

            sae = TopKSAE(cfg['d_input'], cfg['n_latents'],
                          cfg['k'], cfg['k_aux'])
            sae.load_state_dict(
                torch.load(d / f'{name}_weights.pt', map_location='cpu')
            )
            sae.eval()
            sae.to(self.device)
            self._saes[name] = sae

            # Interpretable features
            with open(d / f'interpretable_features_{name}.json',
                      encoding='utf-8') as f:
                features = json.load(f)

            self._features[name] = {
                feat['feature_id']: feat
                for feat in features
                if feat.get('freq', 0) <= self.max_freq
            }

            print(f'  {name}: {cfg["n_latents"]} latents  k={cfg["k"]}  '
                  f'→  {len(self._features[name])} interpretable features')

        print(f'\nSAEInterpreter ready')
        print(f'  device    : {self.device}')
        print(f'  embedding : '
              f'{"HuggingFace" if self.use_huggingface else "OpenRouter"}'
              f' — {self.embedding_model}')
        print(f'  max_freq  : {self.max_freq}')

    def _load_hf_embedding_model(self):
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
                            f"Unexpected response: {list(data.keys())}"
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
                mask = enc['attention_mask'].unsqueeze(-1).float()
                embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                all_embs.append(embs.cpu().numpy())
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    def _embed(self, texts: list, batch_size: int) -> np.ndarray:
        if self.use_huggingface:
            return self._embed_huggingface(texts, batch_size)
        return self._embed_openrouter(texts, batch_size)

    # ── SAE Inference ─────────────────────────────────────────────────────

    def _run_sae(self, embs: np.ndarray, sae_name: str) -> np.ndarray:
        X   = (embs - self.emb_mean) / (self.emb_std + 1e-8)
        X_t = torch.from_numpy(X).float().to(self.device)
        with torch.no_grad():
            _, Z = self._saes[sae_name](X_t)
        return Z.cpu().numpy()

    def _top_features(self, scores: np.ndarray,
                      sae_name: str, top_k: int) -> list:
        features = self._features[sae_name]
        active   = [
            {
                'feature_id': fid,
                'score':      round(float(scores[fid]), 4),
                'label':      features[fid]['label'],
                'sae':        sae_name,
            }
            for fid in features
            if scores[fid] > 0
        ]
        active.sort(key=lambda x: x['score'], reverse=True)
        for rank, feat in enumerate(active[:top_k], 1):
            feat['rank'] = rank
        return active[:top_k]

    # ── Public API ────────────────────────────────────────────────────────

    def __call__(
        self,
        texts:      Union[str, list],
        top_k:      int              = 3,
        sae:        Union[SAEName, Literal['all']] = 'SAE64',
        batch_size: int              = 32,
    ) -> Union[dict, list]:
        """
        Parameters
        ----------
        texts      : str or list[str]
        top_k      : features to return per text per SAE (default 3)
        sae        : 'SAE16' | 'SAE32' | 'SAE64' | 'all'
                     If 'all', returns features from all 3 SAEs merged
                     and re-ranked by score. Default: 'SAE64'
        batch_size : batch size for the embedding API

        Returns
        -------
        dict           if input is str
        list[dict]     if input is list

        Dict structure (single SAE):
        {
            'text': str,
            'sae': 'SAE64',
            'features': [
                {'rank': 1, 'score': float, 'feature_id': int,
                 'label': str, 'sae': str},
                ...
            ]
        }

        Dict structure (sae='all'):
        {
            'text': str,
            'sae': 'all',
            'features': {
                'SAE16': [...],
                'SAE32': [...],
                'SAE64': [...],
            }
        }
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        sae_names = self.SAE_NAMES if sae == 'all' else [sae]

        # Compute embeddings once (shared across all SAEs)
        embs = self._embed(texts, batch_size)

        results = []
        for i, text in enumerate(texts):
            if sae == 'all':
                # Return features grouped by SAE
                features_by_sae = {}
                for name in sae_names:
                    Z      = self._run_sae(embs[i:i+1], name)
                    feats  = self._top_features(Z[0], name, top_k)
                    features_by_sae[name] = feats
                results.append({
                    'text':     text,
                    'sae':      'all',
                    'features': features_by_sae,
                })
            else:
                Z     = self._run_sae(embs[i:i+1], sae)
                feats = self._top_features(Z[0], sae, top_k)
                results.append({
                    'text':     text,
                    'sae':      sae,
                    'features': feats,
                })

        return results[0] if single else results
