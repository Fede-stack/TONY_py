import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import torch.nn as nn
import plotly.graph_objects as go
import plotly.colors as pc

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering monosemantic features

    Based on: "Sparse Autoencoders Find Highly Interpretable Features in Language Models"

    Architecture:
        Encoder: c = ReLU(Mx + b)
        Decoder: x_hat = M^T c  (tied weights)

    Loss: L(x) = ||x - x_hat||² + α||c||₁

    Args:
        d_in: input dimension (e.g., 1024 for Qwen3-Embedding)
        d_hidden: feature dictionary size (d_in * R, where R is the overcompleteness ratio)
        tied_weights: if True, the decoder uses the transpose of the encoder (default, as in the paper)
    """
    
    def __init__(
        self, 
        d_in: int, 
        d_hidden: int, 
        tied_weights: bool = True
    ):
        super().__init__()
        
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.tied_weights = tied_weights
        
        # Encoder: M (d_hidden x d_in) + bias b (d_hidden)
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        
        # Decoder: solo se weights non tied
        if not tied_weights:
            self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        
        # Inizializzazione weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with kaimang uniform"""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight)
    
    def normalize_encoder_weights(self):
        """
        Normalize row-wise encoder weights
        
        """
        with torch.no_grad():
            # Calcola norma di ogni riga (ogni feature)
            norms = torch.norm(self.encoder.weight, dim=1, keepdim=True)
            # Normalizza dividendo per la norma
            self.encoder.weight.div_(norms + 1e-8)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: input activations, shape [batch_size, d_in]
        
        Returns:
            x_hat: reconstruction, shape [batch_size, d_in]
            c: sparse coefficients, shape [batch_size, d_hidden]
        """
        # Encoder: c = ReLU(Mx + b)
        c = torch.relu(self.encoder(x))
        
        # Decoder: x_hat = M^T c
        if self.tied_weights:
            #
            x_hat = torch.matmul(c, self.encoder.weight.t())
        else:
            # decoder separate
            x_hat = self.decoder(c)
        
        return x_hat, c
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        x_hat: torch.Tensor, 
        c: torch.Tensor, 
        alpha: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and its components
        
        Loss: L(x) = ||x - x_hat||² + α||c||₁
        
        Args:
            x: original input
            x_hat: reconstruction
            c: sparse coefficients
            alpha: sparsity coefficient
        
        Returns:
            Dict containing 'total', 'reconstruction', 'sparsity', and 'mean_active_features'
        """
        # Reconstruction loss: MSE
        recon_loss = torch.mean((x - x_hat) ** 2)
        
        #Sparsity loss: L1 norm
        sparsity_loss = torch.mean(torch.abs(c))
        
        #Total loss
        total_loss = recon_loss + alpha * sparsity_loss
        
        #metric: how many features active on average
        mean_active = torch.mean((c > 0).float().sum(dim=-1))
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'sparsity': sparsity_loss,
            'mean_active_features': mean_active
        }
    
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Obtain only the feature activations (without reconstruction)
        Useful for post-training analysis
        
        Args:
            x: input, shape [batch_size, d_in]
        
        Returns:
            c: feature activations, shape [batch_size, d_hidden]
        """

        with torch.no_grad():
            c = torch.relu(self.encoder(x))
        return c
    
    def __repr__(self):
        return (f"SparseAutoencoder(\n"
                f"  d_in={self.d_in},\n"
                f"  d_hidden={self.d_hidden},\n"
                f"  tied_weights={self.tied_weights},\n"
                f"  parameters={sum(p.numel() for p in self.parameters()):,}\n"
                f")")


class SAEInterpreter:
    """
    Interpreter for text using Sparse Autoencoder
    
    Pipeline:
    1. Text → Embedding (SentenceTransformer)
    2. Embedding → Sparse features (SAE)
    3. Sparse features → Significant interpretations
    """
    
    def __init__(
        self,
        sae_checkpoint_path: Optional[str] = None,
        feature_stats_path: Optional[str] = None,
        feature_interpretations_path: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = 'mps',
        threshold_mode: str = 'moderate'
    ):
        self.device = device
        self.threshold_mode = threshold_mode
        
        # Risolve i path: se non specificati, usa i file nella cartella del package
        _SAE_DIR = Path(__file__).parent
        
        sae_checkpoint_path = Path(sae_checkpoint_path) if sae_checkpoint_path \
            else _SAE_DIR / "sae_best.pt"
        feature_stats_path = Path(feature_stats_path) if feature_stats_path \
            else _SAE_DIR / "feature_global_stats.json"
        feature_interpretations_path = Path(feature_interpretations_path) if feature_interpretations_path \
            else _SAE_DIR / "features_labels_qwen.json"
    
        print("Loading models...")
        
        # Load sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model.to(device)
        print(f"✓ Embedding model loaded: {embedding_model_name}")
        
        # Load trained sparse autoencoder
        checkpoint = torch.load(sae_checkpoint_path, map_location=device)
        self.sae = SparseAutoencoder(
            d_in=checkpoint['config']['d_in'],
            d_hidden=checkpoint['config']['d_hidden'],
            tied_weights=True
        )
        self.sae.load_state_dict(checkpoint['sae_state_dict'])
        self.sae.to(device)
        self.sae.eval()
        print(f"✓ SAE loaded: {checkpoint['config']['d_in']} → {checkpoint['config']['d_hidden']}")
        
        # Load global feature statistics
        with open(feature_stats_path, 'r') as f:
            self.feature_stats = json.load(f)
        print(f"✓ Feature stats loaded: {len(self.feature_stats)} features")
        
        # Load feature interpretations if available
        self.feature_interpretations = None
        if feature_interpretations_path.exists():
            with open(feature_interpretations_path, 'r') as f:
                self.feature_interpretations = json.load(f)
            print(f"✓ Interpretations loaded: {len(self.feature_interpretations)} features")

    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text string
        
        Returns:
            embedding: numpy array [embedding_dim]
        """
        with torch.no_grad():
            embedding = self.embedding_model.encode(
                [text],
                convert_to_tensor=False,
                show_progress_bar=False
            )
        return embedding[0]
    
    def get_sparse_features(self, embedding: np.ndarray) -> np.ndarray:
        """
        Extract sparse features from embedding using SAE
        
        Args:
            embedding: numpy array [embedding_dim]
        
        Returns:
            sparse_features: numpy array [num_features]
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            emb_tensor = torch.from_numpy(embedding).float().unsqueeze(0).to(self.device)
            # Get sparse activations from SAE
            sparse = self.sae.get_feature_activations(emb_tensor)
            return sparse[0].cpu().numpy()
    
    def get_significant_features(
        self,
        sparse_features: np.ndarray,
        top_k: Optional[int] = None,
        min_activation: float = 0.0
    ) -> List[Dict]:
        """
        Filter significant features based on thresholds
        
        Args:
            sparse_features: array [num_features]
            top_k: if specified, return only top-K features (ignores threshold)
            min_activation: minimum absolute activation value
        
        Returns:
            List of dicts with significant feature info
        """
        significant = []
        
        for feat_idx, activation in enumerate(sparse_features):
            # Skip if activation too low
            if activation <= min_activation:
                continue
            
            # Get feature statistics
            stats = self.feature_stats[str(feat_idx)]
            
            # Determine threshold based on mode
            if self.threshold_mode == 'conservative':
                threshold = stats['threshold_conservative']  # 75th percentile
            elif self.threshold_mode == 'moderate':
                threshold = stats['threshold_moderate']      # median
            else:  # permissive
                threshold = stats['threshold_permissive']    # mean - 0.5*std
            
            # Check if activation is significant
            is_significant = activation > threshold
            
            # Calculate percentile rank
            if stats['max'] > stats['min']:
                percentile = (activation - stats['min']) / (stats['max'] - stats['min']) * 100
            else:
                percentile = 50.0
            
            # Build feature info dict
            feature_info = {
                'feature_idx': feat_idx,
                'activation': float(activation),
                'threshold': threshold,
                'is_significant': is_significant,
                'percentile_rank': percentile,
                'mean': stats['mean'],
                'std': stats['std'],
                'total_activations': stats['total_activations']
            }
            
            
            if self.feature_interpretations and str(feat_idx) in self.feature_interpretations:
                feature_info['interpretation'] = self.feature_interpretations[str(feat_idx)]
            
            significant.append(feature_info)
        
        # Sort by activation (highest first)
        significant.sort(key=lambda x: x['activation'], reverse=True)
        
        # Return top-k or filter by significance
        if top_k is not None:
            return significant[:top_k]
        else:
            return [f for f in significant if f['is_significant']]
    
    def interpret(
        self,
        text: str,
        top_k: int = 10,
        return_embedding: bool = False,
        return_sparse: bool = False
    ) -> Dict:
        """
        Complete pipeline: text → embedding → sparse → interpretation
        
        Args:
            text: Text to interpret
            top_k: Number of features to return
            return_embedding: If True, include original embedding
            return_sparse: If True, include all sparse features
        
        Returns:
            Dictionary with complete results
        """
        # Step 1: Text to embedding
        embedding = self.encode_text(text)
        
        # Step 2: Embedding to sparse features
        sparse_features = self.get_sparse_features(embedding)
        
        # Step 3: Extract significant features
        significant_features = self.get_significant_features(
            sparse_features,
            top_k=top_k
        )
        
        # Build result dictionary
        result = {
            'text': text,
            'num_active_features': int(np.sum(sparse_features > 0)),
            'num_significant_features': len(significant_features),
            'significant_features': significant_features
        }
        
        # Add optional returns
        if return_embedding:
            result['embedding'] = embedding.tolist()
        if return_sparse:
            result['sparse_features'] = sparse_features.tolist()
        
        return result
    
    def batch_interpret(
        self,
        texts: List[str],
        top_k: int = 10,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Interpret multiple texts in batch
        
        Args:
            texts: List of texts to interpret
            top_k: Features to return per text
            show_progress: Show progress bar
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        # Setup iterator
        iterator = texts
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Interpreting texts")
        
        # Process each text
        for text in iterator:
            result = self.interpret(text, top_k=top_k)
            results.append(result)
        
        return results
    
    def print_interpretation(self, result: Dict, max_features: int = 10):
        """Print interpretation in readable format"""
        print("="*80)
        print("TEXT INTERPRETATION")
        print("="*80)
        print(f"Text: {result['text']}...")
        print(f"\nActive features: {result['num_active_features']}")
        print(f"Significant features: {result['num_significant_features']}")
        print("\n" + "-"*80)
        print("TOP FEATURES")
        print("-"*80)
        
        for i, feat in enumerate(result['significant_features'][:max_features], 1):
            print(f"\n{i}. Feature #{feat['feature_idx']}")
            print(f"   Activation: {feat['activation']:.4f} (threshold: {feat['threshold']:.4f})")
            print(f"   Percentile: {feat['percentile_rank']:.1f}%")
            print(f"   Stats: mean={feat['mean']:.4f}, std={feat['std']:.4f}")
            
            if 'interpretation' in feat:
                print(f"   Interpretation: {feat['interpretation'][:]}...")
        
        print("\n" + "="*80)

    def plot_interpretation(self, result: Dict, max_features: int = 10, renderer: str = "colab"):
        """Plot interpretation as interactive Plotly horizontal bar chart."""
        import plotly.graph_objects as go
        import plotly.colors as pc

        features        = list(reversed(result["significant_features"][:max_features]))
        feature_labels  = [f"#{f['feature_idx']}" for f in features]
        activations     = [f["activation"]         for f in features]
        means           = [f["mean"]               for f in features]
        stds            = [f["std"]                for f in features]
        percentiles     = [f["percentile_rank"]     for f in features]
        interpretations = [f.get("interpretation", "N/A") for f in features]

        min_act = min(percentiles)
        max_act = max(percentiles)

        bar_labels = [
            f"  {a:.3f}  μ={m:.3f}  σ={s:.3f}  p={p:.0f}%"
            for a, m, s, p in zip(activations, means, stds, percentiles)
        ]

        # Interpretazione wrappata a ~60 caratteri per riga
        def wrap(text, width=60):
            words = text.split()
            lines, line = [], []
            for w in words:
                if sum(len(x)+1 for x in line) + len(w) > width:
                    lines.append(" ".join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(" ".join(line))
            return "<br>".join(lines)

        hover_texts = [
            f"<b>Feature {fl}</b><br>"
            f"Activation: {a:.4f}  |  Percentile: {p:.1f}%<br>"
            f"Mean: {m:.4f}  |  Std: {s:.4f}<br>"
            f"─────────────────────────────<br>"
            f"{wrap(interp)}"
            for fl, a, p, m, s, interp in zip(
                feature_labels, activations, percentiles, means, stds, interpretations
            )
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=percentiles,
            y=feature_labels,
            orientation="h",
            text=bar_labels,
            textposition="outside",
            textfont=dict(size=11, color="#141414", family="monospace"),
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                color=activations,
                colorscale="Blues",
                cmin=min_act,
                cmax=max_act,
                colorbar=dict(
                    title="Act.",
                    tickformat=".2f",
                    thickness=12,
                    len=0.5,
                ),
                line_width=0,
            ),
        ))

        fig.update_layout(
            title=dict(
                text=(
                    "SAE Significant Features — Activation Score<br>"
                    f"<span style='font-size:12px;font-weight:normal;color:#555;'>"
                    f"Active: {result['num_active_features']} | "
                    f"Significant: {result['num_significant_features']} | "
                    f"Text: \"{result['text'][:70]}\"</span>"
                ),
                font=dict(size=16),
                x=0.0,
                xanchor="left",
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            showlegend=False,
            hoverlabel=dict(
                font_size=12,
                namelength=-1,
                bgcolor="white",
                bordercolor="#cccccc",
                font_family="monospace",
                align="left",
            ),
            margin=dict(t=100, b=60, l=80, r=200),
            height=500,
            width=1050,
            bargap=0.3,
        )

        fig.update_xaxes(
            title_text="Activation",
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=True,
            zerolinecolor="#cccccc",
            tickformat=".2f",
            tickfont=dict(size=12),
            title_font=dict(size=13),
            range=[0, max_act * 1.85],
        )
        fig.update_yaxes(
            title_text="Feature ID",
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=13),
            title_font=dict(size=13),
        )
        fig.update_traces(cliponaxis=False)
        fig.show()
