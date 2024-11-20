import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenLinearFusion(nn.Module):
    def __init__(self, in_features, out_features, num_tokens, mode='hybrid'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tokens = num_tokens
        self.mode = mode
        
        # Shared dimensions
        self.z_dim = 16
        self.hidden_dim = 32
        
        # Create frequency basis (like TokenLinearGL/SM)
        positions = torch.linspace(0, 1, num_tokens).unsqueeze(-1)
        freqs = torch.linspace(1, 10, self.z_dim//2).unsqueeze(0)
        angles = positions * freqs * 2 * math.pi
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        self.register_buffer('freq_tokens', pos_enc)
        
        # Random basis (like TransformerLinear)
        self.rand_tokens = nn.Parameter(torch.randn(num_tokens, self.z_dim) * 0.02)
        
        # Token processing
        self.token_mixer = nn.Sequential(
            nn.Linear(self.z_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.z_dim)
        )
        
        # Lightweight transformer for token refinement
        self.attention = nn.MultiheadAttention(
            embed_dim=self.z_dim,
            num_heads=2,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.z_dim)
        self.norm2 = nn.LayerNorm(self.z_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim * 4),
            nn.GELU(),
            nn.Linear(self.z_dim * 4, self.z_dim)
        )
        
        # Output projections
        self.key_proj = nn.Linear(self.z_dim, in_features, bias=False)
        self.value_proj = nn.Linear(self.z_dim, out_features, bias=False)
        
        # GELU-based attention (from TokenLinearGL)
        self.use_gelu_attention = nn.Parameter(torch.tensor(0.5))
        
    def get_tokens(self):
        if self.mode == 'frequency':
            return self.freq_tokens
        elif self.mode == 'random':
            return self.rand_tokens
        else:  # hybrid
            # Combine and refine both token types
            combined = torch.cat([self.freq_tokens, self.rand_tokens], dim=-1)
            tokens = self.token_mixer(combined)
            
            # Refine through attention
            residual = tokens
            attn_out, _ = self.attention(tokens, tokens, tokens)
            tokens = residual + attn_out
            tokens = self.norm1(tokens)
            
            # MLP refine
            residual = tokens
            tokens = self.mlp(tokens)
            tokens = residual + tokens
            tokens = self.norm2(tokens)
            
            return tokens
            
    def attention_fusion(self, similarity):
        # Interpolate between GELU and Softmax attention
        gelu_weights = F.gelu(similarity * similarity.size(-1)**0.5)
        gelu_weights = gelu_weights / torch.sqrt(gelu_weights.square().sum(-1, keepdim=True))
        
        softmax_weights = F.softmax(similarity / math.sqrt(self.in_features), dim=-1)
        
        alpha = torch.sigmoid(self.use_gelu_attention)
        return alpha * gelu_weights + (1 - alpha) * softmax_weights

    def forward(self, x, token_mask=None):
        # Get processed tokens
        tokens = self.get_tokens()
        
        if token_mask is not None:
            tokens = tokens[token_mask]
            
        # Project to key/value space
        key_tokens = self.key_proj(tokens)
        value_tokens = self.value_proj(tokens)
        
        # Compute attention with fusion mechanism
        similarity = torch.matmul(x, key_tokens.transpose(-2, -1))
        weights = self.attention_fusion(similarity)
        
        return torch.matmul(weights, value_tokens)

# Test the fusion model
def test_fusion():
    batch_size = 32
    in_features = 64
    out_features = 128
    num_tokens = 16
    
    # Create test data
    x = torch.randn(batch_size, in_features)
    
    # Test different modes
    for mode in ['frequency', 'random', 'hybrid']:
        model = TokenLinearFusion(in_features, out_features, num_tokens, mode=mode)
        out = model(x)
        
        print(f"\n{mode.upper()} Mode Test:")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {out.shape}")
        print(f"Output stats - Mean: {out.mean():.3f}, Std: {out.std():.3f}")
        
        # Test with token mask
        mask = torch.ones(num_tokens, dtype=torch.bool)
        mask[::2] = False  # Use only every other token
        out_masked = model(x, mask)
        print(f"Masked output shape: {out_masked.shape}")

if __name__ == "__main__":
    test_fusion()