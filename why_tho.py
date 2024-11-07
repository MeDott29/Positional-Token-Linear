import torch
import torch.nn as nn
import torch.nn.functional as F

# Why stop there?
class TransformerLinear(nn.Module):
    def __init__(self, in_features, out_features, num_tokens):
        super().__init__()
        assert in_features % 2 == 0, f"in_features must be even, got {in_features}"
        
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        

        # Mini-transformer parameters
        self.hidden_dim = 32
        self.num_heads = 2
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_layers = 12
        self.z_dim = 16
        
        # Token processing layers
        self.token_proj = nn.Linear(self.z_dim, self.hidden_dim, bias=False)
        
        # Create transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.ModuleDict({
                    'q_proj': nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    'k_proj': nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    'v_proj': nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                    'out_proj': nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
                }),
                'norm1': nn.LayerNorm(self.hidden_dim),
                'norm2': nn.LayerNorm(self.hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim, bias=False)
                )
            }) for _ in range(self.num_layers)
        ])
        
        # Output projections
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, in_features, bias=True)
        self.value_proj = nn.Linear(self.hidden_dim, out_features, bias=True)
        
        self._init_weights()

    def _init_weights(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        
        self.token_proj.apply(_init_weight)
        for layer in self.transformer_layers:
            layer['attention']['q_proj'].apply(_init_weight)
            layer['attention']['k_proj'].apply(_init_weight)
            layer['attention']['v_proj'].apply(_init_weight)
            layer['attention']['out_proj'].apply(_init_weight)
            layer['mlp'].apply(_init_weight)
        self.key_proj.apply(_init_weight)
        self.value_proj.apply(_init_weight)

    def transformer_block(self, x):
        B = x.size(0) if x.dim() > 2 else 1
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        # Project tokens
        h = self.token_proj(x)  # [B, num_tokens, hidden_dim]
        
        # Process through transformer layers
        for layer in self.transformer_layers:
            # Multi-head attention
            residual = h
            h = layer['norm1'](h)
            
            q = layer['attention']['q_proj'](h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer['attention']['k_proj'](h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = layer['attention']['v_proj'](h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Scaled dot-product attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            
            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.hidden_dim)
            h = residual + layer['attention']['out_proj'](attn_output)
            
            # MLP
            residual = h
            h = layer['norm2'](h)
            h = residual + layer['mlp'](h)
        
        # Final layer norm
        h = self.final_norm(h)
        
        if B == 1:
            h = h.squeeze(0)
        return h

    def forward(self, x, token_mask=None):
        # Process frequency tokens through transformer
        z_tokens = torch.randn(self.num_tokens, self.z_dim, device=x.device, dtype=x.dtype)
        processed_tokens = self.transformer_block(z_tokens)
        
        if token_mask is not None:
            processed_tokens = processed_tokens[token_mask]
        
        # Project to key/value space
        key_tokens = self.key_proj(processed_tokens)
        value_tokens = self.value_proj(processed_tokens)

        # Standard attention mechanism with input
        similarity = torch.matmul(x, key_tokens.transpose(-2, -1))
        weights = F.softmax(similarity / math.sqrt(self.in_features), dim=-1)
        return torch.matmul(weights, value_tokens)
