import torch
import torch.nn as nn
import torch.nn.functional as F

# Token Linear for Weight Extrapolation.
class TokenLinearGL(nn.Module):
    def __init__(self, in_features, out_features, num_tokens):
        super().__init__()        
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        
        # Create frequency mixture input
        positions = torch.linspace(0, 1, num_tokens).unsqueeze(-1)  # [num_tokens, 1]
        freqs = torch.linspace(1, 10, 32).unsqueeze(0)  # [1, 8] different frequencies
        angles = positions * freqs * 2 * math.pi  # [num_tokens, 8]
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [num_tokens, 16]
        self.register_buffer('tokens', pos_enc)
        
        # MLP projections from frequency space
        self.key_up = nn.Sequential(
            nn.Linear(16, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, in_features, bias=False)
        )
        self.value_up = nn.Sequential(
            nn.Linear(16, 64, bias=False), 
            nn.GELU(),
            nn.Linear(64, out_features, bias=False)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.key_up[0].weight, std=0.02)
        nn.init.normal_(self.key_up[2].weight, std=0.02)
        nn.init.normal_(self.value_up[0].weight, std=0.02)
        nn.init.normal_(self.value_up[2].weight, std=0.02)
        
    def transform(self, attn_map):
        out = (attn_map * (attn_map.size(-1) ** 0.5)) / torch.sqrt(attn_map.square().sum(-1, keepdim=True))
        return torch.nn.functional.gelu(out)

    def forward(self, x, token_mask=None):
        # Project from frequency space to key/value space
        key_tokens = self.key_up(self.tokens)
        value_tokens = self.value_up(self.tokens)
        
        if token_mask is not None:
            key_tokens = key_tokens[token_mask]
            value_tokens = value_tokens[token_mask]

        # Standard attention mechanism
        similarity = torch.matmul(x, key_tokens.transpose(-2, -1))
        weights = self.transform(similarity)
        return torch.matmul(weights, value_tokens)



class TokenLinearSM(nn.Module):
    def __init__(self, in_features, out_features, num_tokens):
        super().__init__()
        assert in_features % 2 == 0, f"in_features must be even, got {in_features}"
        
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        
        # Create frequency mixture input
        positions = torch.linspace(0, 1, num_tokens).unsqueeze(-1)  # [num_tokens, 1]
        freqs = torch.linspace(1, 10, 8).unsqueeze(0)  # [1, 8] different frequencies
        angles = positions * freqs * 2 * math.pi  # [num_tokens, 8]
        pos_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [num_tokens, 16]
        self.register_buffer('tokens', pos_enc)
        
        # Simple linear projections from frequency space
        self.key_up = nn.Linear(16, in_features, bias=False)
        self.value_up = nn.Linear(16, out_features, bias=False)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.key_up.weight, std=0.02)
        nn.init.normal_(self.value_up.weight, std=0.02)

    def softmax(self, attn_map):
        scaled = attn_map / math.sqrt(self.in_features)
        return F.softmax(scaled, dim=-1)

    def forward(self, x, token_mask=None):
        # Project from frequency space to key/value space
        key_tokens = self.key_up(self.tokens)
        value_tokens = self.value_up(self.tokens)
        
        if token_mask is not None:
            key_tokens = key_tokens[token_mask]
            value_tokens = value_tokens[token_mask]

        # Standard attention mechanism
        similarity = torch.matmul(x, key_tokens.transpose(-2, -1))
        weights = self.softmax(similarity)
        return torch.matmul(weights, value_tokens)
