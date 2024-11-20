#
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import time
import math
from scipy.special import sph_harm

class TokenLinearVisualizer:
    def __init__(self, in_features: int, out_features: int, num_tokens: int):
        self.in_features = in_features
        self.out_features = out_features
        self.num_tokens = num_tokens
        
        # Enhanced character sets for different aspects
        self.token_chars = "🔴🟡🟢🔵🟣⚪️⚫️🟤"  # Token activation
        self.attention_chars = "█▓▒░ "  # Attention weights
        self.frequency_chars = "∎∙∘□ "  # Frequency components
        
    def visualize_token_activation(self, tokens: torch.Tensor, width: int = 50):
        """Visualize token activation patterns"""
        # Normalize token activations
        normalized = F.softmax(tokens.mean(dim=-1), dim=-1)
        
        print("\n\033[36m=== Token Activation Patterns ===\033[0m")
        for i, activation in enumerate(normalized):
            level = int(activation.item() * (len(self.token_chars) - 1))
            bar = self.token_chars[level] * int(activation.item() * width)
            print(f"Token {i:2d}: [{bar:<{width}}] {activation.item():.3f}")
            
    def visualize_attention_flow(self, q: torch.Tensor, k: torch.Tensor, frame_count: int = 30):
        """Animate attention flow between query and key spaces"""
        print("\n\033[33m=== Attention Flow Animation ===\033[0m")
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        attention = F.softmax(scores, dim=-1)
        
        for frame in range(frame_count):
            # Create animated flow pattern
            flow = ""
            for i in range(attention.size(0)):
                weight = attention[i].max().item()
                idx = int(weight * (len(self.attention_chars) - 1))
                char = self.attention_chars[idx]
                
                # Add pulsing effect based on frame
                pulse = abs(math.sin(frame * 0.2 + i * 0.5))
                width = int(weight * 20 * pulse) + 1
                flow += char * width
                
            print(f"\r{flow}", end="", flush=True)
            time.sleep(0.1)
        print()
        
    def visualize_frequency_mixing(self, freq_tokens: torch.Tensor):
        """Visualize frequency components in token space"""
        print("\n\033[32m=== Frequency Component Mixing ===\033[0m")
        
        # Generate harmonic basis
        t = np.linspace(0, 2*np.pi, 100)
        harmonics = []
        
        for l in range(2):  # Use first few spherical harmonics
            for m in range(-l, l+1):
                if l == 0 and m == 0:
                    # Y00 term
                    harmonics.append(np.ones_like(t) / np.sqrt(4 * np.pi))
                elif l == 1:
                    # Y1m terms
                    if m == 0:
                        harmonics.append(np.sqrt(3/(4*np.pi)) * np.cos(t))
                    elif m == 1:
                        harmonics.append(-np.sqrt(3/(8*np.pi)) * np.sin(t))
                    else:  # m == -1
                        harmonics.append(np.sqrt(3/(8*np.pi)) * np.sin(t))
        
        # Visualize harmonic mixing
        harmonics = np.array(harmonics)
        for frame in range(30):
            mixed = np.zeros_like(t)
            for i, harmonic in enumerate(harmonics):
                mixed += harmonic * np.sin(frame * 0.2 + i * 0.5)
            
            # Normalize and visualize
            mixed = (mixed - mixed.min()) / (mixed.max() - mixed.min())
            
            viz_line = ""
            for val in mixed[::3]:  # Sample every 3rd point for display
                idx = int(val * (len(self.frequency_chars) - 1))
                viz_line += self.frequency_chars[idx]
                
            print(f"\r{viz_line}", end="", flush=True)
            time.sleep(0.1)
        print()

def demo_visualization():
    # Setup parameters
    in_features = 64
    out_features = 128
    num_tokens = 16
    
    visualizer = TokenLinearVisualizer(in_features, out_features, num_tokens)
    
    # Generate sample data
    tokens = torch.randn(num_tokens, in_features)
    q = torch.randn(10, in_features)
    k = torch.randn(num_tokens, in_features)
    freq_tokens = torch.randn(num_tokens, in_features)
    
    # Run visualizations
    print("\033[35m╔" + "═" * 78 + "╗\033[0m")
    print("\033[35m║\033[0m" + f"{'Token Linear Architecture Visualization':^78}" + "\033[35m║\033[0m")
    print("\033[35m╚" + "═" * 78 + "╝\033[0m")
    
    visualizer.visualize_token_activation(tokens)
    visualizer.visualize_attention_flow(q, k)
    visualizer.visualize_frequency_mixing(freq_tokens)
    
    print("\n\033[35m╔" + "═" * 78 + "╗\033[0m")
    print("\033[35m║\033[0m" + f"{'Press Enter to exit...':^78}" + "\033[35m║\033[0m")
    print("\033[35m╚" + "═" * 78 + "╝\033[0m")
    
    input()

if __name__ == "__main__":
    demo_visualization()