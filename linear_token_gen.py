import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import math
from scipy.special import sph_harm
import time
import sys

class TokenLinearDataGenerator:
    """Generates synthetic training data suited for token linear architectures"""
    
    def __init__(self, in_features: int, out_features: int, num_tokens: int):
        self.in_features = in_features
        self.out_features = out_features
        self.num_tokens = num_tokens
        
    def generate_basis_transformation_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates data where the relationship between input and output involves 
        multiple frequency components - ideal for testing the frequency-based tokens
        
        Returns:
            x: Input tensor of shape [num_samples, in_features]
            y: Target tensor of shape [num_samples, out_features]
        """
        # Generate random frequency components
        freqs = torch.randn(self.in_features, self.out_features)
        phases = torch.randn(self.out_features) * 2 * np.pi
        
        # Generate input data
        x = torch.randn(num_samples, self.in_features)
        
        # Generate output through frequency mixing
        y = torch.zeros(num_samples, self.out_features)
        for i in range(self.out_features):
            y[:, i] = torch.sin(x @ freqs[:, i] + phases[i])
            
        return x, y
    
    def generate_piecewise_data(self, num_samples: int, num_regions: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates piecewise functions where different regions have different linear transformations.
        This tests the model's ability to use different tokens for different input regions.
        
        Returns:
            x: Input tensor of shape [num_samples, in_features]
            y: Target tensor of shape [num_samples, out_features]
        """
        # Generate region centroids
        centroids = torch.randn(num_regions, self.in_features)
        
        # Generate different linear transformations for each region
        transformations = [torch.randn(self.in_features, self.out_features) for _ in range(num_regions)]
        
        # Generate input data
        x = torch.randn(num_samples, self.in_features)
        
        # Assign each point to nearest centroid and apply corresponding transformation
        y = torch.zeros(num_samples, self.out_features)
        for i in range(num_samples):
            # Find nearest centroid
            distances = torch.norm(x[i:i+1] - centroids, dim=1)
            region_idx = torch.argmin(distances)
            
            # Apply region-specific transformation
            y[i] = x[i] @ transformations[region_idx]
            
        return x, y
    
    def generate_attention_pattern_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates data where the output depends on selective attention to different 
        input features based on the input pattern.
        
        Returns:
            x: Input tensor of shape [num_samples, in_features]
            y: Target tensor of shape [num_samples, out_features]
        """
        # Generate attention patterns
        patterns = torch.randn(self.num_tokens, self.in_features)
        pattern_outputs = torch.randn(self.num_tokens, self.out_features)
        
        # Generate input data
        x = torch.randn(num_samples, self.in_features)
        
        # Generate output based on similarity to patterns
        similarities = F.softmax(x @ patterns.T / np.sqrt(self.in_features), dim=-1)
        y = similarities @ pattern_outputs
        
        return x, y
    
    def generate_composite_data(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates data that combines multiple types of relationships that the 
        token-based architecture should be able to learn.
        
        Returns:
            x: Input tensor of shape [num_samples, in_features]
            y: Target tensor of shape [num_samples, out_features]
        """
        # Generate basis data
        x_basis, y_basis = self.generate_basis_transformation_data(num_samples)
        
        # Generate piecewise data
        x_piece, y_piece = self.generate_piecewise_data(num_samples, num_regions=self.num_tokens // 4)
        
        # Generate attention data
        x_attn, y_attn = self.generate_attention_pattern_data(num_samples)
        
        # Combine the different types (using same input but combining outputs)
        x = x_basis  # Could also combine inputs in interesting ways
        y = (y_basis + y_piece + y_attn) / 3.0
        
        return x, y

def test_data_generator():
    """Test function to demonstrate usage of the data generator"""
    # Parameters matching the architecture
    in_features = 64
    out_features = 128
    num_tokens = 16
    num_samples = 1000
    
    # Create generator
    generator = TokenLinearDataGenerator(in_features, out_features, num_tokens)
    
    # Generate different types of data
    datasets = {
        "basis": generator.generate_basis_transformation_data,
        "piecewise": lambda n: generator.generate_piecewise_data(n, num_regions=4),
        "attention": generator.generate_attention_pattern_data,
        "composite": generator.generate_composite_data
    }
    
    # Test each type
    for name, gen_func in datasets.items():
        x, y = gen_func(num_samples)
        print(f"\n{name.upper()} Dataset:")
        print(f"X shape: {x.shape}, Y shape: {y.shape}")
        print(f"X stats - Mean: {x.mean():.3f}, Std: {x.std():.3f}")
        print(f"Y stats - Mean: {y.mean():.3f}, Std: {y.std():.3f}")

if __name__ == "__main__":
    test_data_generator()

# Spherical Harmonics in Minkowski Space-Time
# Setup parameters for spherical harmonics
theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

# Calculate 0th and 1st order spherical harmonics
Y00 = np.sqrt(1.0 / (4 * np.pi)) * np.ones_like(theta) # l=0, m=0
Y10 = np.sqrt(3.0 / (4 * np.pi)) * np.cos(theta)       # l=1, m=0
Y1p1 = -np.sqrt(3.0 / (8 * np.pi)) * np.sin(theta) * np.exp(1j * phi)  # l=1, m=1
Y1n1 = np.sqrt(3.0 / (8 * np.pi)) * np.sin(theta) * np.exp(-1j * phi)  # l=1, m=-1

# Convert to Minkowski coordinates (t, x, y, z)
# Using time parameter for animation
t = np.linspace(0, 2*np.pi, 50)
minkowski_coords = []

print("\033[36mGenerating Minkowski Space-Time coordinates...\033[0m")

for time_step in t:
    # Combine harmonics with time evolution
    r = np.abs(Y00 + Y10 * np.exp(-1j * time_step))
    
    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    # Store as Minkowski 4-vectors
    minkowski_coords.append(np.stack([time_step * np.ones_like(x), x, y, z]))

# Save data for hypersphere visualization
np.save('minkowski_harmonics.npy', np.array(minkowski_coords))

# Define tile value probabilities
tile_stats = {
    "█": {"prob": 0.2, "desc": "High density region"},
    "▓": {"prob": 0.2, "desc": "Medium-high density"},
    "▒": {"prob": 0.2, "desc": "Medium density"},
    "░": {"prob": 0.2, "desc": "Low density"},
    " ": {"prob": 0.2, "desc": "Empty region"},
    "∎": {"prob": 0.25, "desc": "Strong positive interference"},
    "∙": {"prob": 0.25, "desc": "Weak positive interference"},
    "∘": {"prob": 0.25, "desc": "Weak negative interference"},
    "□": {"prob": 0.25, "desc": "Strong negative interference"}
}

print("\033[32mTile Value Statistics:\033[0m")
print("Base Layer:")
for char, stats in list(tile_stats.items())[:5]:
    print(f"{char} - Probability: {stats['prob']:.2f} - {stats['desc']}")
print("\nInterference Layer:")
for char, stats in list(tile_stats.items())[5:]:
    print(f"{char} - Probability: {stats['prob']:.2f} - {stats['desc']}")

user_input = ''
while user_input.lower() != 'exit':
    # First layer - sine wave
    for frame in range(30):
        wave = ""
        interference = ""
        for x in range(50):
            # First layer - Combine 0th and 1st order contributions
            y = 0.5 * (1 + np.cos(x/5 + frame/3))  # Y00 contribution
            y += 0.3 * np.cos(x/5 - frame/3)       # Y10 contribution
            
            # Second layer - Interference pattern
            y2 = 0.4 * np.sin(x/3 + frame/2)       # Y1p1 contribution
            y2 += 0.4 * np.sin(x/3 - frame/2)      # Y1n1 contribution
            
            # Map first layer to ASCII characters
            if y > 0.8:
                char = "█"
            elif y > 0.6:
                char = "▓"
            elif y > 0.4:
                char = "▒"
            elif y > 0.2:
                char = "░"
            else:
                char = " "
            wave += char
            
            # Map interference pattern to different ASCII characters
            if y2 > 0.6:
                char2 = "∎"
            elif y2 > 0.2:
                char2 = "∙"
            elif y2 < -0.2:
                char2 = "∘"
            elif y2 < -0.6:
                char2 = "□"
            else:
                char2 = " "
            interference += char2
        
        print(f"\r\033[33m[{wave}]\033[0m", flush=True)
        print(f"\r\033[34m[{interference}]\033[0m", end="", flush=True)
        time.sleep(0.1)
    
    print("\n\033[35m╔" + "═" * 78 + "╗\033[0m")
    print("\033[35m║\033[0m" + f"{'Type exit to stop or press Enter to continue':^78}" + "\033[35m║\033[0m")
    print("\033[35m╚" + "═" * 78 + "╝\033[0m")
    
    user_input = input()

print("\n\033[35m╔" + "═" * 78 + "╗\033[0m")
print("\033[35m║\033[0m" + f"{'VISUALIZATION COMPLETE':^78}" + "\033[35m║\033[0m")
print("\033[35m╚" + "═" * 78 + "╝\033[0m")
