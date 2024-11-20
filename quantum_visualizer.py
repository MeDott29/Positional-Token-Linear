import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import math
from scipy.special import sph_harm, eval_hermite
import time
import sys
from dataclasses import dataclass
from enum import Enum
import colorsys
from functools import lru_cache
import select

class VisualizationMode(Enum):
    WAVE = "wave"
    HARMONICS = "harmonics" 
    ADVANCED_HARMONICS = "advanced_harmonics"  # New mode
    INTERFERENCE = "interference"
    COMPOSITE = "composite"
    PROBABILITY = "probability"

@dataclass
class VisualParameters:
    """Parameters controlling visualization appearance and behavior"""
    resolution: Tuple[int, int] = (40, 30)  # Adjusted for mirroring
    frame_rate: float = 0.1
    color_scheme: str = "quantum"
    animation_speed: float = 1.0
    quantum_uncertainty: float = 0.1
    frame_count: int = 30
    harmonic_order: int = 3  # Order of advanced harmonics
    
    @property
    def color_map(self) -> Dict[str, str]:
        schemes = {
            "quantum": {
                "background": "\033[40m",
                "wave": "\033[38;5;51m",        # Cyan
                "interference": "\033[38;5;205m", # Magenta
                "harmonics": "\033[38;5;226m",    # Yellow  # Changed from "harmonic"
                "advanced_harmonics": "\033[38;5;141m",  # Purple
                "probability": "\033[38;5;118m", # Green
                "reset": "\033[0m"
            },
            "thermal": {
                "background": "\033[40m",
                "wave": "\033[38;5;196m",
                "interference": "\033[38;5;208m",
                "harmonics": "\033[38;5;226m",    # Changed from "harmonic"
                "advanced_harmonics": "\033[38;5;99m",
                "probability": "\033[38;5;255m",
                "reset": "\033[0m"
            }
        }
        return schemes.get(self.color_scheme, schemes["quantum"])

class WaveFunctionGenerator:
    """Handles wave function generation and caching"""
    
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=128)
    def _generate_hermite(self, n: int, x_points: int) -> np.ndarray:
        """Generate Hermite polynomial with caching"""
        x = np.linspace(-5, 5, x_points)
        return eval_hermite(n, x)

    def generate_advanced_harmonics(self, x: np.ndarray, t: float, order: int = 3) -> np.ndarray:
        """Generate advanced quantum harmonic oscillator states"""
        # Combine multiple harmonic oscillator states
        psi = np.zeros_like(x, dtype=complex)
        for n in range(order + 1):
            # Quantum number dependent phase
            phase = t * (n + 0.5)
            # Energy eigenstate
            hermite = self._generate_hermite(n, len(x))
            # Quantum superposition with phase evolution
            coefficient = np.exp(-1j * phase) / np.sqrt(2**n * math.factorial(n))
            psi += coefficient * hermite * np.exp(-x**2 / 4)
        
        return np.abs(psi)

    def generate_wave(self, x: np.ndarray, t: float) -> np.ndarray:
        """Generate wave function"""
        return 0.5 * (1 + np.cos(x/5 + t/3)) + 0.3 * np.cos(x/5 - t/3)

    def generate_harmonics(self, x: np.ndarray, t: float) -> np.ndarray:
        """Generate spherical harmonics"""
        l_max = 2  # Maximum angular momentum
        result = np.zeros_like(x, dtype=complex)
        
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                # Add time evolution to spherical harmonics
                phase = np.exp(-1j * (l * (l + 1)) * t / 2)
                result += phase * sph_harm(m, l, x, t)
                
        return np.abs(result)

    def generate_interference(self, x: np.ndarray, t: float) -> np.ndarray:
        """Generate interference pattern with quantum effects"""
        # Basic interference
        basic = 0.4 * (np.sin(x/3 + t/2) + np.sin(x/3 - t/2))
        
        # Add quantum tunneling effect
        tunneling = 0.2 * np.exp(-(x - np.pi)**2 / 2) * np.cos(t)
        
        # Combine with phase relationship
        return basic + tunneling

    def generate_probability(self, x: np.ndarray, t: float) -> np.ndarray:
        """Generate probability density with quantum corrections"""
        psi = self.generate_wave(x, t)
        # Add quantum corrections
        correction = 0.1 * np.sin(2 * x) * np.exp(-t/10)
        return np.abs(psi + correction) ** 2

class QuantumVisualizer:
    """Advanced visualization system for quantum-mechanical phenomena"""
    
    def __init__(self, params: VisualParameters):
        self.params = params
        self.wave_gen = WaveFunctionGenerator()
        self.characters = {
            "density": ["█", "▓", "▒", "░", " "],
            "interference": ["∎", "∙", "∘", "□", " "],
            "probability": ["⬣", "⬢", "⬡", "○", " "],
            "advanced": ["◉", "◈", "◇", "○", " "]  # New characters for advanced harmonics
        }
        self._display_buffer = self._create_buffer()

    def _create_buffer(self) -> np.ndarray:
        """Create visualization buffer"""
        return np.full(self.params.resolution, " ", dtype=str)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to [0, 1] range with smooth transitions"""
        # Convert complex data to real by taking magnitude
        if np.iscomplexobj(data):
            data = np.abs(data)
        
        min_val, max_val = data.min(), data.max()
        if min_val == max_val:
            return np.zeros_like(data)
        normalized = (data - min_val) / (max_val - min_val)
        # Apply smooth transition curve
        return 0.5 * (1 + np.tanh(4 * (normalized - 0.5)))

    def _get_visualization_data(self, mode: VisualizationMode, x: np.ndarray, t: float) -> np.ndarray:
        """Get visualization data based on mode"""
        generators = {
            VisualizationMode.WAVE: self.wave_gen.generate_wave,
            VisualizationMode.HARMONICS: self.wave_gen.generate_harmonics,
            VisualizationMode.ADVANCED_HARMONICS: 
                lambda x, t: self.wave_gen.generate_advanced_harmonics(x, t, self.params.harmonic_order),
            VisualizationMode.INTERFERENCE: self.wave_gen.generate_interference,
            VisualizationMode.PROBABILITY: self.wave_gen.generate_probability
        }
        
        if mode == VisualizationMode.COMPOSITE:
            # Take magnitude of complex values before adding
            return sum(np.abs(func(x, t)) for func in generators.values()) / len(generators)
            
        return generators[mode](x, t)

    def render_frame(self, frame: int, mode: VisualizationMode) -> str:
        """Render a single visualization frame with enhanced effects"""
        try:
            x = np.linspace(0, 2*np.pi, self.params.resolution[0])
            t = frame * self.params.animation_speed
            
            # Get and process data
            data = self._get_visualization_data(mode, x, t)
            if mode == VisualizationMode.PROBABILITY:
                data += np.random.normal(0, self.params.quantum_uncertainty, data.shape)
            
            # Normalize and map to characters
            normalized_data = self._normalize_data(data)
            char_map = (self.characters["advanced"] if mode == VisualizationMode.ADVANCED_HARMONICS
                       else self.characters["probability"] if mode == VisualizationMode.PROBABILITY
                       else self.characters["interference"] if mode == VisualizationMode.INTERFERENCE
                       else self.characters["density"])
            chars = np.array(char_map)[(normalized_data * (len(char_map)-1)).astype(int)]
            
            # Add dynamic frame effects
            frame_effect = "⟫" if frame % 2 == 0 else "⟪"
            phase_indicator = "φ" if mode == VisualizationMode.ADVANCED_HARMONICS else ""
            
            return f"{self.params.color_map[mode.value]}{frame_effect}{phase_indicator}{''.join(chars)}{phase_indicator}{frame_effect}{self.params.color_map['reset']}"
            
        except Exception as e:
            print(f"\nError in frame rendering: {str(e)}")
            return ""

    def run_visualization(self):
        """Run main visualization loop with enhanced display"""
        try:
            user_input = ''
            width = self.params.resolution[0]
            
            # Clear screen and show introduction
            print("\033[2J\033[H")  # Clear screen and move cursor to top
            print("\n".join(line.rjust(width * 2) for line in [
                "AI Consciousness Visualization",
                "═" * (width * 2),
                "Layer 1: Base Consciousness",
                "Layer 2: Thought Processing",
                "═" * (width * 2),
            ]))
            
            while user_input.lower() != 'exit':
                for frame in range(self.params.frame_count):
                    # Get base visualizations
                    wave = self.render_frame(frame, VisualizationMode.WAVE)
                    harmonics = self.render_frame(frame, VisualizationMode.HARMONICS)
                    
                    # Create mirrored pattern
                    wave_pattern = wave + wave[::-1]
                    harmonics_pattern = harmonics + harmonics[::-1]
                    
                    # Right justify and display with padding
                    print(f"\r{wave_pattern.rjust(width * 2)}", flush=True)
                    print(f"\r{harmonics_pattern.rjust(width * 2)}", flush=True)
                    
                    # Add subtle "thinking" animation
                    thinking = "∙∙∙" if frame % 3 == 0 else "∙∙ " if frame % 3 == 1 else "∙  "
                    print(f"\r{'Processing' + thinking:>{width * 2}}", flush=True)
                    
                    time.sleep(self.params.frame_rate)
                    
                    # Move cursor up to overwrite previous lines
                    print("\033[3A", end="", flush=True)
                
                # Check for exit without interrupting visualization
                if kbhit():  # For Unix-like systems
                    user_input = sys.stdin.read(1)
                    if user_input.lower() == 'q':
                        break
                
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")
        except Exception as e:
            print(f"\nError in visualization loop: {str(e)}")
        finally:
            print("\n" * 4)  # Clear space after visualization

def kbhit():
    return select.select([sys.stdin], [], [], 0)[0] != []

def main():
    try:
        # Initialize system
        params = VisualParameters()
        visualizer = QuantumVisualizer(params)
        
        # Display initialization message
        print("\033[1;36mQuantum Visualization System Initialized\033[0m")
        print("═" * 80)
        print("Features:")
        features = [
            "Quantum harmonic oscillator states",
            "Advanced quantum harmonics visualization",
            "Spherical harmonics in Minkowski space-time",
            "Wave-particle duality visualization",
            "Interference pattern analysis",
            "Probability density mapping",
            "Quantum uncertainty effects"
        ]
        for feature in features:
            print(f"- {feature}")
        print("═" * 80)
        
        # Run visualization
        visualizer.run_visualization()
        
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()