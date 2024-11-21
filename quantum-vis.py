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
import signal
from functools import lru_cache

class VisualizationMode(Enum):
    WAVE = "wave"
    HARMONICS = "harmonics" 
    ADVANCED_HARMONICS = "advanced_harmonics"
    INTERFERENCE = "interference"
    PROBABILITY = "probability"
    MEASUREMENT = "measurement"

@dataclass
class VisualParameters:
    resolution: Tuple[int, int] = (80, 24)
    frame_rate: float = 0.05
    animation_speed: float = 0.5  # Slowed down for better visualization
    quantum_uncertainty: float = 0.1
    measurement_precision: float = 0.01
    
    @property
    def color_map(self) -> Dict[str, str]:
        return {
            "wave": "\033[38;5;51m",          # Bright cyan
            "harmonics": "\033[38;5;226m",    # Yellow
            "advanced_harmonics": "\033[38;5;141m",  # Purple
            "interference": "\033[38;5;205m",  # Magenta
            "probability": "\033[38;5;118m",   # Green
            "measurement": "\033[38;5;208m",   # Orange
            "scale": "\033[38;5;245m",        # Gray
            "reset": "\033[0m",
            "header": "\033[1;36m",           # Bright cyan bold
            "error": "\033[1;31m",            # Bright red bold
            "value": "\033[1;37m"             # Bright white bold
        }

class ContinuousWaveGenerator:
    def __init__(self):
        self._measurement_history = []
        self._time = 0
        
    def generate_wave(self, x: np.ndarray, t: float) -> np.ndarray:
        base = 0.5 * (1 + np.cos(x/5 + t/3)) + 0.3 * np.cos(x/5 - t/3)
        noise = np.random.normal(0, 0.02, size=x.shape)
        return base + noise
        
    def generate_harmonics(self, x: np.ndarray, t: float) -> np.ndarray:
        l_max = 2
        result = np.zeros_like(x, dtype=complex)
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                phase = np.exp(-1j * (l * (l + 1)) * t / 2)
                result += phase * np.exp(1j * m * x)
        return np.abs(result)

    def generate_advanced_harmonics(self, x: np.ndarray, t: float) -> np.ndarray:
        phases = np.array([1, 1j, -1, -1j]) * t
        components = [np.exp(1j * k * x + p) for k, p in enumerate(phases)]
        return np.abs(sum(components))

    def generate_interference(self, x: np.ndarray, t: float) -> np.ndarray:
        wave1 = np.sin(x + t)
        wave2 = np.sin(2*x - t/2)
        return 0.5 * (wave1 + wave2) + 0.5

    def generate_probability(self, x: np.ndarray, t: float) -> np.ndarray:
        psi = self.generate_wave(x, t)
        return np.abs(psi)**2

    def measure_state(self, x: np.ndarray, t: float) -> np.ndarray:
        wave = self.generate_wave(x, t)
        measurement_strength = 0.3
        
        # Keep track of multiple measurement points
        if len(self._measurement_history) < 5 or t - self._time > 1.0:
            collapse_position = np.random.choice(len(x), p=np.abs(wave)**2/np.sum(np.abs(wave)**2))
            self._measurement_history.append((t, x[collapse_position]))
            self._time = t
            
        # Create visualization that shows measurement collapses
        result = np.zeros_like(x)
        for mt, mx in self._measurement_history[-5:]:
            time_factor = np.exp(-(t - mt))
            result += time_factor * np.exp(-(x - mx)**2 / (2 * measurement_strength))
            
        # Remove old measurements
        if len(self._measurement_history) > 20:
            self._measurement_history = self._measurement_history[-20:]
            
        return result / (result.max() + 1e-10)

class EnhancedVisualizer:
    def __init__(self, params: VisualParameters):
        self.params = params
        self.wave_gen = ContinuousWaveGenerator()
        self.running = True
        self.chars = "▁▂▃▄▅▆▇█"
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        self.running = False
        print(f"\n{self.params.color_map['reset']}Visualization stopped.")

    def render_line(self, data: np.ndarray, mode: str) -> str:
        """Render a single line of visualization"""
        # Add safety checks and handling for invalid data
        if np.all(np.isnan(data)) or np.all(np.isinf(data)):
            return f"{self.params.color_map['error']}Invalid data"
        
        # Handle cases where min and max are equal
        data_range = data.max() - data.min()
        if data_range == 0:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - data.min()) / data_range
        
        # Clip values to ensure they're in valid range
        normalized = np.clip(normalized, 0, 1)
        indices = (normalized * (len(self.chars) - 1)).astype(int)
        indices = np.clip(indices, 0, len(self.chars) - 1)
        
        line = [self.chars[idx] for idx in indices]
        
        # Add measurement markers for measurement mode
        if mode == "measurement":
            for _, pos in self.wave_gen._measurement_history[-5:]:
                idx = int(pos / (2*np.pi) * len(line))
                if 0 <= idx < len(line):
                    line[idx] = "◉"
        
        return (f"{self.params.color_map['scale']}│"
                f"{self.params.color_map[mode]}{''.join(line)}"
                f"{self.params.color_map['scale']}│"
                f"{self.params.color_map['value']} {data.max():6.3f}")

    def run_visualization(self):
        print(f"{self.params.color_map['header']}Quantum State Visualization")
        print(f"{self.params.color_map['scale']}{'═' * (self.params.resolution[0] + 15)}")
        
        frame = 0
        while self.running:
            try:
                x = np.linspace(0, 2*np.pi, self.params.resolution[0])
                t = frame * self.params.animation_speed
                
                # Generate all visualizations
                visualizations = {
                    'wave': self.wave_gen.generate_wave(x, t),
                    'harmonics': self.wave_gen.generate_harmonics(x, t),
                    'advanced_harmonics': self.wave_gen.generate_advanced_harmonics(x, t),
                    'interference': self.wave_gen.generate_interference(x, t),
                    'probability': self.wave_gen.generate_probability(x, t),
                    'measurement': self.wave_gen.measure_state(x, t)
                }
                
                # Clear previous frame
                print("\033[F" * (len(visualizations) + 2))
                
                # Render each mode
                for mode, data in visualizations.items():
                    name = f"{mode.replace('_', ' ').title():18}"
                    print(f"{self.params.color_map['scale']}{name}{self.render_line(data, mode)}")
                
                frame += 1
                time.sleep(self.params.frame_rate)
                
            except Exception as e:
                print(f"{self.params.color_map['error']}Error: {str(e)}")
                continue

def main():
    try:
        params = VisualParameters()
        visualizer = EnhancedVisualizer(params)
        visualizer.run_visualization()
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()