import torch
import numpy as np
from scipy.io import wavfile
import pygame
from PIL import Image, ImageDraw
import colorsys

class SynthwaveGenerator:
    def __init__(self, sample_rate=44100, duration=30):
        self.sample_rate = sample_rate
        self.duration = duration
        self.base_freq = 440  # A4 note
        
        # Synthwave characteristic frequencies
        self.bass_freq = self.base_freq / 4
        self.arp_freqs = [self.base_freq * (2 ** (n/12)) for n in [0, 4, 7, 11]]  # Major seventh chord
        self.pad_freqs = [self.base_freq * (2 ** (n/12)) for n in [0, 3, 7]]  # Minor chord
        
        # Effect parameters
        self.reverb_delay = int(0.1 * sample_rate)  # 100ms reverb
        self.chorus_rate = 2  # Hz
        self.chorus_depth = 0.002  # seconds
        
    def generate_oscillator(self, freq, duration, waveform='saw'):
        """Generate basic waveform"""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        if waveform == 'saw':
            wave = 2 * (t * freq - np.floor(0.5 + t * freq))
        elif waveform == 'square':
            wave = np.sign(np.sin(2 * np.pi * freq * t))
        else:  # sine
            wave = np.sin(2 * np.pi * freq * t)
        return wave
    
    def apply_envelope(self, wave, attack=0.1, decay=0.2, sustain=0.7, release=0.3):
        """Apply ADSR envelope"""
        samples = len(wave)
        envelope = np.zeros(samples)
        
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        release_samples = int(release * self.sample_rate)
        
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:attack_samples+decay_samples] = \
            np.linspace(1, sustain, decay_samples)
        envelope[attack_samples+decay_samples:-release_samples] = sustain
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
        
        return wave * envelope
    
    def generate_bassline(self):
        """Generate synthwave bassline"""
        bass = self.generate_oscillator(self.bass_freq, self.duration, 'saw')
        bass = self.apply_envelope(bass, attack=0.1, decay=0.3, sustain=0.8, release=0.2)
        return bass * 0.3  # Reduce volume
    
    def generate_arpeggios(self):
        """Generate arpeggiated sequence"""
        arp_sequence = []
        step_duration = 0.125  # 16th notes at 120 BPM
        
        for t in range(int(self.duration / step_duration)):
            freq = self.arp_freqs[t % len(self.arp_freqs)]
            note = self.generate_oscillator(freq, step_duration, 'square')
            note = self.apply_envelope(note, attack=0.01, decay=0.1, sustain=0.5, release=0.01)
            arp_sequence.append(note)
            
        return np.concatenate(arp_sequence) * 0.2
    
    def generate_pads(self):
        """Generate atmospheric pads"""
        pads = np.zeros(int(self.sample_rate * self.duration))
        t = np.linspace(0, self.duration, len(pads))
        
        for freq in self.pad_freqs:
            pad = self.generate_oscillator(freq, self.duration, 'sine')
            # Add chorus effect
            mod = np.sin(2 * np.pi * self.chorus_rate * t)
            # Create a delayed copy of the pad
            chorus_pad = np.zeros_like(pad)
            for i in range(len(pad)):
                offset = int(self.chorus_depth * self.sample_rate * mod[i])
                if i + offset < len(pad):
                    chorus_pad[i] = pad[i + offset]
                else:
                    chorus_pad[i] = pad[i]
            
            pads += (pad + chorus_pad) * 0.15
            
        return self.apply_envelope(pads, attack=2.0, decay=1.0, sustain=0.6, release=2.0)
    
    def generate_track(self, output_file="synthwave.wav"):
        """Generate complete synthwave track"""
        # Calculate exact number of samples needed
        total_samples = int(self.sample_rate * self.duration)
        
        # Generate components
        bassline = self.generate_bassline()
        arpeggios = self.generate_arpeggios()
        pads = self.generate_pads()
        
        # Ensure all components have the same length
        def normalize_length(audio, target_length):
            if len(audio) > target_length:
                return audio[:target_length]
            elif len(audio) < target_length:
                padding = np.zeros(target_length - len(audio))
                return np.concatenate([audio, padding])
            return audio
        
        # Normalize lengths
        bassline = normalize_length(bassline, total_samples)
        arpeggios = normalize_length(arpeggios, total_samples)
        pads = normalize_length(pads, total_samples)
        
        # Mix elements
        mix = bassline + arpeggios + pads
        
        # Normalize
        mix = mix / np.max(np.abs(mix))
        
        # Convert to 16-bit PCM
        mix = (mix * 32767).astype(np.int16)
        
        # Save to file
        wavfile.write(output_file, self.sample_rate, mix)
        return mix
    
    def visualize(self, audio_data, output_file="synthwave_vis.gif"):
        """Generate retro-style visualization"""
        frames = []
        width, height = 800, 400
        chunk_size = len(audio_data) // 60  # 60 frames
        
        for i in range(60):
            # Create frame
            img = Image.new('RGB', (width, height), (0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw grid
            for x in range(0, width, 40):
                color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(0.8, 0.8, 0.5))
                draw.line([(x, 0), (x, height)], fill=color, width=1)
            
            # Draw waveform
            chunk = audio_data[i*chunk_size:(i+1)*chunk_size]
            points = []
            for j, sample in enumerate(chunk[::100]):
                x = j * (width / (chunk_size // 100))
                y = (sample / 32767.0) * (height/4) + height/2
                points.append((x, y))
            
            # Draw sun
            sun_color = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(0.9, 0.8, 1.0))
            draw.ellipse([width/2-40, 50, width/2+40, 130], fill=sun_color)
            
            if len(points) > 1:
                draw.line(points, fill=(255, 50, 255), width=2)
            
            frames.append(img)
        
        # Save as animated GIF
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )

# Usage example
if __name__ == "__main__":
    generator = SynthwaveGenerator(duration=30)  # 30 seconds
    audio_data = generator.generate_track("synthwave_output.wav")
    generator.visualize(audio_data, "synthwave_visualization.gif")