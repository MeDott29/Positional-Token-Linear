import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from token_linear import TokenLinearGL
import math
from pathlib import Path
import imageio.v2 as imageio

class AudioTerminalSimulator:
    def __init__(self, width=800, height=600, num_frames=60):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.bg_color = (0, 0, 0)  # Black background
        self.text_color = (0, 255, 0)  # Green text
        self.wave_color = (255, 255, 0)  # Yellow wave
        self.interference_color = (0, 0, 255)  # Blue interference
        
        # Initialize the TokenLinear model
        self.model = TokenLinearGL(
            in_features=64,  # Input features for audio processing
            out_features=50,  # Output features (matches terminal width)
            num_tokens=32    # Number of frequency components
        )
        
        # Characters for visualization
        self.chars = {
            'solid': '█',
            'mediumHigh': '▓',
            'medium': '▒',
            'low': '░',
            'empty': ' ',
            'strongPos': '∎',
            'weakPos': '∙',
            'weakNeg': '∘',
            'strongNeg': '□'
        }

    def generate_audio_data(self, frame):
        """Simulate audio data with multiple frequency components"""
        t = frame / self.num_frames
        
        # Create a mixture of frequencies
        base_freq = 2 * math.pi * t
        frequencies = torch.linspace(1, 10, 64)
        phases = torch.tensor([base_freq * f for f in frequencies])
        
        # Generate complex waveform
        audio_data = torch.sin(phases) + 0.5 * torch.sin(2 * phases)
        audio_data += 0.3 * torch.sin(3 * phases + math.pi/4)
        
        # Normalize
        audio_data = audio_data / audio_data.abs().max()
        return audio_data.unsqueeze(0)  # Add batch dimension

    def process_frame(self, frame):
        """Process one frame of audio data through the model"""
        audio_data = self.generate_audio_data(frame)
        
        with torch.no_grad():
            output = self.model(audio_data)
            
        # Scale output to [-1, 1] range
        output = output.squeeze(0)
        output = output / output.abs().max()
        
        return output

    def map_to_chars(self, values, char_set):
        """Map numerical values to ASCII characters"""
        result = ''
        for v in values:
            v = v.item()
            if v > 0.8:
                result += char_set['solid']
            elif v > 0.4:
                result += char_set['mediumHigh']
            elif v > 0:
                result += char_set['medium']
            elif v > -0.4:
                result += char_set['low']
            else:
                result += char_set['empty']
        return result

    def create_frame_image(self, frame_num):
        """Create a single frame as PIL Image"""
        # Create new image with black background
        image = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(image)
        
        # Process audio data
        output = self.process_frame(frame_num)
        
        # Split output into two visualization layers
        wave_data = output[:len(output)//2]
        interference_data = output[len(output)//2:]
        
        # Convert to ASCII art
        wave_line = self.map_to_chars(wave_data, self.chars)
        interference_line = self.map_to_chars(interference_data, self.chars)
        
        # Draw terminal content
        y_offset = 50
        draw.text((40, y_offset), f"[{wave_line}]", fill=self.wave_color)
        draw.text((40, y_offset + 30), f"[{interference_line}]", fill=self.interference_color)
        
        # Add terminal prompt
        draw.text((40, y_offset + 80), "> _", fill=self.text_color)
        
        return image

    def generate_gif(self, output_path="audio_terminal.gif"):
        """Generate GIF from frames"""
        frames = []
        
        print("Generating frames...")
        for i in range(self.num_frames):
            frame = self.create_frame_image(i)
            frames.append(frame)
            if i % 10 == 0:
                print(f"Generated frame {i}/{self.num_frames}")
        
        print("Saving GIF...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # 50ms per frame (20 fps)
            loop=0
        )
        print(f"Saved animation to {output_path}")

if __name__ == "__main__":
    # Create and run the simulator
    simulator = AudioTerminalSimulator()
    simulator.generate_gif()