"""
Create sample audio file for testing (1 second sine wave)
"""
import numpy as np
from scipy.io import wavfile
import os

# Create samples directory if not exists
os.makedirs("samples", exist_ok=True)

# Parameters
sample_rate = 16000  # 16kHz
duration = 1  # 1 second
frequency = 440  # A4 note (440Hz)

# Generate sine wave
t = np.linspace(0, duration, int(sample_rate * duration))
waveform = np.sin(2 * np.pi * frequency * t)

# Normalize to 16-bit range
waveform = (waveform * 32767).astype(np.int16)

# Save file
output_file = "samples/test_audio.wav"
wavfile.write(output_file, sample_rate, waveform)

print(f"✅ Sample file created: {output_file}")
print(f"   - Sample rate: {sample_rate} Hz")
print(f"   - Duration: {duration}s")
print(f"   - Frequency: {frequency} Hz")
print(f"\n💡 You can upload this file to the web interface for testing!")
