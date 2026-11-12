import soundfile as sf
import os

directory_path = "data/Corrected_Dataset/Depression/Stage2"

# List all files in the directory
files = os.listdir(directory_path)
print("Files in directory:", files)

# Find the first .wav file
wav_files = [f for f in files if f.endswith('.wav')]
if wav_files:
    first_audio_file = os.path.join(directory_path, wav_files[0])
    print(f"\nChecking: {first_audio_file}")
    
    info = sf.info(first_audio_file)
    print(f"Sampling rate: {info.samplerate} Hz")
    print(f"Channels: {info.channels}")
    print(f"Duration: {info.duration:.2f} seconds")
    print(f"Format: {info.format}")
    print(f"Subtype: {info.subtype}")
else:
    print("No .wav files found in the directory!")