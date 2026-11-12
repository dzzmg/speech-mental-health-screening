# Import required libraries
import os
import librosa
import soundfile as sf
from pathlib import Path
import shutil

def resample_preserve_structure(input_root_dir, output_root_dir, target_sr=16000):
    """
    Resample all audio files while preserving the original folder structure
    
    Parameters:
    input_root_dir (str): Root directory with original audio files and subfolders
    output_root_dir (str): Root directory where resampled files will be saved with same structure
    target_sr (int): Target sample rate (default: 16000 for wav2vec2)
    """
    
    # Create output root directory if it doesn't exist
    Path(output_root_dir).mkdir(parents=True, exist_ok=True)
    
    # Supported audio formats
    audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.WAV', '.MP3']
    
    # Counter for statistics
    successful_files = 0
    failed_files = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_root_dir):
        for file in files:
            # Check if file is an audio file
            if any(file.endswith(ext) for ext in audio_extensions):
                # Get full path to original file
                input_file_path = os.path.join(root, file)
                
                # Create corresponding path in output directory
                relative_path = os.path.relpath(input_file_path, input_root_dir)
                output_file_path = os.path.join(output_root_dir, relative_path)
                
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(output_file_path)
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                
                try:
                    # Load audio with original sample rate
                    audio, orig_sr = librosa.load(input_file_path, sr=None)
                    
                    print(f"Resampling: {relative_path} ({orig_sr}Hz → {target_sr}Hz)")
                    
                    # Resample audio
                    audio_resampled = librosa.resample(
                        audio, 
                        orig_sr=orig_sr, 
                        target_sr=target_sr
                    )
                    
                    # Save resampled audio
                    sf.write(output_file_path, audio_resampled, target_sr)
                    successful_files += 1
                    
                except Exception as e:
                    print(f"❌ Error processing {relative_path}: {str(e)}")
                    failed_files.append(relative_path)
    
    # Print summary
    print("\n" + "="*50)
    print("RESAMPLING SUMMARY:")
    print(f"✅ Successfully resampled: {successful_files} files")
    print(f"❌ Failed: {len(failed_files)} files")
    print(f"Original structure: {input_root_dir}")
    print(f"New structure: {output_root_dir}")
    print(f"Target sample rate: {target_sr}Hz")
    
    return successful_files, failed_files

# MAIN EXECUTION
if __name__ == "__main__":
    # Set your paths here
    INPUT_ROOT_DIR = "data/Original_Dataset"  # Change this!
    OUTPUT_ROOT_DIR = "data/Corrected_Dataset"  # Change this!
    
    # Example structure:
    # INPUT_ROOT_DIR/
    # ├── non_depressed/
    # │   ├── patient1.wav
    # │   └── patient2.wav
    # ├── mid_depressed/
    # │   ├── patient3.wav
    # │   └── patient4.wav
    # └── high_depressed/
    #     ├── patient5.wav
    #     └── patient6.wav
    
    # Run resampling
    successful, failed = resample_preserve_structure(INPUT_ROOT_DIR, OUTPUT_ROOT_DIR)
    
    # The output will have the exact same folder structure:
    # OUTPUT_ROOT_DIR/
    # ├── non_depressed/
    # │   ├── patient1.wav (16kHz)
    # │   └── patient2.wav (16kHz)
    # ├── mid_depressed/
    # │   ├── patient3.wav (16kHz)
    # │   └── patient4.wav (16kHz)
    # └── high_depressed/
    #     ├── patient5.wav (16kHz)
    #     └── patient6.wav (16kHz)