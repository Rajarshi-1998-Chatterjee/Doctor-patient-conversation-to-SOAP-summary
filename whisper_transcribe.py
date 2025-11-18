"""Batch transcription script using Hugging Face Whisper large-v3."""

from pathlib import Path
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ============================================================================
# CONFIGURATION - Set these variables to your actual paths
# ============================================================================
AUDIO_FILES_DIRECTORY = "/content/audio_data"  # Folder containing audio files
TRANSCRIPTS_DIRECTORY = "/content/transcripts"  # Folder where transcripts will be saved

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_ID = "openai/whisper-large-v3"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".avi"}


def setup_model():
    """Load and configure the Whisper model for CUDA if available."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading Whisper large-v3 model on {device}...")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    
    print(f"Model loaded successfully on {device}")
    return pipe


def get_audio_files(audio_dir):
    """Get all audio files from the directory."""
    audio_path = Path(audio_dir)
    audio_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(audio_path.rglob(f"*{ext}"))
    
    return sorted(audio_files)


def transcribe_audio_files():
    """Main function to transcribe all audio files."""
    audio_dir = Path(AUDIO_FILES_DIRECTORY)
    transcript_dir = Path(TRANSCRIPTS_DIRECTORY)
    
    # Create transcript directory if it doesn't exist
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all audio files
    audio_files = get_audio_files(audio_dir)
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        print(f"Looking for files with extensions: {SUPPORTED_EXTENSIONS}")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to transcribe")
    
    # Load the model
    asr_pipeline = setup_model()
    
    # Process each audio file
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"\n[{idx}/{len(audio_files)}] Processing: {audio_file.name}")
        
        # Generate output filename
        relative_path = audio_file.relative_to(audio_dir)
        transcript_path = transcript_dir / relative_path.with_suffix(".txt")
        
        # Create subdirectories if needed
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if transcript already exists
        if transcript_path.exists():
            print(f"  → Transcript already exists, skipping: {transcript_path.name}")
            continue
        
        try:
            # Transcribe
            print(f"  → Transcribing...")
            result = asr_pipeline(
                str(audio_file),
                return_timestamps=True,
                generate_kwargs={"task": "transcribe"}
            )
            
            # Extract and save transcript
            transcript_text = result["text"].strip()
            transcript_path.write_text(transcript_text + "\n", encoding="utf-8")
            
            print(f"  ✓ Saved transcript to: {transcript_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_file.name}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("Transcription complete!")
    print(f"Transcripts saved to: {transcript_dir}")


if __name__ == "__main__":
    transcribe_audio_files()
