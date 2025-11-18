"""Fine-tune Whisper large-v3 on custom audio-transcript pairs."""

from pathlib import Path
from dataclasses import dataclass
import torch
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
AUDIO_FILES_DIRECTORY = "/content/drive/MyDrive/Data/audio_recordings/Audio_Recordings"
TRANSCRIPTS_DIRECTORY = "/content/drive/MyDrive/Data/audio_recordings/Clean_Transcripts"
OUTPUT_MODEL_DIRECTORY = "/content/drive/MyDrive/MED_whisper_large_v3-finetuned"

MODEL_ID = "openai/whisper-large-v3-turbo"
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".avi", ".m4a"}
TRANSCRIPT_EXTENSIONS = {".txt", ".json"}
JSON_TRANSCRIPT_KEY = "text"
LANGUAGE = "english"
TASK = "transcribe"

# Training parameters
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
MAX_STEPS = 2000
EVAL_STEPS = 500
SAVE_STEPS = 500


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for padding audio features and labels."""
    processor: WhisperProcessor

    def __call__(self, features):
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def find_audio_transcript_pairs(audio_dir, transcript_dir):
    """Find matching audio and transcript file pairs."""
    audio_path = Path(audio_dir)
    transcript_path = Path(transcript_dir)
    pairs = []

    # Find all audio files
    audio_files = []
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        audio_files.extend(audio_path.rglob(f"*{ext}"))

    print(f"Found {len(audio_files)} audio files")

    # Match with transcripts
    for audio_file in sorted(audio_files):
        relative_path = audio_file.relative_to(audio_path)
        
        for transcript_ext in TRANSCRIPT_EXTENSIONS:
            transcript_file = transcript_path / relative_path.with_suffix(transcript_ext)
            
            if transcript_file.exists():
                # Load transcript
                if transcript_ext == ".txt":
                    transcript = transcript_file.read_text(encoding="utf-8").strip()
                else:  # .json
                    import json
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        transcript = data.get(JSON_TRANSCRIPT_KEY, "").strip() if isinstance(data, dict) else str(data).strip()
                
                if transcript:
                    pairs.append({
                        "audio": str(audio_file),
                        "transcript": transcript
                    })
                    break

    print(f"Found {len(pairs)} matching audio-transcript pairs")
    return pairs


def prepare_dataset(batch, processor):
    """Process audio and transcript for training."""
    audio = batch["audio"]
    
    # Compute input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode transcript
    batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
    
    return batch


def main():
    """Main training function."""
    print("="*60)
    print("Whisper Fine-tuning - Simplified Version")
    print("="*60)

    # Load processor and model
    print(f"\nLoading model: {MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")

    # Find and prepare data
    print("\nSearching for audio-transcript pairs...")
    pairs = find_audio_transcript_pairs(AUDIO_FILES_DIRECTORY, TRANSCRIPTS_DIRECTORY)
    
    if not pairs:
        print("Error: No audio-transcript pairs found!")
        return

    # Split into train/eval (80-20)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]
    
    print(f"\nTraining samples: {len(train_pairs)}")
    print(f"Evaluation samples: {len(eval_pairs)}")

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = Dataset.from_list(train_pairs)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    train_dataset = train_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=train_dataset.column_names,
        desc="Processing training data"
    )

    eval_dataset = Dataset.from_list(eval_pairs)
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.map(
        lambda x: prepare_dataset(x, processor),
        remove_columns=eval_dataset.column_names,
        desc="Processing evaluation data"
    )

    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Evaluation dataset: {len(eval_dataset)} samples")

    # Setup training
    output_dir = Path(OUTPUT_MODEL_DIRECTORY)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=200,
        max_steps=MAX_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()
