"""Fine-tune HuggingFaceTB/SmolLM3-3B on SOAP summary generation from medical dialogues."""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import gc

# ============================================================================
# CONFIGURATION - Optimized for Google Colab with CUDA
# ============================================================================
# Model and data paths - MODIFY THESE VARIABLES
model_path = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
train_path = "/content/train.json"
validation_path = "/content/validation.json"
OUTPUT_MODEL_DIRECTORY = "/content/smollm_soap_finetuned"

# Full precision training (quantization disabled)
USE_4BIT = False

# ============================================================================
# Training Configuration
# ============================================================================
# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training hyperparameters - Optimized for Colab T4/V100 GPU
BATCH_SIZE = 2  # Reduced for full precision training
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
EVAL_STEPS = 100
SAVE_STEPS = 100
LOGGING_STEPS = 10

# Memory optimization settings
MAX_GRAD_NORM = 0.3
GROUP_BY_LENGTH = True
DATALOADER_PIN_MEMORY = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
                    continue
    logger.info(f"Loaded {len(data)} examples")
    return data


def format_prompt_for_training(example: Dict) -> str:
    """Format a single example into the training prompt format."""
    dialogue = example.get("dialogue", "")
    soap = example.get("soap", "")
    
    # Use the exact prompt structure from the dataset
    prompt = example.get("prompt", "")
    if not prompt:
        # Default prompt if not provided
        prompt = """Create a Medical SOAP note summary from the dialogue, following these guidelines:
    S (Subjective): Summarize the patient's reported symptoms, including chief complaint and relevant history. Rely on the patient's statements as the primary source and ensure standardized terminology.
    O (Objective): Highlight critical findings such as vital signs, lab results, and imaging, emphasizing important details like the side of the body affected and specific dosages. Include normal ranges where relevant.
    A (Assessment): Offer a concise assessment combining subjective and objective data. State the primary diagnosis and any differential diagnoses, noting potential complications and the prognostic outlook.
    P (Plan): Outline the management plan, covering medication, diet, consultations, and education. Ensure to mention necessary referrals to other specialties and address compliance challenges.
    Considerations: Compile the report based solely on the transcript provided. Maintain confidentiality and document sensitively. Use concise medical jargon and abbreviations for effective doctor communication.
    Please format the summary in a clean, simple list format without using markdown or bullet points. Use 'S:', 'O:', 'A:', 'P:' directly followed by the text. Avoid any styling or special characters."""
    
    # Format as instruction-following conversation
    full_prompt = f"""<|im_start|>system
You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting.<|im_end|>
<|im_start|>user
{prompt}

### Dialogue:
{dialogue}<|im_end|>
<|im_start|>assistant
{soap}<|im_end|>"""
    
    return full_prompt


def prepare_dataset(data: List[Dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""
    logger.info("Formatting examples...")
    formatted_texts = []
    
    for example in tqdm(data, desc="Formatting"):
        text = format_prompt_for_training(example)
        formatted_texts.append({"text": text})
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_texts)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    return tokenized_dataset


def setup_model_and_tokenizer():
    """Load and prepare model and tokenizer for full precision training."""
    logger.info(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model: {model_path} (full precision)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Configure LoRA for parameter-efficient fine-tuning
    logger.info("Configuring LoRA for parameter-efficient fine-tuning...")
    model.enable_input_require_grads()
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def generate_soap_summary(model, tokenizer, dialogue: str, prompt: str = None) -> str:
    """Generate SOAP summary for a given dialogue."""
    if prompt is None:
        prompt = """Create a Medical SOAP note summary from the dialogue, following these guidelines:
    S (Subjective): Summarize the patient's reported symptoms, including chief complaint and relevant history. Rely on the patient's statements as the primary source and ensure standardized terminology.
    O (Objective): Highlight critical findings such as vital signs, lab results, and imaging, emphasizing important details like the side of the body affected and specific dosages. Include normal ranges where relevant.
    A (Assessment): Offer a concise assessment combining subjective and objective data. State the primary diagnosis and any differential diagnoses, noting potential complications and the prognostic outlook.
    P (Plan): Outline the management plan, covering medication, diet, consultations, and education. Ensure to mention necessary referrals to other specialties and address compliance challenges.
    Considerations: Compile the report based solely on the transcript provided. Maintain confidentiality and document sensitively. Use concise medical jargon and abbreviations for effective doctor communication.
    Please format the summary in a clean, simple list format without using markdown or bullet points. Use 'S:', 'O:', 'A:', 'P:' directly followed by the text. Avoid any styling or special characters."""
    
    input_text = f"""<|im_start|>system
You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting.<|im_end|>
<|im_start|>user
{prompt}

### Dialogue:
{dialogue}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in generated_text:
        soap_summary = generated_text.split("<|im_start|>assistant")[-1]
        soap_summary = soap_summary.replace("<|im_end|>", "").strip()
        return soap_summary
    
    return generated_text


def validate_model(model, tokenizer, validation_data: List[Dict], num_samples: int = 5):
    """Validate model on sample validation examples."""
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SAMPLES")
    logger.info(f"{'='*80}\n")
    
    model.eval()
    
    for i, example in enumerate(validation_data[:num_samples]):
        dialogue = example.get("dialogue", "")
        ground_truth_soap = example.get("soap", "")
        
        logger.info(f"\n{'─'*80}")
        logger.info(f"Sample {i+1}/{num_samples}")
        logger.info(f"{'─'*80}")
        logger.info(f"\nDialogue:\n{dialogue[:200]}...")
        logger.info(f"\nGround Truth SOAP:\n{ground_truth_soap}")
        
        generated_soap = generate_soap_summary(model, tokenizer, dialogue)
        logger.info(f"\nGenerated SOAP:\n{generated_soap}")
        logger.info(f"{'─'*80}\n")


def finetune_smollm():
    """Main function to fine-tune SmolLM3-3B on Google Colab with CUDA."""
    logger.info("="*80)
    logger.info("SmolLM3-3B Fine-tuning for SOAP Summary Generation (Colab-Optimized)")
    logger.info("="*80)
    
    # Check CUDA availability and GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available! Training will be slow on CPU.")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load data
    train_data = load_json_data(train_path)
    validation_data = load_json_data(validation_path)
    
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Validation examples: {len(validation_data)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    output_dir = Path(OUTPUT_MODEL_DIRECTORY)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        eval_strategy="no",  # Disable validation during training
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        max_grad_norm=MAX_GRAD_NORM,
        group_by_length=GROUP_BY_LENGTH,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        ddp_find_unused_parameters=False,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("="*80)
    logger.info("Starting fine-tuning...")
    logger.info("="*80)
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    trainer.train()
    
    # Clear cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final model
    logger.info("\nSaving final model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    logger.info("="*80)
    logger.info("Fine-tuning complete!")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info("="*80)
    
    # Validate on sample examples
    logger.info("\nRunning validation on sample examples...")
    validate_model(model, tokenizer, validation_data, num_samples=5)
    
    logger.info("\nAll done! You can now use the fine-tuned model for SOAP summary generation.")


if __name__ == "__main__":
    finetune_smollm()
