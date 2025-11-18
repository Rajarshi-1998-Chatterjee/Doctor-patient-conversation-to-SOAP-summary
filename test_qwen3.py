"""Test fine-tuned Qwen3-1.7B model on medical SOAP note generation from test.json."""

import json
import logging
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import re

# ============================================================================
# CONFIGURATION
# ============================================================================
# Model and data paths - MODIFY THESE VARIABLES
model_path = "./qwen3_1.7B_soap_finetuned/final_model"  # Path to fine-tuned model
test_path = "./test.json"  # Path to test data
results_path = "./qwen3_test_results.json"  # Path to save results

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 256

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from file."""
    logger.info(f"Loading data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples")
    return data


def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Try loading as a complete model first
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("Loaded complete fine-tuned model")
    except Exception as e:
        logger.warning(f"Could not load as complete model: {e}")
        logger.info("Attempting to load as LoRA adapter...")
        
        # Try loading as LoRA adapter
        base_model_path = "Qwen/Qwen3-1.7B"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info("Loaded LoRA adapter on base model")
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    logger.info("Model loaded successfully and set to evaluation mode")
    
    return model, tokenizer


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU, ROUGE-1, ROUGE-L, and BERTScore metrics."""
    logger.info("\nCalculating evaluation metrics...")
    
    # Initialize smoothing function for BLEU
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU scores
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # Tokenize by words
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        # Calculate BLEU with smoothing
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
    
    avg_bleu = np.mean(bleu_scores)
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rougeL = np.mean(rougeL_scores)
    
    # Calculate BERTScore
    logger.info("Computing BERTScore (this may take a while)...")
    P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
    avg_bertscore = F1.mean().item()
    
    metrics = {
        'bleu': avg_bleu,
        'rouge1': avg_rouge1,
        'rougeL': avg_rougeL,
        'bertscore_f1': avg_bertscore
    }
    
    return metrics


def generate_soap_note(model, tokenizer, dialogue: str) -> str:
    """Generate SOAP note for a given dialogue."""
    input_text = f"""<|im_start|>system
You are an expert medical assistant that generates SOAP notes from dialogues.<|im_end|>
<|im_start|>user
Create a Medical SOAP note summary from the dialogue, following these guidelines:
    S (Subjective): Summarize the patient's reported symptoms, including chief complaint and relevant history. Rely on the patient's statements as the primary source and ensure standardized terminology.
    O (Objective): Highlight critical findings such as vital signs, lab results, and imaging, emphasizing important details like the side of the body affected and specific dosages. Include normal ranges where relevant.
    A (Assessment): Offer a concise assessment combining subjective and objective data. State the primary diagnosis and any differential diagnoses, noting potential complications and the prognostic outlook.
    P (Plan): Outline the management plan, covering medication, diet, consultations, and education. Ensure to mention necessary referrals to other specialties and address compliance challenges.
    Considerations: Compile the report based solely on the transcript provided. Maintain confidentiality and document sensitively. Use concise medical jargon and abbreviations for effective doctor communication.
    Please format the summary in a clean, simple list format without using markdown or bullet points. Use 'S:', 'O:', 'A:', 'P:' directly followed by the text. Avoid any styling or special characters.
{dialogue}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in generated_text:
        soap_note = generated_text.split("<|im_start|>assistant")[-1]
        soap_note = soap_note.replace("<|im_end|>", "").strip()
        
        # Remove think tokens and their content
        # Remove <think>...</think> blocks
        soap_note = re.sub(r'<think>.*?</think>', '', soap_note, flags=re.DOTALL)
        # Remove any remaining think tags
        soap_note = soap_note.replace('<think>', '').replace('</think>', '')
        soap_note = soap_note.strip()
        
        return soap_note
    
    return generated_text


def test_model():
    """Main function to test the fine-tuned Qwen3 model."""
    logger.info("="*80)
    logger.info("Qwen3-1.7B Medical SOAP Note Generation - Testing Pipeline")
    logger.info("="*80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        logger.info(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("CUDA not available. Using CPU.")
    
    # Load test data
    test_data = load_json_data(test_path)
    logger.info(f"Test examples: {len(test_data)}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Generate predictions
    logger.info("\nGenerating predictions...")
    results = []
    
    for i, example in enumerate(tqdm(test_data, desc="Testing")):
        dialogue = example.get("dialogue", "")
        ground_truth_soap = example.get("soap", "")
        
        # Generate prediction
        predicted_soap = generate_soap_note(model, tokenizer, dialogue)
        
        # Store result
        result = {
            "example_id": i,
            "dialogue": dialogue,
            "ground_truth_soap": ground_truth_soap,
            "predicted_soap": predicted_soap,
        }
        results.append(result)
        
        # Log first few examples
        if i < 3:
            logger.info(f"\n{'='*80}")
            logger.info(f"Example {i+1}")
            logger.info(f"{'='*80}")
            logger.info(f"\nDialogue (first 200 chars):\n{dialogue[:200]}...")
            logger.info(f"\nGround Truth SOAP Note:\n{ground_truth_soap}")
            logger.info(f"\nPredicted SOAP Note:\n{predicted_soap}")
            logger.info(f"{'='*80}\n")
    
    # Save results
    logger.info(f"\nSaving results to {results_path}")
    results_dir = Path(results_path).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Results saved successfully!")
    
    # Calculate evaluation metrics
    predictions = [r['predicted_soap'] for r in results]
    references = [r['ground_truth_soap'] for r in results]
    
    metrics = calculate_metrics(predictions, references)
    
    # Generate summary statistics
    logger.info("\n" + "="*80)
    logger.info("Test Summary")
    logger.info("="*80)
    logger.info(f"Total examples tested: {len(results)}")
    logger.info(f"Results saved to: {results_path}")
    
    # Calculate average lengths
    avg_ground_truth_len = sum(len(r['ground_truth_soap']) for r in results) / len(results)
    avg_predicted_len = sum(len(r['predicted_soap']) for r in results) / len(results)
    
    logger.info(f"Average ground truth SOAP note length: {avg_ground_truth_len:.1f} characters")
    logger.info(f"Average predicted SOAP note length: {avg_predicted_len:.1f} characters")
    
    # Display evaluation metrics
    logger.info("\n" + "="*80)
    logger.info("Evaluation Metrics")
    logger.info("="*80)
    logger.info(f"BLEU Score: {metrics['bleu']:.4f}")
    logger.info(f"ROUGE-1 F1: {metrics['rouge1']:.4f}")
    logger.info(f"ROUGE-L F1: {metrics['rougeL']:.4f}")
    logger.info(f"BERTScore F1: {metrics['bertscore_f1']:.4f}")
    logger.info("="*80)
    
    # Save metrics to file
    metrics_path = results_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\nMetrics saved to: {metrics_path}")
    
    logger.info("\nTesting complete!")
    logger.info("="*80)
    
    return results, metrics


if __name__ == "__main__":
    test_model()
