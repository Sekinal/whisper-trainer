#!/usr/bin/env python

import os
from functools import partial
from datasets import load_dataset, Audio, DatasetDict
from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
from huggingface_hub import HfApi, HfFolder

# --- CONFIGURATION ---
# IMPORTANT: Fill these variables before running the script.

# 1. Your Hugging Face Hub username and the desired name for the new dataset repository.
#    The format must be "YourUsername/YourNewDatasetName".
DEST_REPO_ID = "Thermostatic/CommonVoice-17.0-Spanish-Filtered"

# --- SCRIPT ---

def formatting_prompts_func(examples, tokenizer):
    """
    Processes a batch of audio examples to create input_features and labels.
    This function is passed to the .map() method.
    """
    # The tokenizer's feature_extractor and tokenizer both handle lists of inputs
    audio_arrays = [x["array"] for x in examples["audio"]]
    
    # The feature extractor assumes all samples in a batch have the same sampling rate
    sampling_rates = [x["sampling_rate"] for x in examples["audio"]]
    
    features = tokenizer.feature_extractor(
        audio_arrays, sampling_rate=sampling_rates[0]
    )
    
    tokenized_text = tokenizer.tokenizer(examples["sentence"])
    
    # The return dictionary now contains lists of features and labels
    return {
        "input_features": features.input_features,
        "labels": tokenized_text.input_ids,
    }

def main():
    """
    Main function to download, process, and re-upload the dataset.
    """
    # Step 1: Initialize the correct tokenizer from the base model
    # We need the tokenizer to correctly process the audio (feature extraction) and text.
    # The tokenizer object from Unsloth's FastModel conveniently includes both parts.
    print("Initializing tokenizer from 'openai/whisper-large-v3-turbo'...")
    _ , tokenizer = FastModel.from_pretrained(
        model_name="openai/whisper-large-v3-turbo",
        dtype=None,
        load_in_4bit=False,
        auto_model=WhisperForConditionalGeneration,
        whisper_language="Spanish",
        whisper_task="transcribe",
        token=HF_TOKEN,
    )
    print("Tokenizer initialized.")

    # Step 2: Load the original dataset from the Hub
    original_dataset_id = "Thermostatic/CommonVoice-17.0-Spanish-Filtered"
    print(f"Loading original dataset: {original_dataset_id}...")
    full_dataset = load_dataset(original_dataset_id, token=HF_TOKEN)
    print("Dataset loaded.")

    # Step 3: Resample the audio column to the required 16kHz
    # The Whisper model family requires audio to be at a 16,000 Hz sampling rate.
    print("Resampling audio column to 16kHz...")
    subset_dataset = full_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("Audio resampled.")

    # Step 4: Apply the formatting function across the entire dataset
    # This crucial step converts raw audio/text into model-ready `input_features` and `labels`.
    print("Applying processing function to all splits (this may take a while)...")
    
    # We use functools.partial to pass the tokenizer to our mapping function
    processing_function = partial(formatting_prompts_func, tokenizer=tokenizer)
    
    processed_dataset = subset_dataset.map(
        processing_function,
        batched=True,
        batch_size=2048,  # A large batch size speeds up the mapping process
        remove_columns=subset_dataset["train"].column_names, # Remove old columns to keep it clean
        num_proc=max(1, os.cpu_count() // 2), # Use multiple CPU cores for speed
    )
    print("\nDataset processed successfully:")
    print(processed_dataset)

    # Step 5: Upload the final, processed dataset to the Hugging Face Hub
    print(f"\nUploading processed dataset to '{DEST_REPO_ID}'...")
    processed_dataset.push_to_hub(
        repo_id=DEST_REPO_ID,
        token=HF_TOKEN,
        private=True,  # It's good practice to start with a private repo
    )
    print("✅ Upload complete!")
    print(f"You can find your processed dataset at: https://huggingface.co/datasets/{DEST_REPO_ID}")

if __name__ == "__main__":
    main()
