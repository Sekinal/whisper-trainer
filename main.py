from unsloth import FastModel
from transformers import WhisperForConditionalGeneration
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "openai/whisper-large-v3-turbo",
    dtype = None, # Leave as None for auto detection
    load_in_4bit = False, # Set to True to do 4bit quantization which reduces memory
    auto_model = WhisperForConditionalGeneration,
    whisper_language = "Spanish",
    whisper_task = "transcribe",
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = None, # ** MUST set this for Whisper **
)


import numpy as np
import tqdm

#Set this to the language you want to train on
model.generation_config.language = "<|es|>"
model.generation_config.task = "transcribe"
model.config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None

def formatting_prompts_func(examples): # The input is now plural 'examples'
    # The tokenizer's feature_extractor and tokenizer both handle lists of inputs
    audio_arrays = [x["array"] for x in examples["audio"]]
    sampling_rates = [x["sampling_rate"] for x in examples["audio"]]
    
    features = tokenizer.feature_extractor(
        audio_arrays, sampling_rate=sampling_rates[0] # Assuming all are the same
    )
    
    tokenized_text = tokenizer.tokenizer(examples["sentence"])
    
    # The return dictionary now contains lists of features and labels
    return {
        "input_features": features.input_features,
        "labels": tokenized_text.input_ids,
    }

from datasets import load_dataset, Audio, DatasetDict
# 1. Load the original dataset
# This will still download the dataset metadata, but we'll only work with a small part.
full_dataset = load_dataset("Thermostatic/CommonVoice-17.0-Spanish")

# 2. Create a small subset for debugging
# We'll take the first 100 samples from the train and validation splits.
# Using .select() is the standard way to do this.
train_subset = full_dataset["train"].select(range(10000))
validation_subset = full_dataset["validation"].select(range(100))
test_subset = full_dataset["test"].select(range(100))

# Create a new DatasetDict with just our subsets
subset_dataset = DatasetDict({
    "train": train_subset,
    "validation": validation_subset,
    "test": test_subset
})

# 3. Cast the audio column to the correct sampling rate for our subset
subset_dataset = subset_dataset.cast_column("audio", Audio(sampling_rate=16000))

# 4. Apply the mapping function ONLY to our small subset
# This will be much faster and allows you to check for errors quickly.
processed_dataset = subset_dataset.map(
    formatting_prompts_func,
    batched=True,
    batch_size=1024,
    remove_columns=subset_dataset["train"].column_names,
)

print("\nProcessed subset splits:")
print(processed_dataset)

# @title Create compute_metrics and datacollator
import evaluate
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from unsloth import is_bfloat16_supported

trainer = Seq2SeqTrainer(
    model = model,
    train_dataset = processed_dataset["train"],
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer),
    eval_dataset = processed_dataset["validation"],
    tokenizer = tokenizer.feature_extractor,
    args = Seq2SeqTrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 1e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        lr_scheduler_type = "linear",
        label_names = ['labels'],
        eval_steps = 5 ,
        eval_strategy="steps",
        seed = 3407,
        output_dir = "outputs",
        report_to = "wandb", # Use this for WandB etc
        load_best_model_at_end=True,
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

print("\n--- Starting Final Evaluation on the Test Set ---")

# Import necessary libraries for the manual loop
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import torch

# 1. OPTIMIZE THE MODEL FOR INFERENCE WITH UNSLOTH
# This is the key step! It merges LoRA adapters for faster inference.
print("Optimizing model for inference with Unsloth...")
FastModel.for_inference(model)
model.eval()

# 2. SET UP THE DATALOADER (Same as before)
# Use the same data collator as the trainer
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer)

# Create a DataLoader for the test set
# You can use a larger batch_size now due to inference optimizations
test_dataloader = DataLoader(
    processed_dataset["test"],
    batch_size=16, # Increased batch size for faster processing
    collate_fn=data_collator
)

# Load the WER metric from evaluate
wer_metric = evaluate.load("wer")

# Prepare lists to store all predictions and references
all_predictions = []
all_references = []

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# The model is already on the device from the trainer

# 3. RUN THE BATCHED INFERENCE LOOP (Now much faster)
# torch.no_grad() is used to disable gradient calculations
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating WER on test set"):
        # Move batch to the correct device
        input_features = batch["input_features"].to(device)
        label_ids = batch["labels"].to(device)

        # Generate predictions using the optimized model
        # This model.generate() call is now significantly faster
        predicted_ids = model.generate(input_features)

        # Decode the predicted IDs and label IDs into text
        # For labels, we replace -100 with the pad token ID before decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        predictions_text = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        references_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Add the decoded texts to our lists
        all_predictions.extend(predictions_text)
        all_references.extend(references_text)

# 4. COMPUTE AND PRINT THE FINAL WER (Same as before)
final_wer = 100 * wer_metric.compute(predictions=all_predictions, references=all_references)

print("\n--- Test Set Evaluation Results ---")
print(f"Word Error Rate (WER): {final_wer:.2f}%")

# Optional: Print a few examples to see the quality
print("\n--- Example Transcriptions ---")
for i in range(min(5, len(all_predictions))):
    print(f"Reference:  {all_references[i]}")
    print(f"Prediction: {all_predictions[i]}")
    print("-" * 20)