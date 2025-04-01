from pathlib import Path
from typing import List, Dict
import torch
from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   BitsAndBytesConfig,
   TrainingArguments,
   Trainer,
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from datasets import Dataset
import json


# Source: https://github.com/mediar-ai/screenpipe/issues/994

# Used Gemini to ask questions on how to set up fine tuning: https://gemini.google.com/app/ed1e432b1fd58386

# Dataset 1: https://huggingface.co/datasets/Mahesh2841/Agriculture
# Dataset 2: https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only

# Fine Tuned: https://huggingface.co/czalpha/deepseek_coder_1.5b_fine_tuned


# Read the API token from the secrets file
secrets_path = Path("./secrets/secrets.txt")

if secrets_path.exists():
   HF_TOKEN = secrets_path.read_text().strip()
else:
   raise FileNotFoundError(f"Secrets file not found: {secrets_path}")

# Login to Hugging Face
login(HF_TOKEN)

def load_data_from_files(file_path: str) -> List[Dict[str, str]]:
   print(f"[+] Starting to load data from dataset: {file_path}")
   with open(file_path, 'r', encoding='utf-8') as f:
      data = json.load(f)
   print(f"[+] Finished loading data from dataset: {file_path}.")
   return data


def format_instruction(sample):
   return f"<s>[INST] {sample['instruction']} {sample['input']} [/INST] {sample['response']} </s>"


def main():
   print(f"[+] Main function starting...")
   # Configuration
   model_name = "mistralai/Mistral-7B-v0.3"
   output_dir = "./fine_tuned_model"
   
   print("[+] Main function calling training dataset importation...")
   # Data Importing
   training_data = load_data_from_files("./data/agricultural_dataset.json")
   dataset = Dataset.from_list(training_data)
   # Apply map to each individual sample instead of iterating inside the lambda
   dataset = dataset.map(lambda sample: {"text": format_instruction(sample)})
   print("[+] Main function dataset importing finished.")
   
   # Defining the model (without 4-bit quantization)
   model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="cpu", 
      trust_remote_code=True,
      token=HF_TOKEN
   )
   
   # Explicitly tell it to use CPU if already didn't
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   
   # Tokenization
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   tokenizer.pad_token = tokenizer.eos_token
   
   print("[+] Starting LORA configuration.")
   # LORA Configuration
   lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
   )
   
   print("[+] Starting to train the model.")
   # Model Training
   model = get_peft_model(model, lora_config)
   model.print_trainable_parameters()
   
   training_args = TrainingArguments(
      output_dir=output_dir,
      num_train_epochs=3,
      per_device_train_batch_size=1,
      gradient_accumulation_steps=8,
      learning_rate=2e-4,
      fp16=False,
      logging_steps=10,
      save_strategy="epoch",
      remove_unused_columns=False,
      label_names=["labels"]  
   )
   
   def preprocess_function(example):
      full_input = example["instruction"] + " " + example["input"]
      inputs = tokenizer(full_input, padding="max_length", truncation=True, max_length=512)
      labels = tokenizer(example["response"], padding="max_length", truncation=True, max_length=512).input_ids
      inputs["labels"] = labels
      return inputs
   
   # Convert dataset to correct format
   dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "response"])
   
   trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=dataset,
      data_collator=lambda data: {
         "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
         "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
         "labels": torch.stack([torch.tensor(f["labels"]) for f in data]),
      },
   )
   
   trainer.train()
   trainer.save_model()
   trainer.push_to_hub("czalpha/deepseek_coder_1.5b_fine_tuned")
   model.save_pretrained("./deepseek_coder_1.5b_fine_tuned_local")
   tokenizer.save_pretrained("./deepseek_coder_1.5b_fine_tuned_local")
   print(f"[+] Main function ended.")


if __name__ == "__main__":
   print("[+] Starting up...")
   main()
   print("[+] Shutting down...")