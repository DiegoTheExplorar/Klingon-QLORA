import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch

print("Starting the script...")

# Load the dataset
print("Loading the dataset...")
data = pd.read_csv('data.csv')

# Format the data for translation task
print("Formatting the data...")
formatted_data = data.apply(lambda row: f"Translate English to Klingon: {row['english']} -> {row['klingon']}", axis=1)

# Load GPT-2 tokenizer
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token
print("Added padding token to the tokenizer.")

# Create Hugging Face dataset
print("Creating Hugging Face dataset...")
dataset = Dataset.from_pandas(pd.DataFrame({'text': formatted_data}))

# Tokenize the dataset
print("Tokenizing the dataset...")
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
print("Dataset tokenized.")

# Load GPT-2 model
print("Loading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Resize token embeddings to account for the new padding token
model.resize_token_embeddings(len(tokenizer))
print("Resized token embeddings.")

# Define LoRA configuration
print("Defining LoRA configuration...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"],
)

# Apply LoRA to the model
print("Applying LoRA to the model...")
lora_model = get_peft_model(model, lora_config)

# Set up training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./gpt2-klingon-finetuned",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=5,
    save_steps=500,
    fp16=True,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
)

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
                                'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
                                'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])},
)

# Fine-tune the model
print("Starting model fine-tuning...")
trainer.train()

# Save the LoRA-adapted fine-tuned model
print("Saving the fine-tuned model...")
lora_model.save_pretrained("./gpt2-klingon-finetuned")
tokenizer.save_pretrained("./gpt2-klingon-finetuned")

print("Script completed successfully!")