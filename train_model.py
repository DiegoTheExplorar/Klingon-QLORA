import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def load_model(tokenizer):
    print("Loading GPT-2 model...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Resize token embeddings to account for the new padding token
    model.resize_token_embeddings(len(tokenizer))
    print("Resized token embeddings.")
    
    return model

def apply_lora(model):
    print("Defining LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"],
    )
    
    print("Applying LoRA to the model...")
    lora_model = get_peft_model(model, lora_config)
    
    return lora_model

def train_model(lora_model, tokenized_dataset):
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
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
            'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
            'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])
        }
    )
    
    print("Starting model fine-tuning...")
    trainer.train()

def save_model(lora_model, tokenizer):
    print("Saving the fine-tuned model...")
    lora_model.save_pretrained("./gpt2-klingon-finetuned")
    tokenizer.save_pretrained("./gpt2-klingon-finetuned")
