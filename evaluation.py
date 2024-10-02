from datasets import load_metric
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel
import torch
import pandas as pd


bleu_metric = load_metric("bleu")

# Load the dataset and use only the first 100 rows
print("Loading evaluation dataset...")
data = pd.read_csv('data.csv').head(100)

# Prepare the evaluation data
eval_data = data.apply(lambda row: f"Translate English to Klingon: {row['english']} ->", axis=1)
reference_data = data['klingon'].tolist()

# Load fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer for evaluation...")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-klingon-finetuned")
model = GPT2LMHeadModel.from_pretrained("./gpt2-klingon-finetuned")
lora_model = PeftModel.from_pretrained(model, "./gpt2-klingon-finetuned")


lora_model.eval()
lora_model.half()  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

# Function to generate Klingon translation from English input
def generate_translation(english_text):
    input_text = english_text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        generated_ids = lora_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_return_sequences=1,
            do_sample=True,  # Sampling to get diverse outputs
            top_p=0.95,
            temperature=0.7,
        )
    
    # Decode the generated output
    klingon_translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return klingon_translation.replace(input_text, "").strip()

# Perform evaluation and compute BLEU score
print("Starting evaluation...")

predictions = []
references = [[ref.split()] for ref in reference_data] 

# Generate predictions and collect references
for i, text in enumerate(eval_data):
    generated_translation = generate_translation(text)
    print(f"Example {i}:")
    print(f"English Input: {text}")
    print(f"Generated Klingon Translation: {generated_translation}")
    print(f"Reference Klingon Translation: {reference_data[i]}")
    predictions.append(generated_translation.split())  # Tokenize prediction for BLEU

# Compute BLEU score
print("Computing BLEU score...")
bleu_score = bleu_metric.compute(predictions=predictions, references=references)

# Output BLEU score
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
