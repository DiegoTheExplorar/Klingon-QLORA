import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# Load the fine-tuned model and tokenizer
print("Loading fine-tuned model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-klingon-finetuned")
model = GPT2LMHeadModel.from_pretrained("./gpt2-klingon-finetuned")

# Apply the LoRA configuration to the model
lora_model = PeftModel.from_pretrained(model, "./gpt2-klingon-finetuned")

lora_model.eval()
# Use half precision for inference
lora_model.half()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

def translate_to_klingon(english_text):
    # Prepare the input text
    input_text = f"Translate English to Klingon: {english_text} ->"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = lora_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,  # Top-p sampling for better variety
            temperature=0.7,  # Control the randomness of predictions
        )
    
    # Decode the generated output
    klingon_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return klingon_translation.replace(input_text, "").strip() 


english_phrase = "Hello, how are you?"
klingon_translation = translate_to_klingon(english_phrase)
print(f"Klingon Translation: {klingon_translation}")
