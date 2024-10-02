from transformers import GPT2Tokenizer

def setup_tokenizer():
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add padding token to the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    print("Added padding token to the tokenizer.")
    
    return tokenizer
