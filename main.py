from data_processing import load_and_format_data
from tokenizer_setup import setup_tokenizer
from dataset_preparation import prepare_dataset
from train_model import load_model, apply_lora, train_model, save_model

def main():
    print("Starting the script...")
    
    # Load and format data
    formatted_data = load_and_format_data('data.csv')
    
    # Setup tokenizer
    tokenizer = setup_tokenizer()
    
    # Prepare the dataset
    tokenized_dataset = prepare_dataset(formatted_data, tokenizer)
    
    # Load and modify the model with LoRA
    model = load_model(tokenizer)
    lora_model = apply_lora(model)
    
    # Fine-tune the model
    train_model(lora_model, tokenized_dataset)
    
    # Save the fine-tuned model and tokenizer
    save_model(lora_model, tokenizer)
    
    print("Script completed successfully!")

if __name__ == "__main__":
    main()
