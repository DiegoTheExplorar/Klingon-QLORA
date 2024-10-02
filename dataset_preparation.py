from datasets import Dataset
import pandas as pd

def prepare_dataset(formatted_data, tokenizer):
    print("Creating Hugging Face dataset...")
    dataset = Dataset.from_pandas(pd.DataFrame({'text': formatted_data}))
    
    # Tokenize the dataset
    print("Tokenizing the dataset...")
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print("Dataset tokenized.")
    
    return tokenized_dataset
