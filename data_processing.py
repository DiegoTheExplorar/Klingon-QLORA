import pandas as pd

def load_and_format_data(file_path):
    print("Loading the dataset...")
    data = pd.read_csv(file_path)
    
    print("Formatting the data...")
    formatted_data = data.apply(lambda row: f"Translate English to Klingon: {row['english']} -> {row['klingon']}", axis=1)
    
    return formatted_data
