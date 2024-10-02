# Klingon Translation Model

This project implements a fine-tuned GPT-2 model for translating English text to Klingon using LoRA (Low-Rank Adaptation). The model is trained on a custom dataset to generate natural-sounding Klingon translations based on English input. The project also includes evaluation scripts to assess the model's performance using BLEU scores.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)

## Features

- Fine-tuned GPT-2 model for English to Klingon translation
- Utilizes LoRA for efficient fine-tuning and reduced memory footprint
- Comprehensive data processing pipeline
- Inference script for easy translation of new English phrases
- Evaluation script for measuring translation quality using BLEU score
- Modular code structure for easy maintenance and extension

## Installation

To set up the project environment, ensure you have Python 3.7 or later. It is recommended to create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used for training consists of pairs of English and Klingon phrases stored in a CSV file named `data.csv`. The file should have the following structure:

| english             | klingon         |
|---------------------|-----------------|
| Hello, how are you? | nuqneH?         |
| ...                 | ...             |

Ensure your dataset is placed in the project root directory.

## File Structure

```plaintext
.
├── data_processing.py    # Handles data loading and formatting
├── tokenizer_setup.py    # Sets up the GPT-2 tokenizer
├── dataset_preparation.py # Prepares the dataset for training
├── train_model.py        # Contains functions for model training
├── main.py               # Orchestrates the entire training process
├── inference.py          # Performs inference with the fine-tuned model
├── evaluation.py         # Evaluates the model using BLEU scores
├── preprocess.py         # Alternative script combining data processing and training
├── data.csv              # Dataset for training and evaluation
├── requirements.txt      # List of required Python packages
└── README.md             # Project documentation
```

## Usage

### Training

To fine-tune the model, run:

```bash
python main.py
```

This script will load the dataset, tokenize it, apply LoRA, and train the GPT-2 model using the specified parameters. The trained model and tokenizer will be saved in the `./gpt2-klingon-finetuned` directory.

You can modify training parameters in `train_model.py` to adjust learning rate, batch size, number of epochs, etc.

### Inference

To translate English text to Klingon, use the `inference.py` script:

```python
from inference import translate_to_klingon

english_text = "Hello, how are you?"
klingon_translation = translate_to_klingon(english_text)
print(f"Klingon Translation: {klingon_translation}")
```

### Evaluation

To evaluate the model's performance on the dataset and compute the BLEU score, run:

```bash
python evaluation.py
```

This will generate translations for the evaluation set and print out the BLEU score.

## Model Architecture

- Base model: GPT-2
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Task: English to Klingon translation

Key LoRA parameters:
- Rank (r): 16
- Alpha: 32
- Dropout: 0.1
- Target modules: "c_attn", "c_proj"

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the batch size in `train_model.py`.
- For issues with the dataset, ensure your `data.csv` file is properly formatted and located in the project root.
- If you face problems with dependencies, make sure you're using a compatible Python version and have installed all requirements.
- Make sure you use the correct version of torch (the CUDA enabled one)

