import os
import json
import unicodedata
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def load_dataset(file_path):
    """Loads the dataset from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def preprocess_text(text):
    """
    Preprocesses the text by:
    - Normalizing Unicode to decompose characters into base letters and diacritics.
    - Keeping the structure ready for BPE with whitespace pre-tokenization.
    """
    # Normalize to NFD (Normalization Form Decomposed)
    normalized_text = unicodedata.normalize('NFD', text)
    # Replace newlines with spaces
    return normalized_text.replace('\n', ' ')

def calculate_compression_ratio(original_text, tokenized_output):
    """
    Calculates the compression ratio as:
    Compression Ratio = Original Size / Tokenized Size
    """
    original_size = len(original_text)
    tokenized_size = len(tokenized_output)
    if tokenized_size == 0:
        return float('inf')  # Avoid division by zero
    return original_size / tokenized_size

def create_bpe_tokenizer(file_path, vocab_size, output_file):
    """Creates a BPE tokenizer, trains it, and saves the tokens."""
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Define the trainer with the specified vocab size
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
    )

    # Use a pre-tokenizer to split input into whitespace-separated tokens
    tokenizer.pre_tokenizer = Whitespace()

    # Load and preprocess the data from the file
    print("Loading and preprocessing the dataset...")
    raw_text = load_dataset(file_path)
    processed_text = preprocess_text(raw_text)

    # Train the tokenizer
    print("Training the tokenizer...")
    tokenizer.train_from_iterator([processed_text], trainer)

    # Save the tokenizer vocabulary
    vocab = tokenizer.get_vocab()
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    # Print the total number of tokens generated
    total_tokens = len(vocab)
    print(f"Total number of tokens generated: {total_tokens}")

    if total_tokens != vocab_size:
        print(f"Note: The exact vocab size {vocab_size} was not achieved. Current tokens: {total_tokens}")

    print(f"Tokenizer vocabulary saved to {output_file}")

    # Tokenize the text and calculate the compression ratio
    print("Calculating compression ratio...")
    tokenized_output = tokenizer.encode(processed_text).tokens
    compression_ratio = calculate_compression_ratio(processed_text, tokenized_output)
    print(f"Compression Ratio (Original Size / Tokenized Size): {compression_ratio:.2f}")

if __name__ == "__main__":
    # File paths
    input_file = "kn.txt"
    output_file = "kannada_BPE.json"

    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The file {input_file} does not exist.")

    # Set the desired number of tokens
    vocab_size = 5000

    # Create and save the BPE tokenizer
    print("Creating and saving the BPE tokenizer...")
    create_bpe_tokenizer(input_file, vocab_size, output_file)
    print("Done!")
