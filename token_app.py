import json
import os
import random

def load_bpe_model(file_path):
    """Load the BPE model from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_bpe_model(file_path, model):
    """Save the updated BPE model to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(model, f, ensure_ascii=False, indent=4)

def tokenize_text(text, bpe_model):
    """Tokenize Kannada text using the BPE model, adding new tokens as needed."""
    tokens = []
    for char in text:
        if char in bpe_model:
            tokens.append(bpe_model[char])
        else:
            # Create a new token for unseen characters
            new_token = f"tok_{len(bpe_model) + 1}"
            bpe_model[char] = new_token
            tokens.append(new_token)
    return tokens, bpe_model

def generate_random_color():
    """Generate a random color code for terminal output."""
    return f"\033[38;5;{random.randint(16, 231)}m"

def colorize_tokens(tokens):
    """Apply random colors to each token."""
    token_colors = {}
    colored_tokens = []
    for token in tokens:
        if token not in token_colors:
            token_colors[token] = generate_random_color()
        color = token_colors[token]
        colored_tokens.append(f"{color}{token}\033[0m")  # Reset color after token
    return " ".join(colored_tokens), token_colors

def colorize_sentence(text, token_colors, bpe_model):
    """Reprint the original sentence with token colors."""
    colored_sentence = []
    for char in text:
        token = bpe_model.get(char, "")
        color = token_colors.get(token, "\033[0m")
        colored_sentence.append(f"{color}{char}\033[0m")
    return "".join(colored_sentence)

def calculate_compression_ratio(original_text, tokenized_sentence):
    """Calculate the compression ratio of the tokenized text."""
    original_length = len(original_text)
    tokenized_length = len(tokenized_sentence)
    if tokenized_length == 0:
        return 0
    return original_length / tokenized_length

def main():
    bpe_file = "kannada_BPE.json"

    # Load existing BPE model or initialize an empty one
    bpe_model = load_bpe_model(bpe_file)

    # Prompt user for Kannada text input
    kannada_text = input("Enter a Kannada sentence: ")

    # Tokenize the text and update the BPE model
    tokenized_sentence, updated_bpe_model = tokenize_text(kannada_text, bpe_model)

    # Save the updated BPE model back to the file
    save_bpe_model(bpe_file, updated_bpe_model)

    # Display the tokenized sentence with color coding
    colored_tokens, token_colors = colorize_tokens(tokenized_sentence)
    print("\nTokenized Sentence with Color Coding:")
    print(colored_tokens)

    # Reprint the original sentence with token colors
    colored_sentence = colorize_sentence(kannada_text, token_colors, updated_bpe_model)
    print("\nOriginal Sentence with Token Colors:")
    print(colored_sentence)

    # Display the total number of tokens
    print("Total Number of Tokens:", len(tokenized_sentence))

    # Calculate and display the compression ratio
    compression_ratio = calculate_compression_ratio(kannada_text, tokenized_sentence)
    print(f"Compression Ratio: {compression_ratio:.2f}")

if __name__ == "__main__":
    main()
