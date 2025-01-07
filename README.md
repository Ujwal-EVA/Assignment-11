# Kannada Laungauge BPE Project

# Library used
  1. OS
  2. Json
  3. Unicode data
  4. Tokenizer

# Approach
  1. kn.txt - Kannada dataset for greneratign tokens
  2. bpe_tokenizer.py - Pyhton code for generating tokens using the Byte Pair Encoding technique and limiting it to 5000 tokens
  3. kannada_BPE.json - jason file comtaining tokens
  4. token_app.py - Python code for
       1. Taking a sentence as input
       2. Breaking it into tokens and assigning them token numbers from kannada_BPE.json file
       3. In case token does not exisit, it creates a new token and adds it to the kannada_BPE.json file
       4. Display tokenized sentence with colour coding, sentence token, total number of tokens and compression ratio
    
# Outcome
  Created a HuggingFace Space app for an interactive GUI
