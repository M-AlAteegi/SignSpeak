"""
TrainingNGrams.py
-----------------
Trains the N-gram language model used for autocomplete suggestions.

Process:
- Loads English text dataset (e.g., en_US.twitter.txt from Kaggle).
- Tokenizes and builds vocabulary with frequency filtering.
- Constructs N-gram counts (up to max_n).
- Saves:
  - 'en_counts.pkl' (N-gram counts)
  - 'vocab.pkl' (vocabulary list)

Usage:
Run once to generate language model artifacts needed for GUI.py.
Dataset:
Download en_US.twitter.txt from Kaggle and place it in Sign_Language_Complete/.
"""

from utils.NgramAutoComplete import NgramAutocomplete
import nltk

# Download only the required datasets if they are not already available
nltk.download('punkt')
nltk.download('stopwords')

file_path = "Sign_Language_Complete/en_US.twitter.txt"

model = NgramAutocomplete()
model.train(file_path)

model.save_model(ngram_path='Sign_Language_Complete/models/en_counts.pkl', vocab_path='Sign_Language_Complete/models/vocab.pkl')

tokens = ["i", "want"]
suggestions = model.suggest(tokens)
print(suggestions)
