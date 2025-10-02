"""
NgramAutoComplete.py
--------------------
Implements an N-gram language model for word suggestion.

Features:
- Preprocesses and tokenizes input text.
- Builds vocabulary with frequency filtering.
- Counts N-grams and calculates smoothed probabilities.
- Provides methods to:
  - Suggest next word (`suggest`, `suggest_fast`)
  - Get top-k word suggestions (`suggest_top_k`)
- Supports saving and loading models via pickle.

Used by:
- GUI.py (for live suggestions).
- TrainingNGrams.py (for training and saving).
"""

import nltk
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ssl
import certifi
import heapq

ssl._create_default_https_context = ssl._create_unverified_context


class NgramAutocomplete:
    def __init__(self, min_freq=6, max_n=5, unk_token="<unk>"):
        self.min_freq = min_freq
        self.max_n = max_n
        self.unk_token = unk_token
        self.vocab = []
        self.ngram_list = []

    def preprocess_text(self, text):
        from nltk.tokenize import TreebankWordTokenizer
        tokenizer = TreebankWordTokenizer()
        sentences = [s.strip().lower() for s in text.split('\n') if s.strip()]
        return [tokenizer.tokenize(s) for s in sentences]


    def build_vocab(self, tokenized_sentences):
        word_counts = {}
        for sent in tokenized_sentences:
            for word in sent:
                word_counts[word] = word_counts.get(word, 0) + 1

        sorted_vocab = sorted([(w, c) for w, c in word_counts.items() if c >= self.min_freq], key=lambda x: -x[1])
        self.vocab = [word for word, _ in sorted_vocab]

    def replace_oov(self, sentences):
        vocab_set = set(self.vocab)
        return [[word if word in vocab_set else self.unk_token for word in sent] for sent in sentences]

    def count_ngrams(self, sentences, n, start_token="<s>", end_token="<e>"):
        counts = {}
        for sent in sentences:
            tokens = [start_token]*n + sent + [end_token]
            for i in range(len(tokens) - (n if n > 1 else 0)):
                ngram = tuple(tokens[i:i+n])
                counts[ngram] = counts.get(ngram, 0) + 1
        return counts

    def train(self, file_path):
        with open(file_path, "r") as f:
            text = f.read()

        data = self.preprocess_text(text)
        train_data, _ = train_test_split(data, test_size=0.2, random_state=42)
        self.build_vocab(train_data)
        train_data = self.replace_oov(train_data)
        self.ngram_list = [self.count_ngrams(train_data, n) for n in range(1, self.max_n + 1)]

    def get_prob(self, word, prev_ngram, n_grams, nplus1_grams, vocab_size, k=1.0):
        prev_ngram = tuple(prev_ngram)
        prev_count = n_grams.get(prev_ngram, 0)
        nplus1_count = nplus1_grams.get(prev_ngram + (word,), 0)
        return (nplus1_count + k) / (prev_count + k * vocab_size)

    def get_probs(self, prev_ngram, n_grams, nplus1_grams, k=1.0, top_k_vocab=None):
        vocab_full = self.vocab + ["<e>", "<unk>"]
        if top_k_vocab:
            vocab_full = self.vocab[:top_k_vocab] + ["<e>", "<unk>"]
        vocab_size = len(vocab_full)

        return {
            word: self.get_prob(word, prev_ngram, n_grams, nplus1_grams, vocab_size, k)
            for word in vocab_full
        }



    def suggest(self, prev_tokens, k=1.0, start_with=None):
        suggestions = []
        for i in range(len(self.ngram_list) - 1):
            n = i + 1
            n_grams = self.ngram_list[i]
            nplus1_grams = self.ngram_list[i + 1]
            prev_ngram = prev_tokens[-n:] if len(prev_tokens) >= n else ["<s>"] * (n - len(prev_tokens)) + prev_tokens
            prob_dict = self.get_probs(prev_ngram, n_grams, nplus1_grams, k)

            best_word = None
            best_prob = 0
            for word, prob in prob_dict.items():
                if start_with and not word.startswith(start_with):
                    continue
                if prob > best_prob:
                    best_word = word
                    best_prob = prob

            suggestions.append((best_word, best_prob))
        return suggestions



    def suggest_fast(self, prev_tokens, k=1.0, top_k_vocab=1000, start_with=None):
        best_word = None
        best_prob = 0
        context_len = min(len(prev_tokens), self.max_n - 1)

        for n in reversed(range(1, context_len + 1)):
            n_grams = self.ngram_list[n - 1]
            nplus1_grams = self.ngram_list[n]

            prev_ngram = prev_tokens[-n:] if len(prev_tokens) >= n else ["<s>"] * (n - len(prev_tokens)) + prev_tokens
            prob_dict = self.get_probs(prev_ngram, n_grams, nplus1_grams, k, top_k_vocab)

            for word, prob in prob_dict.items():
                if start_with and not word.startswith(start_with):
                    continue
                if prob > best_prob:
                    best_word = word
                    best_prob = prob

            if best_word:
                break

        if not best_word:
            best_word = "<unk>"

        return best_word, best_prob


    def suggest_top_k(self, prev_tokens, k=1.0, top_k_vocab=1000, num_suggestions=3, start_with=None):
        context_len = min(len(prev_tokens), self.max_n - 1)
        suggestions = []

        for n in reversed(range(1, context_len + 1)):
            n_grams = self.ngram_list[n - 1]
            nplus1_grams = self.ngram_list[n]
            prev_ngram = prev_tokens[-n:] if len(prev_tokens) >= n else ["<s>"] * (n - len(prev_tokens)) + prev_tokens
            prob_dict = self.get_probs(prev_ngram, n_grams, nplus1_grams, k, top_k_vocab)

            filtered = [(word, prob) for word, prob in prob_dict.items() if (not start_with or word.startswith(start_with))]
            sorted_suggestions = sorted(filtered, key=lambda x: -x[1])[:num_suggestions]

            if sorted_suggestions:
                return sorted_suggestions

        return [("<unk>", 0.0)]  
    
    
    def save_model(self, ngram_path='en_counts.pkl', vocab_path='vocab.pkl'):
        with open(ngram_path, "wb") as f:
            pickle.dump(self.ngram_list, f)
        with open(vocab_path, "wb") as f:
            pickle.dump(self.vocab, f)

    def load_model(self, ngram_path='en_counts.pkl', vocab_path='vocab.pkl'):
        with open(ngram_path, "rb") as f:
            self.ngram_list = pickle.load(f)
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
