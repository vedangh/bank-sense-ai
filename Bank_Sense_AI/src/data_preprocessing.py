"""
Data Preprocessing Module for Bank Sense Disambiguation
Handles text cleaning, tokenization, sentence validation, and vocabulary indexing.
"""

import re
import string
from typing import List, Tuple, Dict, Optional

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def clean_sentence(text: str) -> str:
    """Normalize whitespace and standard formatting without altering semantic tokens."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_corpus_sentence(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize a sentence into words for static Word2Vec training."""
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS or t == "bank"]
    return tokens

def validate_bank_sentence(text: str) -> Tuple[bool, str]:
    """Validate that the input string is suitable for bank sense prediction."""
    if not text or not text.strip():
        return False, "Input sentence cannot be empty. Please enter a sentence."
    
    cleaned = text.strip()
    if len(cleaned.split()) < 3:
        return False, "Sentence is too short. Please provide at least 3 words for contextual analysis."
    
    # Check for whole-word occurrence of 'bank' or 'banks'
    if not re.search(r"\bbanks?\b", cleaned, re.IGNORECASE):
        return False, "The input sentence must contain the target polysemous word 'bank' or 'banks'."
    
    return True, ""

def build_vocabulary(tokenized_sentences: List[List[str]], min_count: int = 2) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build word-to-index and index-to-word mappings from tokenized sentences."""
    freq: Dict[str, int] = {}
    for sent in tokenized_sentences:
        for word in sent:
            freq[word] = freq.get(word, 0) + 1
            
    word2idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for word, count in freq.items():
        if count >= min_count and word not in word2idx:
            word2idx[word] = idx
            idx += 1
            
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word
