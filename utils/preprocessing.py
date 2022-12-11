import os
import re

def get_stopword_list(path):
    with open(path, mode='r', encoding='utf-8') as f:
        stopwords = f.readlines()

    stopwords = [stopword.strip() for stopword in stopwords]
    return stopwords

def remove_punctuation(row):
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Remove all . , " ... in sentences
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")

    row = row.strip()
    return row
