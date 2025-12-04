
"""
File: textpectations_parsers.py
"""

import json
import string
import re
from collections import Counter
from pypdf import PdfReader

def is_roman_numeral(word):
    """Check if word is a Roman numeral"""
    return bool(re.match(r'^[ivxlcdm]+$', word))


def json_parser(filename, stopwords=None):
    f = open(filename)
    raw = json.load(f)
    text = raw['text']

    # Clean text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    # Filter
    if stopwords:
        words = [word for word in words
                 if word not in stopwords
                 and len(word) > 2
                 and not any(char.isdigit() for char in word)
                 and not is_roman_numeral(word)]

    f.close()
    return {'wordcount': Counter(words), 'numwords': len(words)}

def pdf_parser(filename, stopwords=None):
    # open/read PDF file
    reader = PdfReader(filename)

    # extract text from all pages
    text = ""  # Start with empty string
    for page in reader.pages:  # Loop through each page
        text += page.extract_text()  # Add each page's text

    # clean text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()


    # filter stopwords if provided
    if stopwords:
        words = [word for word in words
                 if word not in stopwords
                 and len(word) > 2
                 and not any(char.isdigit() for char in word)
                 and not is_roman_numeral(word)]

        # return in dict format
    return {'wordcount': Counter(words),
            'numwords': len(words)}
