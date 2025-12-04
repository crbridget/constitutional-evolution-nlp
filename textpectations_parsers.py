
"""
An example of a custom domain-specific parser
"""

import json
from collections import Counter
from pypdf import PdfReader

def json_parser(filename):
    f = open(filename)
    raw = json.load(f)
    text = raw['text']
    words = text.split(" ")
    wc = Counter(words)
    num = len(words)
    f.close()
    return {'wordcount':wc, 'numwords':num}


from pypdf import PdfReader
from collections import Counter


def pdf_parser(filename):
    # open/read PDF file
    reader = PdfReader(filename)

    # extract text from all pages
    text = ""  # Start with empty string
    for page in reader.pages:  # Loop through each page
        text += page.extract_text()  # Add each page's text

    # preprocess text (split into words)
    words = text.split()

    # count words using counter
    wc = Counter(words)

    # calculate total number of words
    num = len(words)

    # return in dict format
    return {'wordcount': wc, 'numwords': num}


# Test it
result = pdf_parser("france_1791.pdf")
print(f"Total words: {result['numwords']}")
print(f"Top 10 words: {result['wordcount'].most_common(10)}")