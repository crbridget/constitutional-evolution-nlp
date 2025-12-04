"""
great_textpectations.py:  An extensible framework for comparative text analysis
"""


from collections import Counter, defaultdict
import random as rnd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

import string
import textpectations_parsers as tp

#Needed libraries
#pip install nltk spacy gensim
#python -m spacy download en_core_web_sm
#pip install umap-learn
#pip install scikit-learn
#pip install plotly


#Data Handling 
import os
from collections import Counter
import re

#Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#pdf parsing 
from pypdf import PdfReader

#lda 
from gensim import corpora, models
#tf idf
from sklearn.feature_extraction.text import TfidfVectorizer
#dimension reduction???
import umap.umap_ as umap
#sankey 
import plotly.graph_objects as go





class Textpectations:

    @staticmethod
    def load_stop_words(stopfile):
        # A list of common or stop words. These get filtered from each file automatically
        english_stopwords = stopwords.words('english')

        if stopfile:
            with open(stopfile, 'r') as f:
                custom_stopwords = set(line.strip().lower() for line in f)
            # Combine both sets
            all_stopwords = english_stopwords.union(custom_stopwords)
        else:
            all_stopwords = english_stopwords

        return all_stopwords

    def __init__(self, stopfile=None):
        """ Constructor to initialize state """

        # Where all the data extracted from the loaded documents is stored
        self.data = defaultdict(dict)

        # Load stopwords once at initialization
        self.stopwords = Textpectations.load_stop_words(stopfile)

    @staticmethod
    def default_parser(filename, stopwords=None):
        """ For processing plain text files (.txt) """
        with open(filename, 'r') as file:
            results = file.read().lower()

        results = results.translate(str.maketrans('', '', string.punctuation))

        words = results.split()

        if stopwords:
            words = [word for word in words
                 if word not in stopwords
                 and len(word) > 2
                 and not any(char.isdigit() for char in word)
                 and not tp.is_roman_numeral(word)]

        results = {
            'wordcount': Counter(words),
            'numwords': len(words)
        }

        print("Parsed ", filename, ": ", results)
        return results

    def load_text(self, filename, label=None, parser=None):
        """ Register a text document with the framework.
         Extract and store data to be used later in our visualizations. """
        if parser is None:
            results = Textpectations.default_parser(filename, self.stopwords)
        else:
            results = parser(filename, self.stopwords)

        # Use filename for the label if none is provided
        if label is None:
            label = filename

        # Store the results for that ONE document into self.data
        # For example, document A:  numwords=10,  document B: numwords=20
        # For A, the results are: {numwords:10}, for B: {numwords:20}
        # This gets stored as: {numwords: {A:10, B:20}}


        for k, v in results.items():
            self.data[k][label] = v


    # REMOVE BEFORE SUBMIT
    def compare_num_words(self):
        """ A very simplistic visualization that creates
        a bar chart comparing num words for each text file
        For HW7, I expect much more interesting visualizations """
        numwords = self.data['numwords']
        for label, nw in numwords.items():
            plt.bar(label, nw)
        plt.show()

    def wordcount_sankey(self, word_list=None, k=5):
        # Map each text to words using a Sankey diagram, where the thickness of the line
        # is the number of times that word occurs in the text. Users can specify a particular
        # set of words, or the words can be the union of the k most common words across
        # each text file (excluding stop words)
        #putting them together
        all_wordcounts=self.data["wordcount"]
        print(all_wordcounts)






        pass

    def second_visualization(self):
        # A visualization array of subplots with one subplot for each text file.
        # Rendering subplots is a good, advanced skill to know!

        # topic modeling subplots - each subplot shows the topic distrubtion for that document (as a bar chart)
        # run LDA on entire corpus -> discover 5-8 topics automatically
        # topics are word clusters
        # each subplot shows % of document devoted to each topic
        
        pass

    def third_visualization(self):
        # A single visualization that overlays data from each of the text files. Make sure your
        # visualization distinguishes the data from each text file using labels or a legend

        # 2D Embedding Scatter Plot
        # TF-IDF vectors (convert text -> numbers) + UMAP (compress high-dimensional data into 2D while preserving similarity relationships)
        # close together = constituions with similar vocab/themes
        # far apart = very different vocab themes

        pass










