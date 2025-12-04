"""
great_textpectations.py:  An extensible framework for comparative text analysis
"""


from collections import Counter, defaultdict
import random as rnd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import sankey
import string
import textpectations_parsers as tp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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





    def wordcount_sankey(self, word_list=None, k=5):
        # Map each text to words using a Sankey diagram, where the thickness of the line is the number of times that word occurs in the text. Users can specify a particular
        # set of words, or the words can be the union of the k most common words in
        # each text file (excluding stop words)

        all_wordcounts = self.data["wordcount"]
        topk_per_doc = {}  # stores top k words for each document

        for doc_label, wc in all_wordcounts.items():
            # get the top k words for this document
            topk = wc.most_common(k)
            topk_per_doc[doc_label] = topk
        unique_words = set()

        for doc_label, topk in topk_per_doc.items():
            for word, count in topk:
                unique_words.add(word)
        rows = []

        for doc_label, topk in topk_per_doc.items():
            for word, count in topk:
                rows.append([doc_label, word, count])
        df = pd.DataFrame(rows, columns=["Document", "Word", "Count"])
        sankey.show_sankey(df, "Document", "Word", "Count")

    def topic_bar_plots(self, n_topics=6):
        """Topic modeling subplots using sklearn LDA"""

        # Step 1: Reconstruct texts
        texts = []
        labels = []

        for label, counter in self.data['wordcount'].items():
            text = ' '.join(counter.elements())
            texts.append(text)
            labels.append(label)

        # Step 2: Create document-term matrix
        vectorizer = CountVectorizer(max_features=1000)
        doc_term_matrix = vectorizer.fit_transform(texts)

        # Step 3: Run LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        doc_topics = lda.fit_transform(doc_term_matrix)

        # Step 4: Create subplots
        n_docs = len(labels)
        n_cols = 3
        n_rows = (n_docs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()

        for i, (label, topics) in enumerate(zip(labels, doc_topics)):
            ax = axes[i]
            ax.bar(range(n_topics), topics, color='steelblue')
            ax.set_title(label, fontsize=10, fontweight='bold')
            ax.set_xlabel('Topic')
            ax.set_ylabel('Proportion')
            ax.set_ylim([0, 1])
            ax.set_xticks(range(n_topics))

        # Hide empty subplots
        for i in range(n_docs, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.suptitle('Topic Distribution by Document', fontsize=14, y=1.02)
        plt.show()

        # Print topics
        print("\n=== Topic Descriptions ===")
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"\nTopic {topic_idx}: {', '.join(top_words)}")


    def similarity_scatterplot(self):
        # A single visualization that overlays data from each of the text files. Make sure your
        # visualization distinguishes the data from each text file using labels or a legend

        # 2D Embedding Scatter Plot
        # TF-IDF vectors (convert text -> numbers) + UMAP (compress high-dimensional data into 2D while preserving similarity relationships)
        # close together = constituions with similar vocab/themes
        # far apart = very different vocab themes

        # Reconstruct texts
        texts = []
        labels = []

        for label, counter in self.data['wordcount'].items():
            text = ' '.join(counter.elements())
            texts.append(text)
            labels.append(label)

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=500)
        tfidf_matrix = vectorizer.fit_transform(texts)

        # UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=3)
        coords = reducer.fit_transform(tfidf_matrix.toarray())

        # Plot - should use coords, not doc_topics!
        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)

        for i, label in enumerate(labels):  # ← Just label, not topics!
            plt.annotate(label,
                             (coords[i, 0], coords[i, 1]),
                             fontsize=9,
                             ha='center')

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Constitutional Document Similarity (TF-IDF + UMAP)')
        plt.tight_layout()
        plt.show()