"""
great_textpectations.py
framework for comparing and visualizing text documents.
"""

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import sankey
import string
import textpectations_parsers as tp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import umap.umap_ as umap


class Textpectations:
    """
    Load, analyze, and visualize multiple text documents.
    """

    @staticmethod
    def load_stop_words(stopfile):
        """
        Load default stopwords
        """
        sw = set(stopwords.words('english'))
        if stopfile:
            with open(stopfile, 'r') as f:
                sw.update(line.strip().lower() for line in f)
        return sw

    def __init__(self, stopfile=None):
        """
        Initialize storage and stopword list
        """
        self.data = defaultdict(dict)
        self.stopwords = self.load_stop_words(stopfile)

    @staticmethod
    def default_parser(filename, stopwords=None):
        """
        parser for .txt files: lowercase, remove punctuation, split, filter, count.
        """
        with open(filename, 'r') as file:
            text = file.read().lower()

        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()

        if stopwords:
            words = [
                w for w in words
                if w not in stopwords
                and len(w) > 2
                and not any(ch.isdigit() for ch in w)
                and not tp.is_roman_numeral(w)
            ]

        return {
            "wordcount": Counter(words),
            "numwords": len(words)
        }

    def load_text(self, filename, label=None, parser=None):
        """
        Load a file, parse it, and store wordcount + metadata.
        """
        if parser is None:
            results = self.default_parser(filename, self.stopwords)
        else:
            results = parser(filename, self.stopwords)

        if label is None:
            label = filename

        for key, value in results.items():
            self.data[key][label] = value


    # SANKEY DIAGRAM

    def wordcount_sankey(self, k=5):
        """
        Show a Sankey diagram linking each document to its top-k words.
        """
        rows = []
        wc_dict = self.data["wordcount"]

        for label, wc in wc_dict.items():
            for word, count in wc.most_common(k):
                rows.append([label, word, count])

        df = pd.DataFrame(rows, columns=["Document", "Word", "Count"])

        fig = sankey.make_sankey(df, "Document", "Word", "Count")
        fig.write_html("sankey_diagram.html")
        sankey.show_sankey(df, "Document", "Word", "Count")


    # TOPIC MODELING, 2nd visualization

    def topic_bar_plots(self, n_topics=6):
        """
        Create bar plots showing topic proportions for each document (LDA).
        """
        texts = [' '.join(c.elements()) for c in self.data["wordcount"].values()]
        labels = list(self.data["wordcount"].keys())

        vec = CountVectorizer(max_features=1000)
        dtm = vec.fit_transform(texts)

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        doc_topics = lda.fit_transform(dtm)

        # Setup subplot grid
        n_docs = len(labels)
        rows = (n_docs + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = axes.flatten()

        for i, (label, topics) in enumerate(zip(labels, doc_topics)):
            ax = axes[i]
            ax.bar(range(n_topics), topics)
            ax.set_title(label)
            ax.set_ylim(0, 1)

        for i in range(n_docs, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        plt.savefig("topic_distribution.png", dpi=300)
        plt.show()

    # TF-IDF + UMAP SIMILARITY PLOT

    def similarity_scatterplot(self):
        """
        Visualize document similarity in a plot using TF-IDF + UMAP.
        """
        texts = [' '.join(c.elements()) for c in self.data["wordcount"].values()]
        labels = list(self.data["wordcount"].keys())

        vec = TfidfVectorizer(max_features=500)
        tfidf = vec.fit_transform(texts)

        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(tfidf.toarray())

        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.7)

        for i, label in enumerate(labels):  
            plt.annotate(label,
                         (coords[i, 0], coords[i, 1]),
                         fontsize=9,
                         ha='center')

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Constitutional Document Similarity (TF-IDF + UMAP)')
        plt.tight_layout()
        plt.show()
