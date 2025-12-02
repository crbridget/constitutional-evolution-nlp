"""
great_textpectations.py:  An extensible framework for comparative text analysis
"""


from collections import Counter, defaultdict
import random as rnd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


class Textpectations:

    def __init__(self):
        """ Constructor to initialize state """

        # Where all the data extracted from the loaded documents is stored
        self.data = defaultdict(dict)

    @staticmethod
    def load_stop_words(stopfile):
        # A list of common or stop words. These get filtered from each file automatically
        english_stopwords = stopwords.words('english')

    @staticmethod
    def default_parser(filename):
        """ For processing plain text files (.txt) """
        with open(filename, 'r') as file:
            results = file.read()

        results = {
            'wordcount': Counter(results.split(" ")),
            'numwords': len(results.split())
        }

        print("Parsed ", filename, ": ", results)
        return results

    def load_text(self, filename, label=None, parser=None):
        """ Register a text document with the framework.
         Extract and store data to be used later in our visualizations. """
        if parser is None:
            results = Textpectations.default_parser(filename)
        else:
            results = parser(filename)

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
        pass

    def second_visualization(self):
        # A visualization array of subplots with one subplot for each text file.
        # Rendering subplots is a good, advanced skill to know!
        pass

    def third_visualization(self):
        # A single visualization that overlays data from each of the text files. Make sure your
        # visualization distinguishes the data from each text file using labels or a legend
        pass










