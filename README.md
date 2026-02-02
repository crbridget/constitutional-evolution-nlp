# Constitutional Evolution: A Text Analysis Framework

A Python-based text analysis tool that visualizes linguistic patterns and thematic similarities across historical constitutional documents from 17 countries spanning 1787-1997.

## Overview

This project applies NLP techniques to compare constitutional texts, revealing how political philosophies, governance structures, and rights frameworks evolved across different nations and time periods. Through custom parsers and visualization methods, it transforms dense legal documents into interpretable insights about constitutional design patterns.

## Features

- **Multi-format text processing**: Custom parsers for PDF and JSON formats with configurable stopword filtering
- **Sankey flow diagrams**: Visualize most frequent terms by document to identify dominant themes
- **Topic modeling**: LDA-based clustering to discover latent thematic patterns across constitutions
- **Document similarity mapping**: TF-IDF + UMAP dimensionality reduction to plot constitutional texts in 2D semantic space

## Technical Implementation

**Text Processing Pipeline:**
- Extracts and normalizes text from PDF/JSON sources
- Filters stopwords, punctuation, numbers, and Roman numerals
- Generates word frequency distributions and document statistics

**Analysis Methods:**
- **TF-IDF vectorization** for term importance weighting
- **Latent Dirichlet Allocation** for topic extraction
- **UMAP** for high-dimensional similarity visualization
- **Sankey diagrams** for term flow analysis

## Dataset

Analyzes 17 constitutional documents:
- USA (1787), France (1791), Mexico (1917), Russia (1918, 1993)
- Germany (1919, 1949), Japan (1947), India (1950)
- North Korea (1972), Spain (1978), Iran (1979), China (1982)
- South Korea (1987), Brazil (1988), South Africa (1996), Poland (1997)

## Key Files

- `great_textpectations.py`: Core analysis framework with visualization methods
- `textpectations_parsers.py`: Custom parsers for PDF and JSON text extraction
- `main.py`: Driver script that loads documents and generates all visualizations

## Output Visualizations

1. **Sankey Diagram** (`sankey_diagram.html`): Interactive flow chart showing top-k words per document
2. **Topic Distribution** (`topic_distribution.png`): Bar plots showing LDA topic proportions across documents
3. **Similarity Scatterplot** (`similarity_scatterplot.png`): 2D projection of document similarity in semantic space

## Technologies

**Languages:** Python  
**Libraries:** NLTK, scikit-learn, UMAP, Matplotlib, Pandas, pypdf  
**Techniques:** TF-IDF, LDA topic modeling, dimensionality reduction, text preprocessing

## Running the Analysis
```python
from great_textpectations import Textpectations
import textpectations_parsers as tp

# Initialize framework
tt = Textpectations()

# Load documents with custom parser
tt.load_text('pdfs/usa_1787.pdf', 'USA (1797)', parser=tp.pdf_parser)

# Generate visualizations
tt.similarity_scatterplot()
tt.wordcount_sankey()
tt.topic_bar_plots()
```

## Insights

The framework reveals:
- **Linguistic clustering** by political system (e.g., socialist vs. democratic constitutions)
- **Temporal evolution** in constitutional language and priorities
- **Thematic patterns** around rights, governance structures, and state powers
- **Cross-cultural influences** through shared terminology and concepts

---

*Built as part of data science coursework exploring NLP applications in political science and comparative government.*
