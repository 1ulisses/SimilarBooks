# SimilarBooks

A book similarity and recommendation project analyzing a dataset of ~11,000 books and implementing a Content-Based Nearest Neighbors (KNN) recommender system.

## Project Structure

```
SimilarBooks/
├── data/
│   ├── raw/           # Raw dataset (books.csv)
│   └── processed/     # Processed data
├── notebook/
│   ├── 01_data_exploration.py   # Exploratory data analysis (Jupytext format)
│   └── 02_knn_recommender.py    # KNN Recommendation model (Jupytext format)
├── outputs/           # Generated outputs
├── requirements.txt   # Project dependencies
└── AGENTS.md
```

## Data

The dataset (`data/raw/books.csv`) contains 11,127 books with 12 columns, including:

- `title`, `authors`, `publisher`
- `average_rating`, `ratings_count`, `text_reviews_count`
- `language_code`, `publication_date`, `num_pages`

**Data quirks** (handled in notebooks):

- The `num_pages` column header has a leading space (`"  num_pages"`).
- Some rows are malformed — load with `on_bad_lines="skip"` in pandas.

## Notebooks

Notebooks are managed with [Jupytext](https://jupytext.readthedocs.io/). The authoritative sources are the `.py` script files.

### 1. Data Exploration (`notebook/01_data_exploration.py`)

- Data loading and cleaning
- Most frequent titles, authors, and publishers
- Most rated and most reviewed books
- Language distribution
- Bayesian average rating (weighted by ratings count)
- Page counts and publication year trends
- Distribution of average ratings

### 2. KNN Recommender System (`notebook/02_knn_recommender.py`)

- Preprocessing & Cleaning: Extract publication year from dates and clean author names.
- Feature Engineering:
  - **Numerical Features**: `average_rating`, `ratings_count`, `text_reviews_count`, `num_pages`, and `publication_year` scaled using `StandardScaler`.
  - **Text Features**: `title` and `authors` converted into a TF-IDF matrix using `TfidfVectorizer(max_features=5000)`.
- Recommender Models (`scikit-learn` `NearestNeighbors` with cosine distance):
  - **Text Search (`search_book`)**: Finds books matching a text query using TF-IDF cosine similarity score.
  - **Book Recommendation (`recommend_book`)**: Finds similar books for a given title based on combined numerical metrics and text features.

## Getting Started

### Dependencies

Install required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

```bash
# Convert .py scripts to Jupytext notebooks if needed
jupytext --to ipynb notebook/01_data_exploration.py
jupytext --to ipynb notebook/02_knn_recommender.py

# Launch Jupyter Notebook
jupyter notebook
```
