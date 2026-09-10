# AGENTS.md

## Project Overview

Book similarity/recommendation project. Early stage — raw data and exploration notebooks only.

## Data

- `data/raw/books.csv` — 11,127 books, 12 columns including `title`, `authors`, `average_rating`, `ratings_count`, `language_code`
- Note: `num_pages` column has a leading space in the CSV header (`"  num_pages"`)
- `data/processed/` and `outputs/` exist but are empty

## Notebooks

- `notebook/01_data_exploration.py` — Jupytext-format script, **authoritative source** (the `.ipynb` is stale)
- `notebook/01_data_exploration.ipynb` — references non-existent CSV files (`Books.csv`, `Ratings.csv`, `Users.csv`); out of sync with `.py`
- Use `on_bad_lines="skip"` when reading `books.csv` (malformed rows exist)

## Dependencies (not managed)

No `requirements.txt` or `pyproject.toml`. Core stack: `pandas`, `numpy`, `matplotlib`, `seaborn`.
