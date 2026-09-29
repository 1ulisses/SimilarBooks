# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

for dirname, _, filenames in os.walk("../data/raw"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
df = pd.read_csv("../data/raw/books.csv", on_bad_lines="skip")
df.index = df["bookID"]

# %%
df["num_pages"] = df["  num_pages"]
df["publication_year"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.year
df["publication_year"] = df["publication_year"].fillna(df["publication_year"].median())
df["authors"] = df["authors"].str.replace("/", " ")

# %%
df = df.drop(
    columns=[
        "publication_date",
        "  num_pages",
        "isbn",
        "isbn13",
        "publisher",
        "language_code",
    ]
)

# %%
num_cols = [
    "average_rating",
    "ratings_count",
    "text_reviews_count",
    "num_pages",
    "publication_year",
]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# %%
text_features = df["title"] + " " + df["authors"]
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(text_features)

# %%
X = hstack([df[num_cols].values, tfidf_matrix])
knn = NearestNeighbors(n_neighbors=6, metric="cosine")
knn.fit(X)

# %%
knn_text = NearestNeighbors(n_neighbors=5, metric="cosine")
knn_text.fit(tfidf_matrix)


# %%
def search_query(query, n=5):
    query_vec = tfidf.transform([query])
    distances, indices = knn_text.kneighbors(query_vec, n_neighbors=n)

    results = df.iloc[indices[0]]
    results = results.copy()
    results["score"] = 1 - distances[0]
    return results[["title", "authors", "average_rating", "score"]]


# %%
print(search_query("crime and punishment"))
