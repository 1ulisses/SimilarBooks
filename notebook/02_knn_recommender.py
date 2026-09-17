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
