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

for dirname, _, filenames in os.walk("../data/raw"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
df = pd.read_csv("../data/raw/books.csv", on_bad_lines="skip")
df.index = df["bookID"]

# %%
df.shape[0]

# %%
df.head()

# %%
df.isnull().sum()

# %%
top_titles = df["title"].value_counts()[:20]
sns.set_context("poster")
plt.figure(figsize=(20, 12))
sns.barplot(
    x=top_titles.values, y=top_titles.index, palette="deep", hue=top_titles.index
)
plt.title("Top 20 livros mais ocorridos")
plt.xlabel("Quantidade de ocorrências")
plt.ylabel("Nomes")
plt.tight_layout()
plt.show()

# %%
top_rated = df.nlargest(20, "ratings_count")

sns.set_context("poster")
plt.figure(figsize=(20, 12))
sns.barplot(
    x=top_rated["ratings_count"].values,
    y=top_rated["title"].values,
    palette="deep",
    hue=top_rated["title"].values,
)
plt.title("Top 20 most rated books")
plt.xlabel("Ratings count")
plt.ylabel("Title")
plt.tight_layout()
plt.show()

# %%
top_lang = df.groupby("language_code").size().sort_values(ascending=False)[:20]
sns.set_context("poster")
plt.figure(figsize=(20, 12))
sns.barplot(
    x=top_lang.values,
    y=top_lang.index,
    palette="deep",
    hue=top_lang.index,
    legend=False,
)
plt.title("Top 20 Languages by Book Count")
plt.xlabel("Count")
plt.ylabel("Lang")
plt.tight_layout()
plt.show()

# %%
C = (df["average_rating"] * df["ratings_count"]).sum() / df["ratings_count"].sum()
m = df["ratings_count"].quantile(0.50)
df["bayesian_avg"] = (df["ratings_count"] / (df["ratings_count"] + m)) * df[
    "average_rating"
] + (m / (df["ratings_count"] + m)) * C

top_books = df.sort_values("bayesian_avg", ascending=False).head(20)

# %%
sns.set_context("poster")
plt.figure(figsize=(20, 12))
sns.barplot(
    y=top_books["title"],
    x=top_books["bayesian_avg"],
    palette="deep",
    hue=top_books["title"].values,
)
plt.title("Top 20 Highest average rating")
plt.xlabel("Average rating")
plt.ylabel("Book")
plt.tight_layout()
plt.show()
