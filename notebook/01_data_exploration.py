# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv("../data/raw/books.csv", on_bad_lines="skip")

# %%
df.index = df["bookID"]

# %%
df.shape[0]

# %%
df.head()

# %%
# Livro que mais ocorrem
sns.set_context("poster")
plt.figure(figsize=(20, 12))
books = df["title"].value_counts()[:20]
sns.barplot(x=books.values, y=books.index, palette="deep")
plt.title("Top 20 Livros Mais Ocorridos")
plt.show()

# %%
sns.set_context("poster")
plt.figure(figsize=(15, 10))
ax = df.groupby("language_code")["title"].count()
sns.barplot(x=ax.values, y=ax.index, palette="deep")
ax.set_title("Quantidade de Livros por Idioma")
ax.set_xlabel("Idioma")
ax.set_ylabel("Quantidade")
plt.show()

# %%
most_rated = (
    df.sort_values("ratings_count", ascending=False).head(10).set_index("title")
)
plt.figure(figsize=(15, 10))
sns.barplot(x=most_rated["ratings_count"], y=most_rated.index, palette="deep")
plt.title("Top 10 Livros Mais Avaliados")
plt.show()
