# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df = pd.read_csv("../data/raw/books.csv", on_bad_lines="skip")

# %%
df.index = df["bookID"]

# %%
df.shape[0]

# %%
df.head()
