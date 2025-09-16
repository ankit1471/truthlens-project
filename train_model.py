# CELL 1: Importing Libraries
#%%
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("All libraries imported successfully!")


# CELL 2: Loading the data
#%%
# Load the fake and real news datasets
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# Let's see how the first few rows look
print("------ Fake News Head ------")
df_fake.head()
# %%
