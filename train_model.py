# train_model.py - FINAL FAST-TRAINING VERSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from sqlalchemy import create_engine
import os
from tqdm import tqdm

# --- SETUP AND FUNCTIONS ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)) # Added str() for safety
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stop_words]
    return ' '.join(text)

# --- MAIN SCRIPT ---

print("--- [START] Starting New FAST Model Training ---")

# 1. DATABASE SE DATA LOAD KARNA
print("\n[STEP 1/5] Loading a random sample of 25,000 articles from database...")
db_path = os.path.join('instance', 'users.db')
if not os.path.exists(db_path):
    print(f"!!! ERROR: Database file not found at '{db_path}' !!!")
    exit()

engine = create_engine(f'sqlite:///{db_path}')

# === YAHAN HAI ASLI MAGIC! ===
# Hum database se keh rahe hain ki humein koi bhi 25,000 random articles de do.
# Isse training time 20 min se 3-5 min ho jayega.
query = "SELECT content, user_feedback FROM news_article WHERE user_feedback IN ('REAL', 'FAKE') ORDER BY RANDOM() LIMIT 25000"

try:
    df = pd.read_sql_query(query, engine)
    if len(df) == 0:
        print("!!! ERROR: No data found in the database. !!!")
        exit()
    print(f"--> Success! Loaded {len(df)} random articles for fast training.")
except Exception as e:
    print(f"!!! ERROR: Could not load data from database. Error: {e} !!!")
    exit()

# 2. DATA PREPROCESSING
print("\n[STEP 2/5] Preprocessing text data...")
tqdm.pandas(desc="Preprocessing Articles")
df['processed_content'] = df['content'].progress_apply(preprocess_text)
print("--> Success! Text preprocessing complete.")

# 3. VECTORIZATION (FEATURES BANANA)
print("\n[STEP 3/5] Creating feature vectors with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3))
X = vectorizer.fit_transform(df['processed_content']).toarray()
y = df['user_feedback'].apply(lambda label: 1 if label == 'REAL' else 0)
print("--> Success! Vectorization complete.")

# 4. MODEL TRAINING
print("\n[STEP 4/5] Training the Logistic Regression model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("--> Success! Model training complete.")

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"--> New Model Accuracy (on 25k sample): {accuracy * 100:.2f}%")

# 5. PURANE MODEL KO HATAKAR NAYE MODEL KO SAVE KARNA
print("\n[STEP 5/5] Replacing old model files with the new ones...")
pickle.dump(model, open('model_new.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer_new.pkl', 'wb'))
if os.path.exists('model.pkl'): os.remove('model.pkl')
if os.path.exists('vectorizer.pkl'): os.remove('vectorizer.pkl')
os.rename('model_new.pkl', 'model.pkl')
os.rename('vectorizer_new.pkl', 'vectorizer.pkl')
print("--> Success! 'model.pkl' and 'vectorizer.pkl' have been updated.")

print("\n--- [COMPLETE] Your app is now ready with the new super-smart model! ---")