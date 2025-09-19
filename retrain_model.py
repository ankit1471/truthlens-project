# retrain_model.py

import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier # <-- Yahan apne original model ka naam daalo
from sqlalchemy import create_engine
import nltk

# --- NLTK Stopwords Download (Ek Baar ke liye) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- Preprocessing Function (Aapke app.py se copy kiya hua) ---
ps = PorterStemmer()
def preprocess_text(text):
    # Agar text None ya float hai (khali data), toh use khali string bana do
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

print("Retraining script started...")

# --- 1. Purana Data Load Karna ---
try:
    df_true = pd.read_csv('True.csv')
    df_fake = pd.read_csv('Fake.csv')
    
    # Label add karo (1 for REAL, 0 for FAKE)
    df_true['label'] = 1
    df_fake['label'] = 0
    
    # Sirf 'text' aur 'label' column rakho
    df_true = df_true[['text', 'label']]
    df_fake = df_fake[['text', 'label']]
    
    # Original data ko jodo
    df_original = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"Loaded {len(df_original)} articles from original CSV files.")

except FileNotFoundError:
    print("Error: True.csv or Fake.csv not found. Cannot proceed.")
    exit()

# --- 2. Naya Data Database se Load Karna ---
db_path = 'instance/users.db'
engine = create_engine(f'sqlite:///{db_path}')

try:
    # Sirf woh data nikalo jahan user ne feedback diya hai
    query = "SELECT content AS text, user_feedback FROM news_article WHERE user_feedback IS NOT NULL"
    df_feedback = pd.read_sql_query(query, engine)
    
    if not df_feedback.empty:
        # User feedback (REAL/FAKE) ko label (1/0) mein badlo
        df_feedback['label'] = df_feedback['user_feedback'].apply(lambda x: 1 if x == 'REAL' else 0)
        df_feedback = df_feedback[['text', 'label']]
        print(f"Loaded {len(df_feedback)} new articles with feedback from the database.")
    else:
        print("No new feedback data found in the database. Retraining on original data only.")
        df_feedback = pd.DataFrame(columns=['text', 'label']) # Khali DataFrame

except Exception as e:
    print(f"Could not load data from database: {e}")
    df_feedback = pd.DataFrame(columns=['text', 'label']) # Khali DataFrame

# --- 3. Data ko Jodna ---
df_combined = pd.concat([df_original, df_feedback], ignore_index=True)
# Duplicates hatao, agar user ne wahi article submit kiya jo pehle se dataset mein tha
df_combined.drop_duplicates(subset=['text'], inplace=True)
print(f"Total unique articles for retraining: {len(df_combined)}")

# --- 4. Model ko Shuru se Train Karna ---
print("Preparing data for training...")
# Khali ya galat data ko aache se handle karo
df_combined['text'] = df_combined['text'].fillna('').astype(str)
df_combined.dropna(subset=['text', 'label'], inplace=True)

X = df_combined['text'].apply(preprocess_text)
y = df_combined['label']

print("Training new vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_vect = vectorizer.fit_transform(X)

print("Training new model...")
# Yahan wahi model use karo jo aapne pehli baar use kiya tha
model = PassiveAggressiveClassifier(max_iter=100) # Example model
model.fit(X_vect, y)
print("Model training complete.")

# --- 5. Naya Model aur Vectorizer Save Karna ---
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("SUCCESS: New model.pkl and vectorizer.pkl have been saved!")

# ```**Important Note:** Line 69 par `PassiveAggressiveClassifier` likha hai. Agar aapne `train_model.py` mein koi aur model (jaise `LogisticRegression`) use kiya tha, toh yahan bhi wahi naam likhna.

### **Step 3: Script ko Kaise Chalayein?**

# 1.  Apni website par jaakar kuch news articles par feedback do taaki database mein naya data aa jaye.
# 2.  Apne terminal mein (jahan `(venv)` active hai), `app.py` server ko `Ctrl + C` se band karo.
# 3.  Ab bas yeh command chalao:
#     ```bash
# 4.  Yeh script chalegi aur aapko terminal mein progress dikhayegi. Jab "SUCCESS" ka message aa jaye, iska matlab hai ki aapki `model.pkl` file update ho chuki hai.
# 5.  Ab apna web server dobara chalu karo: `python app.py`.

# Aapka app ab naye, zyada samajhdar model ko use karega!

### **Important Baatein (Zaroor Padhein)**

# *   **Kab Retrain Karein?:** Yeh script har user ke feedback ke baad nahi chalani hai. Ise hafte mein ek baar, ya jab aapke paas 50-100 naye feedback aa jayein, tab chalana chahiye.
# *   **Data Quality:** Agar users galat feedback denge, toh aapka model galat cheezein seekh lega. Isliye yeh feature un projects ke liye best hai jahan aapko users par bharosa ho.
# *   **Backup:** Pehli baar yeh script chalane se pehle, apni original `model.pkl` aur `vectorizer.pkl` files ka backup (copy karke kahin aur save kar lo) zaroor bana lena.