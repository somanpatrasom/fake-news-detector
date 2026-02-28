import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only needed once)
nltk.download('stopwords')

# ── Load data ──────────────────────────────────────────
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['label'] = 0
true['label'] = 1

df = pd.concat([fake, true], ignore_index=True)

# ── Combine title + text into one column ───────────────
df['content'] = df['title'] + " " + df['text']

# ── Text cleaning function ─────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()                            # lowercase
    text = re.sub(r'\[.*?\]', '', text)            # remove text in brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)           # remove punctuation/numbers
    text = ' '.join([w for w in text.split() if w not in stop_words])  # remove stopwords
    return text

print("Cleaning text... (this may take 30-60 seconds)")
df['content'] = df['content'].apply(clean_text)

print("Done! Here's a cleaned example:")
print(df['content'][0][:300])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── Split data into features (X) and label (y) ────────
X = df['content']
y = df['label']

# ── Split into training and testing sets ──────────────
# 80% to train the model, 20% to test how well it learned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} articles, testing on {len(X_test)} articles")

# ── Convert text to numbers using TF-IDF ──────────────
vectorizer = TfidfVectorizer(max_features=10000)  # use top 10,000 words
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ── Train the model ────────────────────────────────────
print("\nTraining model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ── Test the model ─────────────────────────────────────
predictions = model.predict(X_test_tfidf)

print("\n=== RESULTS ===")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, predictions, target_names=['Fake', 'Real']))

# ── Test on custom input ───────────────────────────────
def predict(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]
    label = "REAL ✅" if result == 1 else "FAKE ❌"
    print(f"\nInput: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence — Fake: {confidence[0]*100:.1f}%  |  Real: {confidence[1]*100:.1f}%")

# Try these out
predict("Breaking: Scientists discover cure for cancer after 10 year study")
predict("SHOCKING: Obama secretly funded ISIS, documents reveal")
predict("Federal Reserve raises interest rates amid inflation concerns")

import joblib

# Save the model and vectorizer to disk
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\n✅ Model and vectorizer saved!")