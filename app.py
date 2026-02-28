import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Load saved model & vectorizer ─────────────────────
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ── Same cleaning function as before ──────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# ── Page config ───────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")

# ── UI ────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("Paste a news headline or article below and the model will predict whether it's **Real** or **Fake**.")
st.markdown("---")

user_input = st.text_area("📰 Enter news text here:", height=200, placeholder="Paste a headline or article...")

if st.button("Analyze", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        st.markdown("---")

        if result == 1:
            st.success("✅ This looks like REAL news")
        else:
            st.error("❌ This looks like FAKE news")

        col1, col2 = st.columns(2)
        col1.metric("Fake Confidence", f"{confidence[0]*100:.1f}%")
        col2.metric("Real Confidence", f"{confidence[1]*100:.1f}%")

        st.markdown("---")
        st.caption("⚠️ This model was trained on 2016–2017 US political news. Results on other topics may vary.")