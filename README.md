# 🔍 Fake News Detector

A machine learning web app that detects whether a news article is **real or fake** with ~99% accuracy.

Built from scratch as a learning project — no prior ML experience going in.

## 🚀 Live Demo
> Run locally with `streamlit run app.py`

## 🧠 How It Works
1. Text is cleaned — lowercased, punctuation removed, stopwords filtered
2. Converted to numerical vectors using **TF-IDF** (top 10,000 features)
3. Classified using **Logistic Regression**
4. Trained on 44,898 real and fake news articles

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 98.90% |
| Fake Precision | 99% |
| Real Precision | 99% |

## 🛠️ Tech Stack
- Python, scikit-learn, NLTK, pandas
- Streamlit (web UI)

## 📁 Dataset
[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon (not included in repo due to size — download from Kaggle)

## ▶️ Run Locally
```bash
pip install pandas scikit-learn nltk streamlit
python model.py      # trains and saves the model
streamlit run app.py # launches the web app
```

## ⚠️ Limitations
- Trained on 2016–2017 US political news — may not generalize to other topics
- Works best with full article text rather than short headlines