# 🧠 Sentiment Analysis Web App

A Flask-based **Sentiment Analysis** project that uses **three powerful NLP models** —  
**VADER**, **Hugging Face (DistilBERT)**, and **Scikit-learn (Logistic Regression)** —  
to classify text sentiment as **Positive**, **Negative**, or **Neutral**.

---

## 🚀 Features

- 🌐 Web-based interface built with **Flask**
- 🤖 Three model options:
  - **Hugging Face DistilBERT** (Transformer-based)
  - **Scikit-learn Logistic Regression** (trained on IMDB dataset)
  - **VADER Sentiment Analyzer** (rule-based)
- 🧹 Includes text cleaning and preprocessing using **NLTK**
- 📊 Displays prediction labels and confidence scores
- 💾 Model training and saving using `train_model.py`

---

## 📦 Requirements

Create a virtual environment and install dependencies:

pip install -r requirements.txt

##Project Structure
-📁 sentiment-analysis/
-│
-├── app.py                      # Flask web app
-├── train_model.py              # Script to train & save sklearn model
-├── IMDB Dataset.csv            # Dataset (download from Kaggle)
-├── sentiment_model_pipeline.joblib  # Saved sklearn model (generated)
-├── templates/
-│   └── index.html              # Frontend HTML page
-├── requirements.txt
-└── README.md



🧑‍💻 Author

Jaypal Dinesh Chavan
Cybersecurity & AI Enthusiast
📧 jaypalchavan1230@gmail.com
