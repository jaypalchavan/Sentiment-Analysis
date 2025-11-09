import pandas as pd
import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib # Used to save our model

print("Downloading NLTK assets...")
nltk.download('punkt')
nltk.download('punkt_tab')

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

nltk.download('wordnet')
print("NLTK assets downloaded.")

def clean_text(text):
    """Applies a series of cleaning steps to the input text."""
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A) # Remove punctuation/numbers
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

# --- Main script execution ---
if __name__ == "__main__":
    
    # 1. Load the dataset
    try:
        df = pd.read_csv("IMDB Dataset.csv")
    except FileNotFoundError:
        print("Error: 'IMDB Dataset.csv' not found.")
        print("Please download it from Kaggle and place it in this directory.")
        exit()
        
    print("Dataset loaded. Starting cleaning (this will take a few minutes)...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # Handle potential empty rows
    df.dropna(subset=['cleaned_review'], inplace=True)
    df = df[df['cleaned_review'].str.strip() != ""]

    print("Cleaning complete. Defining features and target...")
    X = df['cleaned_review']
    y = df['sentiment']

    # 2. Create a Scikit-learn Pipeline
    # A Pipeline chains steps together. Here, it will:
    # 1. Vectorize text with TfidfVectorizer
    # 2. Classify with LogisticRegression
    
    text_clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(solver='liblinear')),
    ])

    print("Training the Scikit-learn model...")
    text_clf_pipeline.fit(X, y)
    print("Model trained successfully.")

    # 3. Save the entire pipeline
    joblib.dump(text_clf_pipeline, 'sentiment_model_pipeline.joblib')
    
    # print("---------------------------------------------------------")
    print("SUCCESS: Model pipeline saved to 'sentiment_model_pipeline.joblib'")
    # print("You can now run the 'app.py' Flask server.")
    print("---------------------------------------------------------")
