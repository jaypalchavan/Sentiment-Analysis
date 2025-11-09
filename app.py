from flask import Flask, render_template, request
import joblib # To load our scikit-learn model
from transformers import pipeline # For Hugging Face
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # For VADER

# --- NLTK Imports (for cleaning text for the sklearn model) ---
# We need the *exact same* cleaning function from the training script
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load All Models On Startup ---

# 1. Load VADER
print("Loading VADER model...")
vader_analyzer = SentimentIntensityAnalyzer()
print("VADER loaded.")

# 2. Load Hugging Face
print("Loading Hugging Face model (distilbert)...")
hf_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Hugging Face loaded.")

# 3. Load Scikit-Learn Pipeline
print("Loading Scikit-learn model pipeline...")
try:
    sklearn_pipeline = joblib.load('sentiment_model_pipeline.joblib')
    print("Scikit-learn model loaded.")
except FileNotFoundError:
    print("ERROR: 'sentiment_model_pipeline.joblib' not found.")
    print("Please run 'train_model.py' first to create the model file.")
    exit()

# --- Helper Cleaning Function (must match train_model.py) ---
# We don't need to re-download nltk assets, just use them
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Cleans text for the Scikit-learn model."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I|re.A)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)

# --- Define Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    original_text = ""
    selected_model = "hf" # Default model
    
    if request.method == 'POST':
        original_text = request.form['text']
        selected_model = request.form['model_choice']
        
        if not original_text.strip():
            result = {"label": "ERROR", "score": "Please enter some text."}
        else:
            if selected_model == 'hf':
                # Run Hugging Face
                res = hf_pipeline(original_text)[0]
                result = {"label": res['label'], "score": f"{res['score']*100:.2f}%"}
            
            elif selected_model == 'sklearn':
                # Run Scikit-learn
                # 1. Clean the text
                cleaned = clean_text(original_text)
                # 2. Predict
                prediction = sklearn_pipeline.predict([cleaned])[0]
                # 3. Get probabilities
                proba = sklearn_pipeline.predict_proba([cleaned])[0]
                score = max(proba)
                result = {"label": prediction.upper(), "score": f"{score*100:.2f}%"}

            elif selected_model == 'vader':
                # Run VADER
                res = vader_analyzer.polarity_scores(original_text)
                # VADER gives a 'compound' score from -1 to 1
                score = res['compound']
                if score >= 0.05:
                    label = "POSITIVE"
                elif score <= -0.05:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                # We'll show the compound score
                result = {"label": label, "score": f"Compound: {score*100:.4f}"}

    # Re-render the same page, now with the 'result' variable
    return render_template('index.html', 
                           result=result, 
                           original_text=original_text, 
                           selected_model=selected_model)

# --- Run the App ---

if __name__ == '__main__':
    print("Starting Flask server... Go to http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
