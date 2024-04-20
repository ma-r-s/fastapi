import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Load the trained model from the joblib file
vectorizer = TfidfVectorizer()
stemmer = SnowballStemmer("spanish")
nltk.download("stopwords")
stopwords_es = set(stopwords.words("spanish"))


app = FastAPI()

# Load both the fitted vectorizer and the trained model from the same joblib file
loaded_vectorizer, loaded_model = joblib.load("vectorizer_model.pkl")


# Function to preprocess a review
def preprocess_review(review):
    # Convert to lowercase
    review = review.lower()

    # Remove special characters
    review = re.sub(r"\W", " ", review)

    # Remove stopwords and perform stemming
    review = [stemmer.stem(word) for word in review.split() if word not in stopwords_es]

    return " ".join(review)


# Function to rate a review
def rate_review(review):
    # Preprocess the review
    preprocessed_review = preprocess_review(review)

    # Transform the preprocessed review using the loaded vectorizer
    tfidf_review = loaded_vectorizer.transform([preprocessed_review])

    # Predict the score using the loaded model
    predicted_score = loaded_model.predict(tfidf_review)

    return predicted_score


@app.get("/")
async def root():
    return {"message": "Welcome to the review rating API!"}


@app.get("/predict/")
def predict_review(review: str):
    # Rate the review using the loaded vectorizer and model
    predicted_score = rate_review(review)
    return {"review": review, "predicted_score": int(predicted_score)}
