
from pathlib import Path
import re
import os
import json
import joblib
import sqlite3

from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# --------------------------------------------------
# 1. PROJECT PATHS AND ENVIRONMENT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
DATABASE_PATH = BASE_DIR / "fake_news_history.db"

load_dotenv(BASE_DIR / ".env")

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


# --------------------------------------------------
# 2. DATABASE SETUP
# --------------------------------------------------

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()

    print("SQLite database initialized successfully!")


initialize_database()


# --------------------------------------------------
# 3. DATABASE HELPER FUNCTIONS
# --------------------------------------------------

def save_prediction(title, content, prediction, confidence):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    created_at = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    cursor.execute("""
        INSERT INTO predictions
        (title, content, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (title, content, prediction, confidence, created_at))

    prediction_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return prediction_id, created_at


# --------------------------------------------------
# 4. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="NewsIntel API",
    description="News Intelligence Platform with Machine Learning and Live News.",
    version="1.3.0"
)


# --------------------------------------------------
# 5. ENABLE FRONTEND CONNECTION - CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 6. LOAD TRAINED MODEL AND VECTORIZER
# --------------------------------------------------

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Machine learning model loaded successfully!")
    print("TF-IDF vectorizer loaded successfully!")
    print("Model classes:", model.classes_)

except Exception as error:

    model = None
    vectorizer = None

    print("Error loading ML files:", error)


# --------------------------------------------------
# 7. REQUEST DATA FORMAT
# --------------------------------------------------

class NewsRequest(BaseModel):

    title: str = Field(
        default="",
        max_length=1000,
        description="News article title"
    )

    text: str = Field(
        ...,
        min_length=30,
        max_length=20000,
        description="News article text"
    )


# --------------------------------------------------
# 8. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 9. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "NewsIntel API is running",
        "status": "online",
        "version": "1.3.0"
    }


# --------------------------------------------------
# 10. HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "message": "NewsIntel API is running",
        "status": "online",
        "version": "1.3.0",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "news_api_configured": bool(GNEWS_API_KEY)
    }


# --------------------------------------------------
# 11. LIVE NEWS FEED ENDPOINT
# --------------------------------------------------

@app.get("/news")
def get_live_news(

    category: str = Query(
        default="general",
        pattern="^(general|world|nation|business|technology|entertainment|sports|science|health)$"
    ),

    query: str = Query(
        default="",
        max_length=100,
        description="Optional keyword search"
    ),

    max_results: int = Query(
        default=10,
        ge=1,
        le=10
    )
):

    if not GNEWS_API_KEY:

        raise HTTPException(
            status_code=503,
            detail="GNews API key is missing. Check your .env file."
        )

    if query.strip():

        endpoint = "https://gnews.io/api/v4/search"

        parameters = {
            "q": query.strip(),
            "lang": "en",
            "country": "in",
            "max": max_results,
            "apikey": GNEWS_API_KEY
        }

    else:

        endpoint = "https://gnews.io/api/v4/top-headlines"

        parameters = {
            "category": category,
            "lang": "en",
            "country": "in",
            "max": max_results,
            "apikey": GNEWS_API_KEY
        }

    request_url = endpoint + "?" + urlencode(parameters)

    request = Request(
        request_url,
        headers={
            "User-Agent": "NewsIntel/1.0"
        }
    )

    try:

        with urlopen(request, timeout=15) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

    except HTTPError as error:

        error_message = error.read().decode(
            "utf-8",
            errors="replace"
        )

        print("GNews HTTP error:", error.code, error_message)

        raise HTTPException(
            status_code=502,
            detail=(
                f"GNews returned HTTP {error.code}. "
                "Check your API key, request limits, and plan."
            )
        )

    except (URLError, TimeoutError) as error:

        print("GNews connection error:", error)

        raise HTTPException(
            status_code=502,
            detail="Unable to connect to GNews. Check your internet connection."
        )

    except (json.JSONDecodeError, UnicodeDecodeError):

        raise HTTPException(
            status_code=502,
            detail="GNews returned an invalid response."
        )

    articles = data.get("articles", [])

    formatted_articles = []

    for article in articles:

        source = article.get("source") or {}

        formatted_articles.append({
            "id": article.get("id"),
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "url": article.get("url"),
            "image": article.get("image"),
            "published_at": article.get("publishedAt"),
            "source": source.get("name", "Unknown source")
        })

    return {
        "status": "success",
        "provider": "GNews",
        "total": data.get(
            "totalArticles",
            len(formatted_articles)
        ),
        "returned_articles": len(formatted_articles),
        "category": category,
        "query": query,
        "articles": formatted_articles,
        "message": "News fetched successfully",
        "disclaimer": (
            "News availability and article content depend on "
            "the GNews API plan. Headlines are not independently verified."
        )
    }


# --------------------------------------------------
# 12. NEWS PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict_news(request: NewsRequest):

    if model is None or vectorizer is None:

        raise HTTPException(
            status_code=503,
            detail="Machine learning model is not available."
        )

    combined_content = request.title + " " + request.text

    cleaned_text = preprocess_text(combined_content)

    if not cleaned_text:

        raise HTTPException(
            status_code=400,
            detail="Please provide meaningful news content."
        )

    try:

        transformed_text = vectorizer.transform(
            [cleaned_text]
        )

        prediction = model.predict(
            transformed_text
        )[0]

        if prediction == 0:
            result = "FAKE"

        elif prediction == 1:
            result = "REAL"

        else:
            raise ValueError(
                f"Unexpected model prediction: {prediction}"
            )

        confidence = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                transformed_text
            )[0]

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )

        prediction_id, created_at = save_prediction(
            request.title,
            request.text,
            result,
            confidence
        )

        return {
            "id": prediction_id,
            "prediction": result,
            "confidence": confidence,
            "date": created_at,
            "message": (
                "The machine learning model classified "
                "this article as " + result
            ),
            "disclaimer": (
                "This confidence score represents the model's "
                "prediction confidence. It is not proof of whether "
                "the news is factually true."
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# --------------------------------------------------
# 13. GET PREDICTION HISTORY
# --------------------------------------------------

@app.get("/history")
def get_prediction_history():

    connection = None

    try:

        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute("""
            SELECT id, title, content, prediction,
                   confidence, created_at
            FROM predictions
            ORDER BY id DESC
        """)

        records = cursor.fetchall()

        history = [
            dict(record)
            for record in records
        ]

        return {
            "total": len(history),
            "history": history
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve history: {str(error)}"
        )

    finally:

        if connection:
            connection.close()


# --------------------------------------------------
# 14. CLEAR PREDICTION HISTORY
# --------------------------------------------------

@app.delete("/history")
def clear_prediction_history():

    connection = None

    try:

        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute("DELETE FROM predictions")

        connection.commit()

        return {
            "message": "Prediction history cleared successfully."
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to clear history: {str(error)}"
        )

    finally:

        if connection:
            connection.close()
