# NewsIntel AI – AI-Powered News Intelligence Platform

NewsIntel AI is a web-based platform that combines Machine Learning and Generative AI to analyse news articles, detect potentially fake news, and generate useful insights.

## Features

- Fake News Detection using Machine Learning
- Prediction confidence score
- AI-powered news summarization
- Key points extraction
- AI-based article explanation
- Ask AI questions about news articles
- Named entity extraction
- Live news feed using GNews API
- Prediction history

## Technologies Used

**Frontend**
- React.js
- Vite
- JavaScript
- CSS
- Recharts
- Lucide React

**Backend**
- Python
- FastAPI
- Uvicorn
- Pandas
- Scikit-learn
- Joblib

**Machine Learning**
- TF-IDF Vectorization
- Text Classification
- Trained Machine Learning Model

**Generative AI**
- Ollama
- Qwen3 Local Language Model

**Other Tools**
- SQLite
- Git and GitHub
- Visual Studio Code

## Project Structure

```text
NewsIntel-AI/
├── backend/
│   ├── main.py
│   └── evaluate_model.py
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── outputs/
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ajaycbsc24-debug/NewsIntel-AI.git
cd NewsIntel-AI
```

### 2. Set up the backend

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 3. Configure the GNews API

Create a `.env` file in the project root:

```env
GNEWS_API_KEY=your_gnews_api_key
```

Replace the placeholder with your own API key. Never upload your `.env` file or API key to GitHub.

### 4. Start the backend

```powershell
python -m uvicorn backend.main:app --reload
```

Backend: http://127.0.0.1:8000

API documentation: http://127.0.0.1:8000/docs

### 5. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL shown in the terminal, usually http://localhost:5173.

### 6. Start the local AI model

Install Ollama and ensure the configured Qwen3 model is downloaded and running locally.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend status |
| GET | `/news` | Retrieve news articles |
| POST | `/predict` | Classify news |
| POST | `/ai/summarize` | Summarize an article |
| POST | `/ai/key-points` | Extract key points |
| POST | `/ai/explain` | Explain article content |
| POST | `/ai/ask` | Ask questions about an article |
| POST | `/ai/entities` | Extract named entities |
| GET | `/history` | Retrieve prediction history |

## Disclaimer

NewsIntel AI is an educational project. Machine Learning predictions and confidence scores are not proof that an article is factually true or false. Always verify important information using reliable sources.

## Developer

**Ajay C**

B.Sc. (Hons.) Computer Science Student  
RV University

## GitHub Repository

https://github.com/ajaycbsc24-debug/NewsIntel-AI
