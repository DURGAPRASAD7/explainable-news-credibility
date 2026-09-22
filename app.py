from pathlib import Path
import json
import re
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

app = Flask(__name__)

VECTOR_FILE = MODELS / "tfidf_vectorizer.joblib"
MODEL_FILE = MODELS / "news_classifier.joblib"
META_FILE = MODELS / "model_metadata.json"

vectorizer = None
model = None
metadata = {}

def load_models():
    global vectorizer, model, metadata
    if VECTOR_FILE.exists() and MODEL_FILE.exists():
        vectorizer = joblib.load(VECTOR_FILE)
        model = joblib.load(MODEL_FILE)
        if META_FILE.exists():
            metadata = json.loads(META_FILE.read_text(encoding="utf-8"))

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", " URL ", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def linguistic_features(text):
    raw = str(text)
    words = re.findall(r"\b[\w'-]+\b", raw)
    sentences = re.split(r"[.!?]+", raw)
    sentences = [s for s in sentences if s.strip()]
    upper_words = [w for w in words if w.isupper() and len(w) > 2]
    exclamations = raw.count("!")
    questions = raw.count("?")
    avg_sentence_words = len(words) / max(len(sentences), 1)
    uppercase_ratio = len(upper_words) / max(len(words), 1)
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "average_sentence_length": round(avg_sentence_words, 2),
        "uppercase_word_ratio": round(uppercase_ratio * 100, 2),
        "exclamation_marks": exclamations,
        "question_marks": questions
    }

def explain_prediction(cleaned, x):
    # Local linear explanation: TF-IDF value * model coefficient.
    feature_names = vectorizer.get_feature_names_out()
    row = x.tocoo()
    contributions = []
    for idx, value in zip(row.col, row.data):
        contribution = float(value * model.coef_[0][idx])
        contributions.append((feature_names[idx], contribution))
    # Positive contribution -> Real (class 1); negative -> Fake (class 0)
    positive = sorted([p for p in contributions if p[1] > 0], key=lambda z: z[1], reverse=True)[:8]
    negative = sorted([p for p in contributions if p[1] < 0], key=lambda z: z[1])[:8]

    return {
        "supports_real": [{"term": t, "impact": round(v, 5)} for t, v in positive],
        "supports_fake": [{"term": t, "impact": round(abs(v), 5)} for t, v in negative],
        "method": "Local TF-IDF × Logistic Regression feature contribution"
    }

def credibility(prob_real):
    if prob_real >= 0.75:
        return "Credible", "High"
    if prob_real >= 0.50:
        return "More likely credible", "Moderate"
    if prob_real >= 0.25:
        return "Potentially misleading", "Moderate"
    return "Low credibility signal", "High"

load_models()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model": metadata.get("model", "Not trained")
    })

@app.route("/api/model-info")
def model_info():
    if not metadata:
        return jsonify({"trained": False})
    return jsonify({
        "trained": True,
        "dataset": metadata.get("dataset"),
        "rows": metadata.get("rows"),
        "accuracy": metadata.get("accuracy"),
        "model": metadata.get("model")
    })

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if model is None or vectorizer is None:
        @app.routereturn jsonify({
            "error": "Model not found. Run: python train_model.py"
        }), 503

    data = request.get_json(silent=True) or {}
    title = clean_text(data.get("title", ""))
    text = clean_text(data.get("text", ""))

    if not title and not text:
        return jsonify({"error": "Please enter a headline or article text."}), 400

    combined = clean_text(title + " " + text)
    if len(combined) < 30:
        return jsonify({"error": "Please provide at least 30 characters of news content."}), 400

    x = vectorizer.transform([combined])
    prob = model.predict_proba(x)[0]
    # model classes are [0,1] where 0=fake, 1=real
    class_prob = {int(c): float(p) for c, p in zip(model.classes_, prob)}
    prob_real = class_prob.get(1, 0.0)
    prob_fake = class_prob.get(0, 0.0)
    predicted = 1 if prob_real >= 0.5 else 0

    level, certainty = credibility(prob_real)
    confidence = max(prob_real, prob_fake)

    explanation = explain_prediction(combined, x)
    features = linguistic_features(combined)

    return jsonify({
        "prediction": "Real-pattern" if predicted == 1 else "Fake-pattern",
        "credibility": level,
        "certainty": certainty,
        "confidence": round(confidence * 100, 2),
        "real_probability": round(prob_real * 100, 2),
        "fake_probability": round(prob_fake * 100, 2),
        "linguistic_features": features,
        "explanation": explanation,
        "notice": "This is a text-pattern assessment learned from the training dataset. It is not independent factual verification."
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
