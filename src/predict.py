import os
import joblib
import numpy as np
from src.preprocess import preprocess_text

EMOTION_META = {
    0: {"name": "Sadness", "emoji": "😢", "color": "#3b82f6", "badge": "bg-blue-100 text-blue-800"},
    1: {"name": "Joy", "emoji": "😊", "color": "#eab308", "badge": "bg-yellow-100 text-yellow-800"},
    2: {"name": "Love", "emoji": "❤️", "color": "#ec4899", "badge": "bg-pink-100 text-pink-800"},
    3: {"name": "Anger", "emoji": "😠", "color": "#ef4444", "badge": "bg-red-100 text-red-800"},
    4: {"name": "Fear", "emoji": "😨", "color": "#8b5cf6", "badge": "bg-purple-100 text-purple-800"},
    5: {"name": "Surprise", "emoji": "😲", "color": "#06b6d4", "badge": "bg-cyan-100 text-cyan-800"}
}

class EmotionPredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models")
            
        model_path = os.path.join(models_dir, "emotion_model.pkl")
        vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Trained model or vectorizer not found in models/ directory. Run src/train.py first.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, raw_text: str):
        if not raw_text or not raw_text.strip():
            return {
                "error": "Empty text provided."
            }

        clean_text = preprocess_text(raw_text)
        if not clean_text:
            clean_text = raw_text.lower().strip()

        vectorized = self.vectorizer.transform([clean_text])
        
        # Get probability distribution if available
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vectorized)[0]
        else:
            # Fallback for decision function softmax approximation
            decision = self.model.decision_function(vectorized)[0]
            exp_d = np.exp(decision - np.max(decision))
            probs = exp_d / exp_d.sum()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        breakdown = []
        for idx in range(len(EMOTION_META)):
            meta = EMOTION_META[idx]
            breakdown.append({
                "id": idx,
                "emotion": meta["name"],
                "emoji": meta["emoji"],
                "color": meta["color"],
                "probability": round(float(probs[idx]), 4),
                "percentage": round(float(probs[idx]) * 100, 1)
            })

        # Sort breakdown by probability descending
        breakdown = sorted(breakdown, key=lambda x: x["probability"], reverse=True)

        top_meta = EMOTION_META[pred_class]

        return {
            "raw_text": raw_text,
            "cleaned_text": clean_text,
            "predicted_class_id": pred_class,
            "emotion": top_meta["name"],
            "emoji": top_meta["emoji"],
            "color": top_meta["color"],
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 1),
            "breakdown": breakdown
        }

if __name__ == "__main__":
    predictor = EmotionPredictor()
    sample = "I am so excited and happy to start this amazing journey!"
    res = predictor.predict(sample)
    print("Input:", res["raw_text"])
    print("Predicted Emotion:", res["emoji"], res["emotion"], f"({res['confidence_percentage']}%)")
