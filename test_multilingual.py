
from part_b.multilingual.predict_svm import load_model, predict

# Load trained model once
model = load_model()

# Test English, Malay and Manglish
comments = [
    "Thank you for helping me!",
    "Kau memang bodoh lah.",
    "Bro kau ni so annoying lah.",
    "你是傻逼",
    "I totally disagree with you."
]

results = predict(comments, model)

for result in results:
    print("Text:", result["text"])
    print("Prediction:", result["label"])
    print("SVM Decision Score:", round(result["decision_score"], 4))
    print("-" * 50)
