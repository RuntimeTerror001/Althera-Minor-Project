import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load combined data
data = pd.read_csv("combined_data.csv")

# Replace missing values
data = data.fillna(0)

# Feature columns
feature_cols = [
    "HeartRate",
    "SpO2",
    "Motion",
    "simple_reaction_ms",
    "choice_reaction_ms",
    "finger_taps",
    "word_recall_score",
    "number_recall_score",
    "stroop_accuracy_percent"
]

features = data[feature_cols]

# Create a cognitive score formula:
#   - Lower reaction times are better → penalize high values
#   - Higher recall scores are better → reward high values
#   - Higher stroop accuracy is better → reward high values
#   - Higher finger taps indicate better motor control → reward
#   - Very high heart rate or very low SpO2 may indicate stress → slight penalty
data["CognitiveScore"] = (
    100
    - (data["simple_reaction_ms"] * 0.05)
    - (data["choice_reaction_ms"] * 0.02)
    + (data["finger_taps"] * 0.3)
    + (data["word_recall_score"] * 5)
    + (data["number_recall_score"] * 4)
    + (data["stroop_accuracy_percent"] * 0.2)
    - abs(data["HeartRate"] - 72) * 0.1   # penalize deviation from resting HR
    + (data["SpO2"] - 90) * 0.3           # reward higher SpO2
)

# Clamp scores to [0, 100]
data["CognitiveScore"] = data["CognitiveScore"].clip(0, 100)

target = data["CognitiveScore"]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=10,
)

model.fit(features, target)

# Save model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cognitive_model.pkl")
joblib.dump(model, model_path)

print(f"Model trained successfully on {len(data)} samples.")
print(f"Score range in training data: {target.min():.1f} — {target.max():.1f}")
print(f"Model saved to: {model_path}")