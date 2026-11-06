from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "sms_dataset.csv").dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
predictions = model.predict(X_test_vec)

print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
print("\\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

model_dir = ROOT / "model"
model_dir.mkdir(exist_ok=True)
with (model_dir / "spam_model.pkl").open("wb") as f:
    pickle.dump(model, f)
with (model_dir / "tfidf_vectorizer.pkl").open("wb") as f:
    pickle.dump(vectorizer, f)
print("\\nSaved trained model files in model/")
