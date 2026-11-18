import pandas as pd
import joblib
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Create Models folder
os.makedirs("Models", exist_ok=True)

# 1. Load cleaned dataset
df = pd.read_csv(
    "Dataset/final_cleaned_dataset.csv",
    encoding="utf-8"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Prepare text
df["combined_text"] = df["combined_text"].fillna("").astype(str)

df = df[
    df["combined_text"].str.strip() != ""
].copy()

X = df["combined_text"]

# 3. Generate sentiment labels using VADER
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):

    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"

    elif score <= -0.05:
        return "Negative"

    else:
        return "Neutral"


df["sentiment"] = X.apply(get_sentiment)

y = df["sentiment"]

print("\nSentiment distribution:")
print(y.value_counts())

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Define models
models = {

    "Linear_SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=3000
    ),

    "Logistic_Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Naive_Bayes": MultinomialNB(
        alpha=0.5
    ),

    "Random_Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

results = {}

# 6. Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    # Configure TF-IDF
    if name == "Random_Forest":

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

    else:

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

    # Fit vectorizer only on training data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model.fit(X_train_vec, y_train)

    # Make predictions
    predictions = model.predict(X_test_vec)

    # Evaluate performance
    results[name] = {

        "Accuracy": accuracy_score(
            y_test, predictions
        ) * 100,

        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ) * 100,

        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ) * 100,

        "F1 Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ) * 100,

        "Confusion Matrix": confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        ).tolist(),

        "Labels": model.classes_.tolist()
    }

    # Save model and vectorizer together
    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        f"Models/{name}_bundle.pkl"
    )

    print(f"{name} saved successfully!")

    print("Accuracy:", round(results[name]["Accuracy"], 2), "%")
    print("F1 Score:", round(results[name]["F1 Score"], 2), "%")


# 7. Save evaluation results
joblib.dump(
    results,
    "Models/model_results.pkl"
)

# 8. Display final results
print("\n========== MODEL PERFORMANCE ==========")

for name, metrics in results.items():

    print(f"\n{name}")
    print("Accuracy:", round(metrics["Accuracy"], 2), "%")
    print("Precision:", round(metrics["Precision"], 2), "%")
    print("Recall:", round(metrics["Recall"], 2), "%")
    print("F1 Score:", round(metrics["F1 Score"], 2), "%")

print("\nAll four models trained and saved successfully!")