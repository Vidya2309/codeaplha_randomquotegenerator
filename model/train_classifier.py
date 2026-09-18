import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

# ---------- 1. Load labeled data, merge overlapping categories ----------
with open("labeled_quotes.json", encoding="utf-8") as f:
    raw = json.load(f)

MERGE = {"philosophy": "wisdom"}  # philosophy & wisdom overlap heavily in short quotes
for d in raw:
    d["category"] = MERGE.get(d["category"], d["category"])

texts = [d["quote"] for d in raw]
labels = [d["category"] for d in raw]

# ---------- 2. Cross-validate to report an honest accuracy figure ----------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
    ("nb", MultinomialNB(alpha=1.0)),
])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, texts, labels, cv=skf)

# Cross-validated predictions give a stable per-class report across the whole
# dataset instead of relying on one small, noisy train/test split.
cv_preds = cross_val_predict(pipeline, texts, labels, cv=skf)
report = classification_report(labels, cv_preds, zero_division=0)

n_classes = len(set(labels))
random_baseline = 1 / n_classes

print(f"5-fold CV accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
print(f"Random baseline ({n_classes} classes): {random_baseline:.2%}")
print(report)

# ---------- 3. Retrain on ALL labeled data for the production model ----------
pipeline.fit(texts, labels)
joblib.dump(pipeline, "quote_classifier.joblib")

# ---------- 4. Auto-classify additional unlabeled quotes with this model ----------
with open("fallback-quotes.json", encoding="utf-8") as f:
    extra = json.load(f)

extra_texts = [q["content"] for q in extra]
predicted = pipeline.predict(extra_texts)
probs = pipeline.predict_proba(extra_texts)
confidences = probs.max(axis=1)

classified_extra = []
for q, cat, conf in zip(extra, predicted, confidences):
    classified_extra.append({
        "content": q["content"],
        "author": q["author"],
        "category": cat,
        "confidence": round(float(conf), 3),
        "source": "model-classified"
    })

labeled_final = [
    {"content": d["quote"], "author": d["author"], "category": d["category"],
     "confidence": 1.0, "source": "hand-labeled"}
    for d in raw
]

full_dataset = labeled_final + classified_extra

# ---------- 5. Precompute TF-IDF vectors for mood-matching (same vector space) ----------
vectorizer = pipeline.named_steps["tfidf"]
all_texts = [q["content"] for q in full_dataset]
tfidf_matrix = vectorizer.transform(all_texts)

vectors = tfidf_matrix.toarray().tolist()
for q, vec in zip(full_dataset, vectors):
    nonzero = {i: round(v, 4) for i, v in enumerate(vec) if v > 0}
    q["vector"] = nonzero

vocab = vectorizer.get_feature_names_out().tolist()
idf = vectorizer.idf_.tolist()  # needed client-side to vectorize new mood text the same way

output = {"vocab": vocab, "idf": idf, "quotes": full_dataset}

with open("../data/quotes-dataset.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

with open("model_report.txt", "w") as f:
    f.write("Naive Bayes Quote Category Classifier -- Evaluation\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Labeled training examples: {len(raw)}\n")
    f.write(f"Categories ({n_classes}): {sorted(set(labels))}\n")
    f.write(f"Random baseline: {random_baseline:.2%}\n\n")
    f.write(f"5-fold cross-validation accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})\n\n")
    f.write("Per-class report (cross-validated predictions across full dataset):\n")
    f.write(report)
    f.write("\nNote: accuracy is well above the random baseline but limited by the small\n")
    f.write("hand-labeled dataset (~85 examples) and genuine semantic overlap between\n")
    f.write("categories like 'wisdom' and 'motivation' in short-text quotes. More labeled\n")
    f.write("data or a sentence-embedding-based classifier would likely improve this further.\n")

print(f"\nExported {len(full_dataset)} quotes total ({len(labeled_final)} hand-labeled, {len(classified_extra)} model-classified) to ../data/quotes-dataset.json")
