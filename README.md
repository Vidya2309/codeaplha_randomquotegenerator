# Quote Explorer

A quote discovery app that classifies and matches quotes using a Naive Bayes
text classifier trained from scratch — not by trusting category tags handed
to it by a third-party API.

**Live demo:** _add your deployed link here_
**Original project this evolved from:** https://github.com/Vidya2309/codeaplha_randomquotegenerator

## Why this exists

The original version of this project called the Quotable API, filtered by
the categories the API already provided, and rendered the result — the same
"fetch → filter → render" shape as a lot of beginner API projects. This
version replaces that shape with something that does real work locally:

1. A **Naive Bayes classifier**, trained on a small hand-labeled quote
   dataset, assigns categories itself instead of relying on API tags.
2. A **client-side TF-IDF + cosine similarity search** ("Match my mood")
   converts free-text input into the same vector space the classifier
   learned, and ranks all quotes by similarity — a minimal vector-search
   engine running entirely in the browser, no server needed.
3. **Resilient API integration** — Discover still tries the live Quotable
   API first, but falls back to the locally classified dataset if the API
   is slow, down, or rate-limited (it has a history of reliability issues).

## How the classifier works

- `model/build_dataset.py` — 99 quotes, hand-labeled across 6 categories
  (motivation, philosophy, literature, wisdom, happiness, stress).
- `model/train_classifier.py`:
  - Merges `philosophy` into `wisdom` (the two overlapped too heavily in
    short-text vocabulary to be reliably distinguished — see below).
  - Trains a `TfidfVectorizer` + `MultinomialNB` pipeline (scikit-learn).
  - Evaluates with 5-fold stratified cross-validation.
  - Retrains on the full labeled set and uses it to auto-classify 25
    additional unlabeled quotes.
  - Exports the vocabulary, IDF weights, and every quote's TF-IDF vector to
    `data/quotes-dataset.json`, which the frontend loads directly — no
    inference server required at runtime.

### Honest results

```
5-fold cross-validation accuracy: 42.4% (+/- 5.0%)
Random baseline (5 classes):      20.0%
```

Full per-class report in [`model/model_report.txt`](model/model_report.txt).

The model performs roughly **2x better than chance**, but it's not evenly
good across categories — it's strong on `motivation` and `wisdom`, and weak
on `happiness`, `literature`, and `stress`. That's a genuine limitation, not
a rounding error, and it comes down to two things:

- **Not enough labeled data.** ~15–20 examples per category is very little
  for a bag-of-words model. Naive Bayes needs to see a word associated with
  a class multiple times before it learns the association.
- **Bag-of-words can't see meaning, only word overlap.** Quotes about
  "happiness" and "wisdom" often share almost no vocabulary in common even
  when they're thematically close, and TF-IDF has no way to know that
  "joy" and "happiness" are related concepts.

The natural next step — noted here rather than silently fixed — would be to
replace TF-IDF with sentence embeddings (e.g. a small `sentence-transformers`
model), which capture meaning rather than exact word overlap and would
likely close most of that gap.

## Features

- **Discover** — random quote, live API with local-model fallback, category
  filter driven by classifier output.
- **Match my mood** — type how you're feeling, get a quote ranked by cosine
  similarity in the classifier's own vector space, with the similarity
  score and matched category shown transparently.
- **Favorites** — saved locally (`localStorage`), persists across sessions.
- **Copy to clipboard** and **Twitter share**, carried over from the
  original project.
- Loading skeleton + explicit offline notice when the live API isn't
  reachable, instead of failing silently.

## Project structure

```
index.html / style.css / script.js   → the app (no build step needed)
data/quotes-dataset.json             → classifier output the app consumes
model/
  build_dataset.py                   → hand-labeled seed dataset
  train_classifier.py                → training, evaluation, export
  labeled_quotes.json                → the 99 labeled examples
  fallback-quotes.json               → 25 quotes the model auto-classifies
  quote_classifier.joblib            → the trained scikit-learn pipeline
  model_report.txt                   → full cross-validated evaluation
```

## Running locally

```bash
# regenerate the dataset / retrain the model (optional — data/quotes-dataset.json
# is already checked in)
cd model
pip install scikit-learn joblib
python build_dataset.py
python train_classifier.py

# run the app — any static server works, e.g.
cd ..
python -m http.server 8000
# open http://localhost:8000
```

## What I'd do with more time

- Swap TF-IDF for sentence embeddings to fix the weak categories.
- Grow the labeled set past ~100 examples — accuracy is data-starved right
  now, not algorithm-limited.
- Add a confusion matrix visualization to the README instead of just the
  text report.
