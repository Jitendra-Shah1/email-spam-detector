# 📧 Email Spam Detector

An ML project that classifies email messages as **Spam** or **Not Spam**, served via **FastAPI** with a **Streamlit** frontend.

## Overview

- Dataset: `emails.csv` (`text`, `spam` columns)
- EDA on message length, word count, sentence count + word clouds for spam vs. not-spam
- Text cleaning: lowercase → expand contractions → remove punctuation → tokenize → remove stopwords → stem (shared in `preprocessing_utils.py`)
- Vectorized with `CountVectorizer`, trained 6 models: Naive Bayes, Logistic Regression, Decision Tree, Random Forest, SVC, Gradient Boosting
- Compared on Accuracy, Precision, Recall, F1 → shortlisted **Logistic Regression** and **Random Forest**
- Tuned both with **Optuna** (5-fold CV, F1 objective)
- Final model: **Random Forest** (`max_depth=200`, `max_samples=0.75`)

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 99% |
| Precision | 98% |
| Recall | 97% |
| F1 | 97% |
| ROC-AUC | 99.68% |

## Pipeline

Preprocessing, vectorizing, and the model are combined into one `sklearn` `Pipeline` and trained on **raw** text (not pre-cleaned text), so training and serving use identical logic. Saved as `spam_pipeline.pkl`.

```python
Pipeline([
    ('preprocessing', FunctionTransformer(preprocess_series)),
    ('vectorizer', CountVectorizer(max_features=5000, min_df=5)),
    ('model', RandomForestClassifier(max_depth=200, max_samples=0.75))
])
```

⚠️ `preprocessing_utils.py` must be importable by the FastAPI backend — the pickled pipeline references it by module name.

## Project Structure
```

email-spam-detector/
│
├── dataset/
│   └── emails.csv
│
├── backend/
│   ├── model/
│   │   ├── predict.py
│   │   ├── spam_pipeline.pkl
│   │   └── text_preprocessing.py
│   │
│   ├── schema/
│   │   └── userinput.py
│   │
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── notebook/
│   └── spam_classifier.ipynb
│
└── README.md

```

## Running

```bash
# Terminal 1 - backend
cd model
pip install -r requirements.txt
uvicorn main:app --reload

# Terminal 2 - frontend
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Example

**Request**
```json
POST /predict
{ "message": "Congratulations! You've WON a $1000 prize, click here NOW!!" }
```

**Response**
```json
{ "response": "spam" }
```

## Future Improvements

- Return prediction probabilities via `predict_proba`
- Dockerize the app for consistent, portable deployment
- Deploy backend and frontend to the cloud (e.g. Render, Railway, AWS)
- Unit tests, experiment tracking