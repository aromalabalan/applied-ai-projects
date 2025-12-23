# Resume Matching & Classification

Brief pipeline to classify resumes and match job descriptions using TF-IDF and a Random Forest classifier.

**Overview**
- This project trains a `RandomForestClassifier` on TF-IDF features extracted from resumes to predict job categories.
- It also provides a simple job-to-resume matching utility using cosine similarity over TF-IDF vectors.
- The main workflow is implemented in the notebook `Resume_matching.ipynb`.

**Key Files**
- `Resume_matching.ipynb` — main notebook with data download, preprocessing, training, evaluation, and matching functions.
- `requirements.txt` — Python dependencies (use `pip install -r requirements.txt`).
- Saved artifacts (produced by the notebook): `resume_classifier_model.pkl`, `resume_tfidf_vectorizer.pkl`, `resume_label_encoder.pkl`.

**Dependencies**
Install dependencies:

```
pip install -r requirements.txt
```

Typical packages used include: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `wordcloud`, and `kagglehub`.

**Quick Start**
1. Open the notebook `Resume_matching.ipynb` and run the cells in order. The notebook downloads the dataset using `kagglehub`, extracts the CSV, preprocesses text, trains the model, evaluates results, and saves artifacts.
2. After running, you'll have saved model files in the notebook working directory.

**Usage Examples (within the notebook)**
Run the example cells or call these functions after executing the notebook cells that define them:

```python
# Match a job description to top resumes/categories
x = match_job_with_resumes("Python ML engineer with NLP experience", top_n=5)
print(x)

# Predict a single resume's category and confidence
category, confidence = predict_resume_category("html developer with 6 yr experience")
print(category, confidence)
```

If you want to use the saved artifacts from a separate script or REPL, load them with `joblib.load` and either reimplement the helper functions or run the notebook's cells to get the same `tfidf_vectorizer`, `rf_classifier`, and `label_encoder` objects in memory.

```python
from joblib import load
tfidf_vectorizer = load('resume_tfidf_vectorizer.pkl')
rf_classifier = load('resume_classifier_model.pkl')
label_encoder = load('resume_label_encoder.pkl')
```

**Notes & Tips**
- The notebook expects the Kaggle dataset `gauravduttakiit/resume-dataset` and looks for `UpdatedResumeDataSet.csv` inside the downloaded folder or archive.
- Text is lowercased and non-alphabetic characters removed in preprocessing — adjust `preprocess_text` in the notebook if you need different behavior.
- TF-IDF uses up to 5000 features and unigrams+bigrams by default; tune `TfidfVectorizer` parameters for your data.

**Contact / Next Steps**
- To improve results consider: larger datasets, more advanced text preprocessing, different classifiers (e.g., linear models, ensembles), or transformer-based embeddings for semantic matching.
