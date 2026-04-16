#  Movie Recommendation System

A content-based movie recommendation system built with Python that suggests similar movies using **TF-IDF vectorization** and **cosine similarity**. Trained on the TMDB Movies Metadata dataset containing over **45,000 movies**.

---

##  Overview

This project processes movie metadata (overviews, taglines, and genres) to generate a rich textual representation for each film. Using NLP preprocessing and TF-IDF vectorization, the system computes pairwise cosine similarity scores to recommend movies that are contextually similar to a given title.

---

##  Features

-  **Content-Based Filtering** — Recommends movies based on plot overviews, taglines, and genres
-  **NLP Preprocessing** — Lowercasing, punctuation removal, stopword filtering, and lemmatization
-  **TF-IDF Vectorization** — Converts text tags into high-dimensional sparse feature vectors (up to 50,000 features, unigrams + bigrams)
-  **Cosine Similarity** — Measures textual similarity between movies in vector space
-  **Model Persistence** — All trained models and data are serialized using `pickle` for fast inference

---

##  Project Structure

```
movie_recommendation_system/
│
├── movie_recommendation_system.ipynb   # Main Jupyter Notebook (EDA + model training)
├── movies_metadata.csv                 # Raw TMDB dataset (~45,466 movies, 24 columns)
│
├── df.pkl                              # Processed DataFrame (pickled)
├── tfidf.pkl                           # Fitted TF-IDF vectorizer (pickled)
├── tfidf_matrix.pkl                    # TF-IDF feature matrix (pickled)
├── indices.pkl                         # Movie title → index mapping (pickled)
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Ignored files (large data, venv, cache)
└── README.md                           # Project documentation
```

> **Note:** `*.csv` and `*.pkl` files are excluded from version control via `.gitignore` due to their large size.

---

##  How It Works

### Pipeline

```
Raw CSV Data
    │
    ▼
Feature Selection
(title, overview, genres, tagline, vote_average, popularity)
    │
    ▼
Data Cleaning
(drop nulls on title, fill NaN in overview/tagline with "")
    │
    ▼
Genre Parsing
(ast.literal_eval → extract genre names as space-separated string)
    │
    ▼
Tag Construction
tags = overview + " " + tagline + " " + genres
    │
    ▼
NLP Preprocessing  (per tag)
  - Lowercase
  - Remove punctuation (regex)
  - Remove English stopwords (NLTK)
  - Lemmatize words (WordNetLemmatizer)
    │
    ▼
TF-IDF Vectorization
(max_features=50,000 | ngram_range=(1,2) | stop_words='english')
Shape: (42,277 movies × 50,000 features)
    │
    ▼
Cosine Similarity (on-the-fly per query)
    │
    ▼
Top-N Most Similar Movies
```

### Recommendation Function

```python
def recommend(title, n=10):
    if title not in indices:
        return ['Movie not found']
    idx = indices[title]
    sim_score = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_score.argsort()[::-1][1:n+1]
    return df['title'].iloc[similar_idx]
```

**Example:**
```python
recommend('Avatar')
# Returns top 10 contextually similar movies
```

---

##  Dataset

| Property       | Details                          |
|----------------|----------------------------------|
| **Source**     | TMDB Movies Metadata (Kaggle)    |
| **Raw size**   | 45,466 rows × 24 columns         |
| **After clean**| ~42,277 unique movies             |
| **Key columns used** | `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity` |

---

##  Tech Stack

| Library         | Version   | Purpose                          |
|-----------------|-----------|----------------------------------|
| `pandas`        | 2.2.2     | Data loading and manipulation    |
| `numpy`         | 2.0.1     | Numerical operations             |
| `scikit-learn`  | 1.5.1     | TF-IDF vectorization, cosine similarity |
| `scipy`         | 1.13.1    | Sparse matrix operations         |
| `nltk`          | —         | Stopwords, lemmatization         |
| `matplotlib`    | —         | EDA visualizations               |
| `seaborn`       | —         | EDA visualizations               |

---

##  Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/movie_recommendation_system.git
cd movie_recommendation_system
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install nltk matplotlib seaborn jupyter
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Get the Dataset

Download `movies_metadata.csv` from [Kaggle — TMDB Movie Metadata](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and place it in the project root.

### 6. Run the Notebook

```bash
jupyter notebook movie_recommendation_system.ipynb
```

Run all cells to train the model and generate the `.pkl` artifacts.

---

##  Pre-trained Artifacts

After running the notebook, the following serialized files are generated:

| File               | Description                                      |
|--------------------|--------------------------------------------------|
| `df.pkl`           | Preprocessed DataFrame with tags                 |
| `tfidf.pkl`        | Fitted `TfidfVectorizer` instance                |
| `tfidf_matrix.pkl` | Sparse TF-IDF feature matrix (42,277 × 50,000)  |
| `indices.pkl`      | `pd.Series` mapping movie titles to row indices  |

These are loaded at inference time without retraining.

---

##  Future Improvements

- [ ] Add collaborative filtering using user ratings
- [ ] Integrate a weighted scoring system (combine content similarity + `vote_average` + `popularity`)
- [ ] Build an interactive web interface (Flask / Streamlit)
- [ ] Fetch live movie poster images via TMDB API
- [ ] Implement fuzzy title matching for typo-tolerant queries

---

##  License

This project is open-source and available under the [MIT License](LICENSE).

---

##  Author

Built as a personal machine learning project exploring NLP-based content filtering for film recommendations.
