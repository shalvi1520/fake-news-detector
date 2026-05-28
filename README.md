# 🗞️ Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

## 📌 Objective
A machine learning application that classifies news articles as **Real** or **Fake** using NLP and four ML algorithms — with an interactive Streamlit dashboard.

## ✨ Features
- ✅ Multi-model classification (Logistic Regression, Naive Bayes, Random Forest, Passive Aggressive)
- ✅ Real-time URL article scraping and classification
- ✅ Confidence score gauge for predictions
- ✅ All-models comparison on same article
- ✅ Performance analytics dashboard (accuracy, F1, confusion matrices, ROC curves)
- ✅ Word cloud & EDA visualizations

## 🛠️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| ML/NLP | Scikit-learn, NLTK |
| Visualization | Plotly, Seaborn, Matplotlib, WordCloud |
| Web App | Streamlit |
| URL Scraping | newspaper3k |
| Dataset | ISOT Fake News (Kaggle) |

## 📊 Model Results
| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | ~98.7% | ~98.7% |
| Passive Aggressive | ~99.2% | ~99.2% |
| Random Forest | ~99.0% | ~99.0% |
| Naive Bayes | ~95.8% | ~95.8% |

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
```

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
Place `Fake.csv` and `True.csv` in `data/raw/`

### 4. Run Pipeline (Train Models)
```bash
python run_pipeline.py
```

### 5. Launch Web App
```bash
streamlit run app/app.py
```

## 📁 Project Structure
fake-news-detector/
├── data/raw/              # Dataset CSVs (not tracked in git)
├── notebooks/             # EDA notebook
├── src/                   # ML pipeline modules
│   ├── preprocess.py      # Text cleaning & TF-IDF
│   ├── train.py           # Model training
│   └── evaluate.py        # Metrics & plots
├── models/                # Saved .pkl files (not tracked)
├── app/                   # Streamlit application
│   ├── app.py
│   └── utils.py
├── assets/                # Generated charts & images
├── run_pipeline.py        # One-click pipeline runner
└── requirements.txt
