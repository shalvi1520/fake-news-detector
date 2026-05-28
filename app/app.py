"""
Fake News Detection System — Streamlit App
Run: streamlit run app/app.py
"""

import os
import sys
import json
import joblib
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.utils import predict_news, load_models_and_vectorizer

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .real-badge {
        background-color: #1a4731;
        color: #2ecc71;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #2ecc71;
    }
    .fake-badge {
        background-color: #4a1414;
        color: #e74c3c;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #e74c3c;
    }
    .metric-box {
        background-color: #1a1d23;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models_and_vectorizer()


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/news.png", width=80)
st.sidebar.title("🗞️ Fake News Detector")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Detect News", "🌐 Analyze URL", "📊 Model Analytics", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Select Model:**")
try:
    models, vectorizer = get_models()
    model_choice = st.sidebar.selectbox("", list(models.keys()))
    st.sidebar.success(f"✅ {len(models)} models loaded")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.sidebar.info("Run: `python src/preprocess.py` then `python src/train.py`")
    st.stop()


# ════════════════════════════════════════════════════════════
# PAGE 1: DETECT NEWS
# ════════════════════════════════════════════════════════════
if page == "🔍 Detect News":
    st.title("🗞️ Fake News Detection System")
    st.markdown("Paste a news article below and the AI will classify it as **Real** or **Fake**.")
    st.markdown("---")

    input_text = st.text_area(
        "📝 Paste News Article Here",
        height=250,
        placeholder="Paste a full news article or headline here...",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        detect_btn = st.button("🔍 Analyze Article", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    if detect_btn:
        if not input_text.strip():
            st.warning("⚠️ Please paste a news article first!")
        else:
            with st.spinner("🤖 Analyzing article..."):
                result = predict_news(input_text, models[model_choice], vectorizer)

            st.markdown("---")
            st.subheader("📋 Analysis Results")

            col_res, col_conf, col_words = st.columns(3)

            with col_res:
                if result["is_real"]:
                    st.markdown(f'<div class="real-badge">✅ REAL NEWS</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="fake-badge">❌ FAKE NEWS</div>',
                                unsafe_allow_html=True)

            with col_conf:
                if result["confidence"]:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["confidence"],
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={
                            "axis": {"range": [50, 100]},
                            "bar": {"color": "#2ecc71" if result["is_real"] else "#e74c3c"},
                            "steps": [
                                {"range": [50, 70], "color": "#2c2c2c"},
                                {"range": [70, 90], "color": "#1a1a1a"},
                                {"range": [90, 100], "color": "#0d0d0d"},
                            ],
                        },
                        title={"text": "Confidence"},
                    ))
                    fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

            with col_words:
                words = result["cleaned_text"].split()
                st.metric("Words Analyzed", len(words))
                st.metric("Model Used", model_choice)

            # Compare all models
            st.markdown("---")
            st.subheader("🤖 All Models Comparison")
            model_results = []
            for name, model in models.items():
                r = predict_news(input_text, model, vectorizer)
                model_results.append({
                    "Model": name,
                    "Verdict": "✅ REAL" if r["is_real"] else "❌ FAKE",
                    "Confidence": f"{r['confidence']:.1f}%" if r["confidence"] else "N/A"
                })
            st.table(pd.DataFrame(model_results))

# ════════════════════════════════════════════════════════════
# PAGE: ANALYZE URL
# ════════════════════════════════════════════════════════════
elif page == "🌐 Analyze URL":
    st.title("🌐 Real-Time URL News Analyzer")
    st.markdown("Enter a news article URL and we'll scrape + classify it instantly.")
    st.markdown("---")

    from app.utils import fetch_article_from_url

    url_input = st.text_input("🔗 Paste News Article URL", 
                               placeholder="https://www.bbc.com/news/...")
    analyze_url_btn = st.button("🚀 Fetch & Analyze", type="primary")

    if analyze_url_btn and url_input:
        with st.spinner("🌐 Fetching article..."):
            article_data = fetch_article_from_url(url_input)

        if not article_data["success"]:
            st.error(f"❌ Could not fetch article: {article_data['error']}")
        else:
            st.success("✅ Article fetched successfully!")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("📰 Article Info")
                st.markdown(f"**Title:** {article_data['title']}")
                st.markdown(f"**Published:** {article_data['publish_date']}")
                if article_data["authors"]:
                    st.markdown(f"**Authors:** {', '.join(article_data['authors'])}")
                with st.expander("📄 Article Text (Preview)"):
                    st.write(article_data["text"][:2000] + "...")

            with col2:
                st.subheader("🤖 Classification")
                full_text = article_data["title"] + " " + article_data["text"]
                result = predict_news(full_text, models[model_choice], vectorizer)

                if result["is_real"]:
                    st.markdown('<div class="real-badge">✅ REAL NEWS</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="fake-badge">❌ FAKE NEWS</div>',
                                unsafe_allow_html=True)

                if result["confidence"]:
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
# ════════════════════════════════════════════════════════════
# PAGE 2: MODEL ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    st.title("📊 Model Performance Analytics")
    st.markdown("---")

    try:
        with open("models/results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        st.error("❌ No results found. Please run `python src/train.py` first.")
        st.stop()

    # Metrics table
    st.subheader("📈 Performance Summary")
    data = []
    for name, metrics in results.items():
        if name == "best_model":
            continue
        data.append({
            "Model": name,
            "Accuracy (%)": metrics["accuracy"],
            "F1 Score (%)": metrics["f1_score"],
        })
    df = pd.DataFrame(data).sort_values("Accuracy (%)", ascending=False)
    st.dataframe(df, use_container_width=True)

    # Bar chart
    fig = px.bar(
        df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_map={"Accuracy (%)": "#3498db", "F1 Score (%)": "#2ecc71"},
        title="Model Comparison — Accuracy vs F1 Score",
    )
    fig.update_layout(yaxis_range=[85, 100])
    st.plotly_chart(fig, use_container_width=True)

    # Show saved plots if available
    st.subheader("🗺️ Saved Evaluation Plots")
    cols = st.columns(2)
    plot_files = [f for f in os.listdir("assets") if f.endswith(".png")]
    for i, plot_file in enumerate(sorted(plot_files)):
        with cols[i % 2]:
            st.image(f"assets/{plot_file}", caption=plot_file.replace("_", " ").replace(".png","").title())


# ════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ## 🗞️ Fake News Detection System

    This application uses **Natural Language Processing (NLP)** and **Machine Learning** 
    to classify news articles as Real or Fake.

    ### 🔧 How It Works
    1. **Text Preprocessing** — Article is cleaned, stopwords removed, and stemmed
    2. **TF-IDF Vectorization** — Text is converted to a numerical feature matrix
    3. **ML Classification** — Trained model predicts Real (1) or Fake (0)
    4. **Confidence Score** — Probability of prediction shown as a gauge

    ### 🤖 Models Used
    | Model | Type | Strength |
    |---|---|---|
    | Logistic Regression | Linear | Fast, interpretable |
    | Naive Bayes | Probabilistic | Excellent for text |
    | Random Forest | Ensemble | High accuracy |
    | Passive Aggressive | Online Learning | Handles large data |

    ### 📊 Dataset
    - **ISOT Fake News Dataset** (Kaggle)
    - ~44,000 articles (Real + Fake)
    - Sources: Reuters (real), various fake news sites (fake)

    ### 🛠️ Tech Stack
    `Python` · `Scikit-learn` · `NLTK` · `Streamlit` · `Plotly` · `Pandas`
    """)