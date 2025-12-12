# main.py
"""
Modern Premium ML Labs portal — single-file Streamlit app
Features:
- 11 labs (see menu)
- Gradient UI, cards, Lottie animation (optional)
- Star difficulty rating per lab (saved in session_state)
- Progress bar showing completed labs
- File uploader + download results
- Chat input helper
- Interactive plots (plotly) + static matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import scipy.stats as stats

# Try to import mlxtend (apriori). If missing, we'll handle gracefully.
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_OK = True
except Exception:
    MLXTEND_OK = False

# Try lottie
try:
    from streamlit_lottie import st_lottie
    LOTTIE_OK = True
except Exception:
    LOTTIE_OK = False

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="Машинное обучение-лабороторная работа", layout="wide")

# Custom CSS for premium gradient look
st.markdown(
    """
    <style>
    /* Background gradient */
    .main .block-container{
        background: linear-gradient(180deg, #f8f6ff 0%, #ffffff 100%);
    }
    /* Title */
    .big-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(90deg,#0ea5e9,#7c3aed);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 6px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 18px;
        font-weight:700;
    }
    /* Card */
    .card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(124,58,237,0.12);
        background: linear-gradient(180deg, rgba(237,242,255,0.8), rgba(250,250,255,0.6));
        box-shadow: 0 6px 18px rgba(99,102,241,0.06);
        margin-bottom: 18px;
    }
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg,#7c3aed,#06b6d4) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        font-weight: 800 !important;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#f3e8ff,#eef2ff);
    }
    .sidebar .sidebar-content {
        font-weight:700;
    }
    /* Star rating look tweak */
    .rating-row { display:flex; gap:6px; align-items:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">ML Labs — Modern Premium UI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">11 практических лабораторных работ по машинному обучению — интерактивно и красиво</div>', unsafe_allow_html=True)

# ---------------------------
# Data dir
# ---------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------
# Session state initialization
# ---------------------------
if "ratings" not in st.session_state:
    # ratings: map menu_key -> "1..5"
    st.session_state.ratings = {}
if "completed" not in st.session_state:
    st.session_state.completed = set()
if "uploads" not in st.session_state:
    st.session_state.uploads = {}

# ---------------------------
# Helper utilities
# ---------------------------
def save_csv_to_bytes(df: pd.DataFrame):
    buff = io.BytesIO()
    df.to_csv(buff, index=False)
    buff.seek(0)
    return buff

def download_button_df(df: pd.DataFrame, label="Download CSV", filename="data.csv"):
    b = save_csv_to_bytes(df)
    st.download_button(label, b, file_name=filename, mime="text/csv")

def show_lottie_from_url(url, height=200):
    if LOTTIE_OK:
        try:
            st_lottie(url, height=height, key=url)
        except Exception:
            st.write("")  # ignore
    else:
        # fallback - small decorative image via HTML gradient box
        st.markdown("<div style='height:150px;background:linear-gradient(90deg,#7c3aed,#06b6d4);border-radius:10px;margin-bottom:10px'></div>", unsafe_allow_html=True)

def save_rating(key, value):
    st.session_state.ratings[key] = value
    # mark as completed if rated >0
    try:
        if int(value) >= 1:
            st.session_state.completed.add(key)
    except Exception:
        pass

def show_star_widget(key):
    # radio-like horizontal stars
    cols = st.columns([1,4])
    with cols[0]:
        st.markdown(f"**{key} — Difficulty**")
    with cols[1]:
        current = st.session_state.ratings.get(key, "3")
        val = st.selectbox("", ["1","2","3","4","5"], index=int(current)-1, key=f"rating_{key}")
        save_rating(key, val)
        st.write("⭐" * int(val))

def progress_bar_for_completed():
    total = 11
    completed = len(st.session_state.completed)
    pct = int((completed/total)*100)
    st.sidebar.markdown("### Progress")
    st.sidebar.progress(pct)
    st.sidebar.write(f"{completed}/{total} completed")

# ---------------------------
# Left sidebar: navigation + uploader + progress
# ---------------------------
LAB_MENU = [
    "1. Первичный анализ наборов данных",
    "2. Анализ главных компонент (PCA)",
    "3. Линейная регрессия",
    "4. Наивный байесовский классификатор",
    "5. Машина опорных векторов (SVM)",
    "6. Бустинг (AdaBoost)",
    "7. Нейронные сети",
    "8. Кластеризация (KMeans & EM)",
    "9. Ассоциативные правила",
    "10. Онлайн обработка данных",
    "11. Распределённые вычисления"
]
choice = st.sidebar.selectbox("Выберите лабораторию", LAB_MENU)
show_lottie_from_url("https://assets2.lottiefiles.com/packages/lf20_Stt1R6.json", height=150) if LOTTIE_OK else None

# File uploader (global)
uploaded = st.sidebar.file_uploader("Upload CSV (optional) — will be used in some labs", type=["csv"])
if uploaded:
    try:
        df_up = pd.read_csv(uploaded)
        st.session_state.uploads["last"] = df_up
        st.sidebar.success("Uploaded OK")
    except Exception as e:
        st.sidebar.error("Upload failed: " + str(e))

# Chat helper
st.sidebar.markdown("---")
st.sidebar.markdown("## Помощник")
chat_val = st.sidebar.text_input("Задайте вопрос помощнику (demo):", "")
if st.sidebar.button("Ask"):
    # very simple canned responses (demo — no external LLM)
    q = chat_val.lower()
    if "как" in q or "почему" in q:
        st.sidebar.info("Это демо-ассистент. Задайте конкретный вопрос по лабе, например: 'Как увеличить n_samples?'")
    else:
        st.sidebar.info("Попробуйте спросить про параметры или ошибки — демо ответ.")

progress_bar_for_completed()

# ---------------------------
# Lab implementations
# ---------------------------

# 1) Primary analysis
def lab_primary_analysis():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1. Первичный анализ наборов данных")
    st.write("Генерация регресс. и классиф. синтетических датасетов; describe, corr heatmap, class counts.")
    n = st.slider("n_samples", 100, 5000, 800, step=100, key="lab1_n")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="lab1_seed")
    rng = np.random.RandomState(seed)
    Xr, yr = make_regression(n_samples=n, n_features=6, noise=12.0, random_state=seed)
    dfr = pd.DataFrame(Xr, columns=[f"r_f{i+1}" for i in range(Xr.shape[1])])
    dfr["target"] = yr
    st.subheader("Preview (regression)")
    st.dataframe(dfr.head())
    st.subheader("Statistics")
    st.dataframe(dfr.describe().T)
    fig = px.imshow(dfr.corr(), text_auto=".2f", title="Correlation matrix (regression)")
    st.plotly_chart(fig, use_container_width=True)
    # classification
    Xc, yc = make_classification(n_samples=n, n_features=8, n_informative=4, n_classes=3, random_state=seed)
    dfc = pd.DataFrame(Xc, columns=[f"c_f{i+1}" for i in range(Xc.shape[1])])
    dfc["target"] = yc
    st.subheader("Class distribution (classification)")
    fig2 = px.histogram(dfc, x="target", title="Class counts")
    st.plotly_chart(fig2, use_container_width=True)
    # download
    st.markdown("**Download datasets**")
    col1, col2 = st.columns(2)
    with col1:
        download_button_df(dfr, label="Download regression CSV", filename="lab1_regression.csv")
    with col2:
        download_button_df(dfc, label="Download classification CSV", filename="lab1_classification.csv")
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("1. Первичный анализ наборов данных")

# 2) PCA
def lab_pca():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("2. Анализ главных компонент (PCA)")
    st.write("PCA on synthetic 8-D data, interactive scatter of first two components and explained variance bar.")
    n = st.slider("n_samples (PCA)", 100, 5000, 800, step=50, key="pca_n")
    seed = st.number_input("random_state (PCA)", min_value=0, value=42, step=1, key="pca_seed")
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 8))
    pca = PCA(n_components=8)
    Xp = pca.fit_transform(X)
    ex = pca.explained_variance_ratio_
    dfp = pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(Xp.shape[1])])
    st.subheader("Explained variance ratio")
    fig = px.bar(x=[f"PC{i+1}" for i in range(len(ex))], y=ex, labels={"x":"PC","y":"Explained variance"})
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("PC1 vs PC2 (colored by PC3)")
    fig2 = px.scatter(dfp, x="PC1", y="PC2", color=dfp["PC3"], color_continuous_scale="Viridis", title="PCA scatter")
    st.plotly_chart(fig2, use_container_width=True)
    download_button_df(dfp.head(200), label="Download PCA sample", filename="lab2_pca_sample.csv")
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("2. Анализ главных компонент (PCA)")

# 3) Linear regression
def lab_linear_regression():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("3. Линейная регрессия")
    st.write("Synthetic 2D regression: train/test split, MSE/R2, interactive scatter with fit line.")
    n = st.slider("n_samples", 100, 2000, 500, step=50, key="lr_n")
    noise = st.slider("noise", 0.0, 50.0, 5.0, key="lr_noise")
    test_size = st.slider("test_size", 0.05, 0.5, 0.2, key="lr_test")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="lr_seed")
    rng = np.random.RandomState(seed)
    X = rng.uniform(-10,10,size=(n,2))
    coef = rng.uniform(-4,4,size=2)
    y = X.dot(coef) + rng.normal(0, noise, size=n)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_size, random_state=seed)
    model = LinearRegression().fit(Xtr, ytr)
    ypred = model.predict(Xte)
    st.metric("MSE", f"{mean_squared_error(yte, ypred):.3f}", delta=None)
    st.metric("R2", f"{r2_score(yte, ypred):.3f}", delta=None)
    # interactive scatter
    dfp = pd.DataFrame({"y_true": yte, "y_pred": ypred})
    fig = px.scatter(dfp, x="y_true", y="y_pred", trendline="ols", title="True vs Pred")
    st.plotly_chart(fig, use_container_width=True)
    download_button_df(dfp, label="Download predictions", filename="lab3_reg_preds.csv")
    joblib.dump(model, DATA_DIR / "lab3_linear.joblib")
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("3. Линейная регрессия")

# 4) Naive Bayes
def lab_naive_bayes():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("4. Наивный байесовский классификатор")
    st.write("GaussianNB on synthetic classification data; show confusion matrix and class probabilities.")
    n = st.slider("n_samples", 100, 3000, 600, step=50, key="nb_n")
    test_size = st.slider("test_size", 0.05, 0.5, 0.25, key="nb_test")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="nb_seed")
    X, y = make_classification(n_samples=n, n_features=8, n_informative=4, n_classes=3, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_size, random_state=seed)
    model = GaussianNB().fit(Xtr, ytr)
    preds = model.predict(Xte)
    st.write("Accuracy:", accuracy_score(yte, preds))
    fig = px.imshow(confusion_matrix(yte, preds), text_auto=True, title="Confusion Matrix")
    st.plotly_chart(fig, use_container_width=True)
    joblib.dump(model, DATA_DIR / "lab4_gnb.joblib")
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("4. Наивный байесовский классификатор")

# 5) SVM
def lab_svm():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("5. Машина опорных векторов (SVM)")
    st.write("2D blob data; SVM decision boundary + accuracy.")
    kernel = st.selectbox("kernel", ["linear","rbf","poly"], key="svm_kernel")
    C = st.slider("C", 0.01, 10.0, 1.0, key="svm_C")
    n_samples = st.slider("n_samples", 50, 2000, 300, step=50, key="svm_n")
    noise = st.slider("cluster_std", 0.05, 3.0, 0.6, key="svm_noise")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="svm_seed")
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise, random_state=seed)
    scaler = None
    if kernel != "linear":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X
    clf = SVC(kernel=kernel, C=C, probability=True).fit(Xs, y)
    preds = clf.predict(Xs)
    st.write("Accuracy:", accuracy_score(y, preds))
    # decision boundary (mesh)
    x_min, x_max = Xs[:,0].min()-1, Xs[:,0].max()+1
    y_min, y_max = Xs[:,1].min()-1, Xs[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,200), np.linspace(y_min,y_max,200))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure()
    fig.add_trace(go.Contour(z=Z, x=np.linspace(x_min,x_max,200), y=np.linspace(y_min,y_max,200), showscale=False, opacity=0.4, colorscale="Viridis"))
    fig.add_trace(go.Scatter(x=Xs[:,0], y=Xs[:,1], mode="markers", marker=dict(color=y, colorscale="Portland", line=dict(color="black", width=1)), name="data"))
    fig.update_layout(title=f"SVM decision boundary (kernel={kernel})", xaxis_title="x1", yaxis_title="x2")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("5. Машина опорных векторов (SVM)")

# 6) Boosting (AdaBoost)
def lab_boosting():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("6. Бустинг (AdaBoost)")
    st.write("AdaBoost classifier demo and feature importance visualization.")
    n = st.slider("n_samples", 100, 3000, 800, step=50, key="boost_n")
    n_estimators = st.slider("n_estimators", 10, 300, 50, key="boost_nest")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="boost_seed")
    X, y = make_classification(n_samples=n, n_features=12, n_informative=6, n_classes=2, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=seed)
    model = AdaBoostClassifier(n_estimators=n_estimators, random_state=seed).fit(Xtr, ytr)
    preds = model.predict(Xte)
    st.write("Accuracy:", accuracy_score(yte, preds))
    # feature importances (AdaBoost exposes feature_importances_)
    try:
        fi = model.feature_importances_
        fig = px.bar(x=list(range(len(fi))), y=fi, labels={"x":"feature","y":"importance"}, title="Feature importances (AdaBoost)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Feature importances not available for this estimator.")
    joblib.dump(model, DATA_DIR / "lab6_adaboost.joblib")
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("6. Бустинг (AdaBoost)")

# 7) Neural networks
def lab_neural_nets():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("7. Нейронные сети (MLP)")
    st.write("MLP classifier/regressor. Show confusion matrix or regression scatter.")
    mode = st.selectbox("Mode", ["classification","regression"], key="mlp_mode")
    hidden = st.slider("hidden layer size (units)", 8, 256, 64, step=8, key="mlp_hidden")
    max_iter = st.slider("max_iter", 50, 500, 200, step=10, key="mlp_iter")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="mlp_seed")
    if mode == "classification":
        X, y = make_classification(n_samples=600, n_features=12, n_informative=8, n_classes=3, random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=seed)
        model = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=max_iter, random_state=seed).fit(Xtr, ytr)
        preds = model.predict(Xte)
        st.write("Accuracy:", accuracy_score(yte, preds))
        fig = px.imshow(confusion_matrix(yte, preds), text_auto=True, title="Confusion matrix (MLP)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        X, y = make_regression(n_samples=600, n_features=10, noise=12.0, random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25, random_state=seed)
        model = MLPRegressor(hidden_layer_sizes=(hidden,), max_iter=max_iter, random_state=seed).fit(Xtr, ytr)
        preds = model.predict(Xte)
        st.write({"MSE": mean_squared_error(yte, preds), "R2": r2_score(yte, preds)})
        fig = px.scatter(x=yte, y=preds, labels={"x":"y_true","y":"y_pred"}, title="True vs Pred (MLP)")
        st.plotly_chart(fig, use_container_width=True)
    joblib.dump(model, DATA_DIR / "lab7_mlp.joblib")
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("7. Нейронные сети")

# 8) Clustering
def lab_clustering():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("8. Кластеризация (KMeans & EM)")
    st.write("Compare KMeans vs GaussianMixture on synthetic blobs.")
    algo = st.selectbox("Algo", ["KMeans","GMM","DBSCAN"], key="cluster_algo")
    n = st.slider("n_samples", 100, 3000, 600, step=50, key="cluster_n")
    centers = st.slider("true centers", 2, 8, 3, key="cluster_centers")
    std = st.slider("cluster std", 0.05, 3.0, 0.6, key="cluster_std")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="cluster_seed")
    X, y_true = make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=seed)
    if algo == "KMeans":
        k = st.slider("k (KMeans)", 2, 12, centers, key="k_kmeans")
        model = KMeans(n_clusters=k, random_state=seed).fit(X)
        labels = model.predict(X)
    elif algo == "GMM":
        model = GaussianMixture(n_components=centers, random_state=seed).fit(X)
        labels = model.predict(X)
    else:
        eps = st.slider("eps (DBSCAN)", 0.1, 5.0, 0.5, key="db_eps")
        min_s = st.slider("min_samples (DBSCAN)", 1, 30, 5, key="db_min")
        model = DBSCAN(eps=eps, min_samples=min_s).fit(X)
        labels = model.labels_
    fig = px.scatter(x=X[:,0], y=X[:,1], color=labels.astype(str), title=f"{algo} result")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("8. Кластеризация (KMeans & EM)")

# 9) Association rules (apriori)
def lab_association_rules():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("9. Ассоциативные правила (Apriori)")
    st.write("Generate synthetic transactions and run apriori -> association_rules (if mlxtend installed).")
    n_trans = st.slider("n_transactions", 50, 5000, 800, step=50, key="assoc_n")
    n_items = st.slider("n_items", 5, 40, 10, key="assoc_items")
    support = st.slider("min_support", 0.01, 0.5, 0.05, step=0.01, key="assoc_support")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="assoc_seed")
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n_trans):
        row = (rng.rand(n_items) < 0.12).astype(int)  # sparse
        data.append(row)
    df = pd.DataFrame(data, columns=[f"item_{i+1}" for i in range(n_items)])
    st.dataframe(df.head())
    if not MLXTEND_OK:
        st.error("mlxtend not installed — apriori unavailable. Install with `pip install mlxtend`")
    else:
        try:
            freq = apriori(df, min_support=support, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=0.5)
            st.subheader("Frequent itemsets")
            st.dataframe(freq.sort_values("support", ascending=False).head(20))
            st.subheader("Association rules (sample)")
            if not rules.empty:
                st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]].head(20))
            else:
                st.info("No rules found with the chosen thresholds.")
        except Exception as e:
            st.error("Apriori failed: " + str(e))
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("9. Ассоциативные правила")

# 10) Online processing simulation
def lab_online_processing():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("10. Онлайн обработка данных (stream simulation)")
    st.write("Simulate streaming numeric data and show sliding-window stats + live chart.")
    total = st.slider("total events", 200, 20000, 2000, step=100, key="stream_total")
    window = st.slider("window size", 10, 500, 100, key="stream_window")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="stream_seed")
    rng = np.random.RandomState(seed)
    series = np.cumsum(rng.normal(0, 1, size=total))
    pos = st.slider("stream position", 0, max(0, total-window), 0, key="stream_pos")
    window_data = series[pos:pos+window]
    dfw = pd.DataFrame({"value": window_data})
    st.line_chart(dfw)
    st.write(dfw.describe().T)
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("10. Онлайн обработка данных")

# 11) Distributed computing sim (toy)
def lab_distributed_sim():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("11. Распределённые вычисления (симуляция)")
    st.write("Simulate splitting tasks across workers and visualize load and speedup.")
    workers = st.slider("workers", 1, 32, 4, key="dist_workers")
    tasks = st.slider("tasks count", 10, 200, 60, key="dist_tasks")
    seed = st.number_input("random_state", min_value=0, value=42, step=1, key="dist_seed")
    rng = np.random.RandomState(seed)
    task_times = rng.uniform(0.01, 0.6, size=tasks)
    serial = task_times.sum()
    groups = np.array_split(task_times, workers)
    parallel = max([arr.sum() for arr in groups])
    speedup = serial / parallel
    st.write({"serial_total": float(serial), "parallel_est": float(parallel), "speedup_est": float(speedup)})
    fig = px.bar(x=list(range(1, workers+1)), y=[arr.sum() for arr in groups], labels={"x":"worker","y":"load"}, title="Simulated worker loads")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    show_star_widget("11. Распределённые вычисления")

# ---------------------------
# Router: call chosen lab
# ---------------------------
lab_map = {
    LAB_MENU[0]: lab_primary_analysis,
    LAB_MENU[1]: lab_pca,
    LAB_MENU[2]: lab_linear_regression,
    LAB_MENU[3]: lab_naive_bayes,
    LAB_MENU[4]: lab_svm,
    LAB_MENU[5]: lab_boosting,
    LAB_MENU[6]: lab_neural_nets,
    LAB_MENU[7]: lab_clustering,
    LAB_MENU[8]: lab_association_rules,
    LAB_MENU[9]: lab_online_processing,
    LAB_MENU[10]: lab_distributed_sim
}

# Render the selected lab
try:
    lab_func = lab_map.get(choice)
    if lab_func:
        lab_func()
    else:
        st.info("Select a lab from the left menu.")
except Exception as e:
    st.error("An error occurred while running the lab: " + str(e))

# ---------------------------
# Bottom: show ratings summary & download
# ---------------------------
st.markdown("---")
st.header("Summary & Export")
st.write("Current difficulty ratings (you can change them any time):")
if st.session_state.ratings:
    df_rates = pd.DataFrame.from_dict(st.session_state.ratings, orient="index", columns=["rating"])
    st.dataframe(df_rates)
    b = io.BytesIO()
    df_rates.to_csv(b)
    b.seek(0)
    st.download_button("Download ratings CSV", data=b, file_name="lab_ratings.csv")
else:
    st.info("No ratings yet — open labs and rate them!")

st.markdown("### Overall progress")
total = len(LAB_MENU)
completed = len(st.session_state.completed)
st.write(f"Completed labs: **{completed}/{total}**")
st.progress(int((completed/total)*100))

# Footer
st.markdown("<hr>")
st.markdown("© ML Labs")
