import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# =========================
# НАСТРОЙКИ
# =========================
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================
@st.cache_data
def load_data():
    return pd.read_csv('data/heart.csv')

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:", ["Raw Data Visualization", "Analysis Results"])

st.sidebar.markdown("---")
age_range = st.sidebar.slider(
    "Диапазон возраста",
    int(df.age.min()),
    int(df.age.max()),
    (int(df.age.min()), int(df.age.max()))
)

sex_filter = st.sidebar.selectbox("Пол", ["Все", "Мужчины", "Женщины"])

# =========================
# ФИЛЬТРАЦИЯ
# =========================
filtered_df = df[
    (df.age >= age_range[0]) & (df.age <= age_range[1])
]

if sex_filter == "Мужчины":
    filtered_df = filtered_df[filtered_df.sex == 1]
elif sex_filter == "Женщины":
    filtered_df = filtered_df[filtered_df.sex == 0]

# =========================
# СТРАНИЦА 1
# =========================
if page == "Raw Data Visualization":
    st.title("📄 Raw Data Visualization")

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Количество записей", filtered_df.shape[0])
    col2.metric("Средний возраст", round(filtered_df.age.mean(), 1))
    col3.metric("Доля с заболеванием (%)", round(filtered_df.target.mean() * 100, 1))

    st.markdown("---")

    # Таблица
    st.subheader("Таблица данных")
    st.dataframe(filtered_df, use_container_width=True)

    # Гистограмма возраста
    st.subheader("Распределение возраста")
    fig_age = px.histogram(filtered_df, x="age", nbins=20)
    st.plotly_chart(fig_age, use_container_width=True)

    # Boxplot холестерина
    st.subheader("Распределение холестерина")
    fig_chol = px.box(filtered_df, y="chol")
    st.plotly_chart(fig_chol, use_container_width=True)

    # Корреляционная матрица
    st.subheader("Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(filtered_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =========================
# СТРАНИЦА 2
# =========================
if page == "Analysis Results":
    st.title("📊 Analysis Results")

    X = filtered_df.drop("target", axis=1)
    y = filtered_df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy модели", round(acc, 3))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["No Disease", "Disease"],
        y=["No Disease", "Disease"]
    )
    st.plotly_chart(fig_cm)

    # ROC Curve
    st.subheader("ROC-кривая")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig_roc = px.line(
        x=fpr, y=tpr,
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        title=f"ROC Curve (AUC = {roc_auc:.2f})"
    )
    st.plotly_chart(fig_roc)

    st.success(
        "📌 Insight: при выбранных фильтрах модель демонстрирует стабильное качество "
        "предсказания сердечных заболеваний."
    )
