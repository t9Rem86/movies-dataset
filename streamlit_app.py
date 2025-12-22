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

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('data/heart.csv')

df = load_data()

st.sidebar.title("📍 Навигация")
page = st.sidebar.radio("Выберите страницу:", ["Визуализация исходных данных", "Результаты анализа"])

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
if page == "Визуализация исходных данных":
    st.title("📄 Визуализация исходных данных")

    # KPI
    col1, col2, col3 = st.columns(3)
    col1.metric("Количество записей", filtered_df.shape[0])
    col2.metric("Средний возраст", round(filtered_df.age.mean(), 1))
    col3.metric("Доля с заболеванием (%)", round(filtered_df.target.mean() * 100, 1))

    st.markdown("---")

    # Таблица
    st.subheader("📊Таблица данных")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("📋 Статистическое описание признаков")

    stats_df = filtered_df.describe().T[
        ["mean", "50%", "std", "min", "max"]
    ]
    
    stats_df.rename(
        columns={
            "mean": "Среднее",
            "50%": "Медиана",
            "std": "Стандартное отклонение",
            "min": "Минимум",
            "max": "Максимум"
        },
        inplace=True
    )
    
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("📊 Распределения признаков")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Распределение возраста (age)**")
        fig_age = px.histogram(filtered_df, x="age", nbins=20)
        st.plotly_chart(fig_age, use_container_width=True, key="age_hist")
    
    with col2:
        st.markdown("**Распределение холестерина (chol)**")
        fig_chol = px.box(filtered_df, y="chol")
        st.plotly_chart(fig_chol, use_container_width=True, key="chol_box")
    
    st.markdown("**Распределение максимального пульса (thalach)**")
    fig_thalach = px.histogram(filtered_df, x="thalach", nbins=20)
    st.plotly_chart(fig_thalach, use_container_width=True, key="thalach_hist")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**Пол (sex)**")
        sex_counts = filtered_df["sex"].value_counts().reset_index()
        sex_counts.columns = ["sex", "count"]
        sex_counts["sex"] = sex_counts["sex"].replace({0: "women", 1: "men"})
        fig_sex = px.bar(sex_counts, x="sex", y="count")
        st.plotly_chart(fig_sex, use_container_width=True, key="sex_bar")
    
    with col4:
        st.markdown("**Тип боли в груди (cp)**")
        cp_counts = filtered_df["cp"].value_counts().reset_index()
        cp_counts.columns = ["cp", "count"]
        fig_cp = px.bar(cp_counts, x="cp", y="count")
        st.plotly_chart(fig_cp, use_container_width=True, key="cp_bar")

    st.subheader("Корреляционная матрица")

    # Размер графика меньше
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Рисуем heatmap с значениями на пересечениях
    sns.heatmap(
        filtered_df.corr(), 
        cmap="coolwarm", 
        annot=True, 
        fmt=".2f", 
        annot_kws={"size":6},  # уменьшили шрифт
        linewidths=0.5, 
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("🔵 Scatter plots (пары признаков)")

    st.markdown("**Возраст vs Максимальный пульс**")
    fig_scatter_1 = px.scatter(
        filtered_df,
        x="age",
        y="thalach",
        color="target",
        labels={"target": "Заболевание"}
    )
    st.plotly_chart(fig_scatter_1, use_container_width=True, key="scatter_age_thalach")
    
    st.markdown("**Возраст vs Холестерин**")
    fig_scatter_2 = px.scatter(
        filtered_df,
        x="age",
        y="chol",
        color="target",
        labels={"target": "Заболевание"}
    )
    st.plotly_chart(fig_scatter_2, use_container_width=True, key="scatter_age_chol")


    st.subheader("🥧 Пропорции категориальных признаков")

    st.markdown("**Пол пациентов**")
    fig_pie_sex = px.pie(
        filtered_df,
        names="sex",
        title="Распределение по полу"
    )
    st.plotly_chart(fig_pie_sex, use_container_width=True, key="pie_sex")
    
    st.markdown("**Наличие сердечного заболевания**")
    fig_pie_target = px.pie(
        filtered_df,
        names="target",
        title="Распределение целевой переменной"
    )
    st.plotly_chart(fig_pie_target, use_container_width=True, key="pie_target")


# =========================
# СТРАНИЦА 2: РЕЗУЛЬТАТЫ АНАЛИЗА
# =========================
if page == "Результаты анализа":
    st.title("📊 Результаты анализа")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # =========================
    # ВЫБОР МОДЕЛИ
    # =========================
    st.sidebar.markdown("### ⚙️ Настройки анализа")

    model_type = st.sidebar.selectbox(
        "Выберите модель",
        ["Логистическая регрессия", "Случайный лес"]
    )

    # =========================
    # ПОДГОТОВКА ДАННЫХ
    # =========================
    X = filtered_df.drop("target", axis=1)
    y = filtered_df["target"]

    # 🔒 ПРОВЕРКА: есть ли оба класса
    if y.nunique() < 2:
        st.warning(
            "⚠️ Для выбранных фильтров присутствует только один класс целевой переменной.\n\n"
            "Модель классификации не может быть обучена.\n"
            "Расширьте диапазон возраста или измените фильтр по полу."
        )
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 🔒 ДОП. ПРОВЕРКА после разбиения
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.warning(
            "⚠️ После разбиения данных в обучающей или тестовой выборке "
            "остался только один класс.\n\n"
            "Измените фильтры для корректного анализа."
        )
        st.stop()

    # =========================
    # ОБУЧЕНИЕ МОДЕЛИ
    # =========================
    if model_type == "Логистическая регрессия":
        model = LogisticRegression(max_iter=1000)
        model_name = "Логистическая регрессия"
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        model_name = "Случайный лес"

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # predict_proba теперь безопасен
    y_prob = model.predict_proba(X_test)[:, 1]

    # =========================
    # KPI-МЕТРИКИ
    # =========================
    acc = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    col1, col2 = st.columns(2)
    col1.metric("Точность (Accuracy)", round(acc, 3))
    col2.metric("ROC-AUC", round(auc_score, 3))

    st.markdown("---")

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("Матрица ошибок классификации")

    cm = confusion_matrix(y_test, y_pred)

    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Предсказанный класс",
            y="Истинный класс",
            color="Количество"
        ),
        x=["Нет заболевания", "Есть заболевание"],
        y=["Нет заболевания", "Есть заболевание"],
        title="Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # =========================
    # ROC-КРИВАЯ
    # =========================
    st.subheader("ROC-кривая")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig_roc = px.line(
        x=fpr,
        y=tpr,
        labels={
            "x": "Доля ложноположительных (FPR)",
            "y": "Доля истинно положительных (TPR)"
        },
        title=f"ROC-кривая (AUC = {auc_score:.2f})"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.subheader("🔥 Влияние признаков на результат")

    if model_type == "Логистическая регрессия":
        importance = pd.Series(
            model.coef_[0],
            index=X.columns
        ).sort_values()

        title = "Коэффициенты логистической регрессии"
        x_label = "Значение коэффициента"

    else:
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values()

        title = "Важность признаков (Случайный лес)"
        x_label = "Вклад признака"

    fig_imp = px.bar(
        importance,
        orientation="h",
        labels={
            "value": x_label,
            "index": "Признак"
        },
        title=title
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # =========================
    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    # =========================
    st.subheader("📋 Результаты модели")

    comparison_df = pd.DataFrame({
        "Модель": [model_name],
        "Точность (Accuracy)": [round(acc, 3)],
        "ROC-AUC": [round(auc_score, 3)]
    })

    st.dataframe(comparison_df, use_container_width=True)

    # =========================
    # INSIGHTS
    # =========================
    st.success(
        f"""
📌 **Инсайты для выбранных фильтров:**

- Используемая модель: **{model_name}**
- Точность классификации: **{acc:.2f}**
- ROC-AUC: **{auc_score:.2f}**
- Наиболее значимые признаки:
  **{importance.index[-1]}**, **{importance.index[-2]}**

ℹ️ Сужение фильтров по возрасту и полу может приводить
к изменению структуры классов и, как следствие,
к изменению качества модели.
"""
    )

