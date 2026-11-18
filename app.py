import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sentiment Chat Analyser",
    page_icon="💬",
    layout="wide"
)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset",
    "final_cleaned_dataset.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "Models",
    "model_results.pkl"
)

MODEL_PATHS = {
    "Linear SVM": "Models/Linear_SVM_bundle.pkl",
    "Logistic Regression": "Models/Logistic_Regression_bundle.pkl",
    "Naive Bayes": "Models/Naive_Bayes_bundle.pkl",
    "Random Forest": "Models/Random_Forest_bundle.pkl"
}

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    models = {}

    for name, relative_path in MODEL_PATHS.items():

        path = os.path.join(BASE_DIR, relative_path)

        if os.path.exists(path):
            models[name] = joblib.load(path)

    return models


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH,
        encoding="utf-8"
    )

    # Replace missing text values with empty strings
    for column in ["subject", "body", "combined_text"]:

        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    return df


# ==========================================
# LOAD MODEL RESULTS
# ==========================================

@st.cache_data
def load_results():

    if os.path.exists(RESULTS_PATH):
        return joblib.load(RESULTS_PATH)

    return {}


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# LOAD MODELS AND RESULTS
# ==========================================

try:

    models = load_models()
    model_error = None

except Exception as e:

    models = {}
    model_error = str(e)


try:

    results = load_results()
    results_error = None

except Exception as e:

    results = {}
    results_error = str(e)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("💬 Sentiment Chat Analyser")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Model Performance",
        "Sentiment Analysis",
        "Analysis History",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Machine Learning Based Sentiment Analysis"
)


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.title("💬 Sentiment Chat Analyser")

    st.subheader(
        "Machine Learning Based Chat Sentiment Analysis System"
    )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write("""
    Sentiment Chat Analyser is a machine learning based
    application designed to analyse chat messages and
    identify their sentiment.

    The system uses four machine learning algorithms
    to classify text into Positive, Negative and Neutral
    sentiments.

    It provides an interactive dashboard where users
    can enter messages and receive sentiment predictions.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("ML Models", "4")

    col2.metric("Sentiment Classes", "3")

    col3.metric("Dataset Records", "7,861")

    st.markdown("---")

    st.subheader("Machine Learning Models")

    col1, col2 = st.columns(2)

    with col1:

        st.info("🤖 Linear SVM")

        st.info("🤖 Logistic Regression")

    with col2:

        st.info("🤖 Naive Bayes")

        st.info("🤖 Random Forest")

    st.markdown("---")

    if results:

        st.subheader("Best Performing Model")

        best_model = max(
            results,
            key=lambda x: results[x]["F1 Score"]
        )

        st.success(
            f"🏆 {best_model} - "
            f"Accuracy: {results[best_model]['Accuracy']:.2f}%"
        )


# ==========================================
# DATASET OVERVIEW
# ==========================================

elif page == "Dataset Overview":

    st.title("📊 Dataset Overview")

    try:

        df = load_data()

        # Dataset metrics

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{df.shape[0]:,}"
        )

        col2.metric(
            "Total Columns",
            df.shape[1]
        )

        # Count records where BOTH subject and body are empty

        text_missing = 0

        if "subject" in df.columns and "body" in df.columns:

            text_missing = int(
                (
                    (df["subject"].str.strip() == "") &
                    (df["body"].str.strip() == "")
                ).sum()
            )

        col3.metric(
            "Records Without Text",
            text_missing
        )

        st.markdown("---")

        st.success(
            "Cleaned dataset loaded successfully!"
        )

        st.write(
            f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns"
        )

        st.markdown("---")

        # Dataset preview

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.markdown("---")

        # Missing value analysis

        st.subheader("Missing Value Analysis")

        missing_df = pd.DataFrame({

            "Column Name": df.columns,

            "Data Type": df.dtypes.astype(str).values,

            "Non-Null Values": df.notnull().sum().values,

            "Missing Values": df.isnull().sum().values

        })

        st.dataframe(
            missing_df,
            use_container_width=True
        )

        st.info(
            f"Total missing values across all columns: "
            f"{int(df.isnull().sum().sum()):,}"
        )

        st.caption(
            "Missing metadata such as sender name, receiver name, "
            "or date may remain. Subject and body are the important "
            "text fields for sentiment analysis. A record is counted "
            "as missing text only when both fields are empty."
        )

        st.markdown("---")

        # Dataset statistics

        st.subheader("Dataset Statistics")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )


# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("📊 Model Performance Comparison")

    st.write("""
    Compare the performance of the four machine learning
    algorithms used in Sentiment Chat Analyser.
    """)

    st.markdown("---")

    if results:

        # Create comparison dataframe

        comparison_df = pd.DataFrame({

            "Model": list(results.keys()),

            "Accuracy": [
                results[m]["Accuracy"]
                for m in results
            ],

            "Precision": [
                results[m]["Precision"]
                for m in results
            ],

            "Recall": [
                results[m]["Recall"]
                for m in results
            ],

            "F1 Score": [
                results[m]["F1 Score"]
                for m in results
            ]

        })

        # Model selection

        selected_result_model = st.selectbox(
            "Select Model",
            list(results.keys())
        )

        selected_metrics = results[selected_result_model]

        st.subheader(
            f"{selected_result_model} Performance"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{selected_metrics['Accuracy']:.2f}%"
        )

        col2.metric(
            "Precision",
            f"{selected_metrics['Precision']:.2f}%"
        )

        col3.metric(
            "Recall",
            f"{selected_metrics['Recall']:.2f}%"
        )

        col4.metric(
            "F1 Score",
            f"{selected_metrics['F1 Score']:.2f}%"
        )

        st.markdown("---")

        # Comparison table

        st.subheader("Model Comparison Table")

        st.dataframe(
            comparison_df.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Performance chart

        st.subheader("Model Performance Graph")

        chart_df = comparison_df.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ]

        st.bar_chart(chart_df)

        st.markdown("---")

        # Confusion matrix

        st.subheader("Confusion Matrix")

        st.write(
            f"Confusion matrix for {selected_result_model}"
        )

        cm = np.array(
            selected_metrics["Confusion Matrix"]
        )

        labels = selected_metrics["Labels"]

        cm_df = pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        st.caption(
            "Rows represent actual sentiment labels and columns "
            "represent predicted sentiment labels."
        )

        st.markdown("---")

        # Best model

        best_model = max(
            results,
            key=lambda x: results[x]["F1 Score"]
        )

        st.subheader("🏆 Best Performing Model")

        st.success(
            f"{best_model} achieved the highest F1 Score of "
            f"{results[best_model]['F1 Score']:.2f}% "
            f"with an accuracy of "
            f"{results[best_model]['Accuracy']:.2f}%."
        )

    else:

        st.warning(
            "Model results not found. "
            "Please run train_models.py first."
        )


# ==========================================
# SENTIMENT ANALYSIS
# ==========================================

elif page == "Sentiment Analysis":

    st.title("💬 Sentiment Chat Analyser")

    st.write(
        "Enter your message to analyse its sentiment."
    )

    st.markdown("---")

    if model_error:

        st.error(
            f"Unable to load trained models: {model_error}"
        )

    elif not models:

        st.error(
            "No trained models found. "
            "Please check your Models folder."
        )

    else:

        # Message input

        user_text = st.text_area(
            "Enter Your Message",
            placeholder="Type your chat message here...",
            height=150,
            key="analysis_message"
        )

        # Model selection

        selected_model = st.selectbox(
            "Select Machine Learning Model",
            list(models.keys()),
            key="analysis_model"
        )

        st.markdown("---")

        # Analyse button

        if st.button(
            "🔍 Analyse Sentiment",
            use_container_width=True,
            key="analyse_button"
        ):

            if user_text.strip():

                try:

                    selected = models[selected_model]

                    model = selected["model"]

                    vectorizer = selected["vectorizer"]

                    text_vector = vectorizer.transform(
                        [user_text]
                    )

                    prediction = model.predict(
                        text_vector
                    )[0]

                    st.markdown("---")

                    st.subheader("Sentiment Result")

                    if prediction == "Positive":

                        st.success(
                            "😊 POSITIVE SENTIMENT"
                        )

                    elif prediction == "Negative":

                        st.error(
                            "😞 NEGATIVE SENTIMENT"
                        )

                    else:

                        st.info(
                            "😐 NEUTRAL SENTIMENT"
                        )

                    st.write(
                        "**Selected Model:**",
                        selected_model
                    )

                    st.write(
                        "**Predicted Sentiment:**",
                        prediction
                    )

                    # Save analysis history

                    st.session_state.history.append({

                        "Date & Time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),

                        "Message": user_text,

                        "Model": selected_model,

                        "Sentiment": str(prediction)

                    })

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )

            else:

                st.warning(
                    "Please enter a chat message."
                )


# ==========================================
# ANALYSIS HISTORY
# ==========================================

elif page == "Analysis History":

    st.title("📋 Analysis History")

    st.write(
        "View your previously analysed chat messages."
    )

    st.markdown("---")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader("History Statistics")

        total = len(st.session_state.history)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Analyses",
            total
        )

        col2.metric(
            "Positive",
            sum(
                1 for item in st.session_state.history
                if item["Sentiment"] == "Positive"
            )
        )

        col3.metric(
            "Negative",
            sum(
                1 for item in st.session_state.history
                if item["Sentiment"] == "Negative"
            )
        )

        # Download history

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Analysis History",
            data=csv,
            file_name="sentiment_analysis_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Analysis History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No analysis history available. "
            "Analyse a message to see your history here."
        )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("ℹ️ About Sentiment Chat Analyser")

    st.markdown("---")

    st.subheader("Project Description")

    st.write("""
    Sentiment Chat Analyser is a machine learning based
    project developed to identify the emotional tone
    of text messages.

    The application analyses user-provided messages
    and classifies them into three categories:

    - Positive
    - Negative
    - Neutral

    It uses four different machine learning algorithms
    to perform sentiment classification.
    """)

    st.markdown("---")

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Joblib
    - VADER Sentiment
    """)

    st.markdown("---")

    st.subheader("Machine Learning Algorithms")

    st.write("""
    1. Linear Support Vector Machine
    2. Logistic Regression
    3. Naive Bayes
    4. Random Forest
    """)

    st.markdown("---")

    st.subheader("Project Objective")

    st.write("""
    The main objective is to develop an interactive
    sentiment analysis system that can automatically
    understand and classify the sentiment of chat messages.
    """)

    st.success(
        "Thank you for using Sentiment Chat Analyser!"
    )