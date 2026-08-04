import pandas as pd
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


st.set_page_config(
    page_title="AI Tweet Detection",
    page_icon="🔎",
    layout="wide"
)

DATA_PATH = "combined_ai_tweet_detection_dataset.csv"
HF_BERT_MODEL = "abdGazal/ai-tweet-bert"

LABELS = {0: "Human-written", 1: "AI-generated"}
UNCERTAIN_THRESHOLD = 0.65


st.markdown(
    """
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        max-width: 950px;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #f7f7fb 0%, #eef2ff 100%);
        border: 1px solid #dfe3f0;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .uncertain-card {
        padding: 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #fff7ed 0%, #fffbeb 100%);
        border: 1px solid #fed7aa;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .small-note {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .section-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


@st.cache_resource(show_spinner="Preparing lightweight models. This may take a minute...")
def train_lightweight_models(df):
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logistic_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12000,
            min_df=2,
            max_df=0.95
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    random_forest_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=7000,
            min_df=2,
            max_df=0.95
        )),
        ("model", RandomForestClassifier(
            n_estimators=80,
            max_depth=60,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        ))
    ])

    models = {
        "Random Forest": random_forest_model,
        "Logistic Regression": logistic_model
    }

    results = []
    matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0)
        })

        matrices[name] = confusion_matrix(y_test, pred)

    results.append({
        "Model": "BERT",
        "Accuracy": 0.9209,
        "Precision": 0.9220,
        "Recall": 0.9376,
        "F1-score": 0.9297
    })

    return models, pd.DataFrame(results), matrices


@st.cache_resource(show_spinner="Loading BERT from Hugging Face. This may take a little longer...")
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(HF_BERT_MODEL)
    model.eval()
    return tokenizer, model


def label_name(value):
    return LABELS[int(value)]


def predict_lightweight(model, text):
    prediction = int(model.predict([text])[0])
    probability_ai = float(model.predict_proba([text])[0][1])
    confidence = probability_ai if prediction == 1 else 1 - probability_ai
    return prediction, probability_ai, confidence


def predict_bert(text):
    tokenizer, model = load_bert_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    probability_human = float(probabilities[0])
    probability_ai = float(probabilities[1])

    prediction = 1 if probability_ai >= probability_human else 0
    confidence = max(probability_human, probability_ai)

    return prediction, probability_ai, confidence


def get_prediction(model_choice, text, lightweight_models):
    if model_choice == "BERT":
        return predict_bert(text)

    return predict_lightweight(lightweight_models[model_choice], text)


def final_display_label(prediction, confidence):
    if confidence < UNCERTAIN_THRESHOLD:
        return "Uncertain / Needs review"
    return label_name(prediction)


def confidence_note(prediction, confidence):
    if confidence < UNCERTAIN_THRESHOLD:
        return (
            "This text does not give the model enough strong evidence. "
            "A short or generic sentence may look slightly human or AI-like depending on the model, so it should be reviewed instead of treated as a firm prediction."
        )
    if prediction == 1:
        return "The text has patterns that are closer to AI-generated writing in the training data."
    return "The text has patterns that are closer to human-written tweets in the training data."


df = load_data()
lightweight_models, results_df, confusion_matrices = train_lightweight_models(df)


with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox(
        "Model",
        ["BERT", "Random Forest", "Logistic Regression"],
        help="Choose the model used for the main prediction."
    )

    st.divider()

    st.subheader("Confidence rule")
    st.write(
        f"Predictions below {int(UNCERTAIN_THRESHOLD * 100)}% confidence are shown as uncertain."
    )

    st.divider()

    st.subheader("How to use")
    st.write("1. Paste a tweet.")
    st.write("2. Choose a model.")
    st.write("3. Click Predict.")
    st.write("4. Review the prediction and confidence.")

    st.divider()

    st.caption(
        "This app is a review aid. It should not be used as final proof that a post was written by AI."
    )


st.markdown('<div class="main-title">AI Tweet Detection</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="subtitle">
    This tool reviews tweet text and estimates whether its writing pattern looks closer to human-written or AI-generated text.
    It is meant for quick testing, model comparison, and project demonstration. The result is a model prediction, not a final judgment.
    </div>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs(["Try the Model", "Model Evaluation", "Project Notes"])


with tab1:
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Single Tweet Prediction")

        example_choice = st.selectbox(
            "Try an example or write your own",
            [
                "Write my own tweet",
                "Example 1: News-style update",
                "Example 2: Casual personal tweet",
                "Example 3: Public claim"
            ]
        )

        examples = {
            "Write my own tweet": "",
            "Example 1: News-style update": "Breaking update: officials announced new safety measures following concerns raised by residents online.",
            "Example 2: Casual personal tweet": "I just had the longest day ever, but this food made everything better.",
            "Example 3: Public claim": "New reports say the policy will affect thousands of voters before the next election."
        }

        tweet_text = st.text_area(
            "Enter a tweet",
            value=examples[example_choice],
            height=170,
            placeholder="Paste or type a tweet here..."
        )

        predict_button = st.button("Predict", type="primary")

    with right_col:
        st.subheader("Current model")
        st.markdown(
            f"""
            <div class="section-card">
            <b>{model_choice}</b><br>
            <span class="small-note">
            BERT reads more sentence context, while Random Forest and Logistic Regression use word and phrase patterns.
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Reminder")
        st.markdown(
            """
            <div class="section-card">
            <span class="small-note">
            Short or generic text may produce uncertain results because there is not enough writing signal.
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if predict_button:
        if tweet_text.strip() == "":
            st.warning("Please enter a tweet first.")
        else:
            prediction, probability_ai, confidence = get_prediction(
                model_choice,
                tweet_text,
                lightweight_models
            )

            display_label = final_display_label(prediction, confidence)
            card_class = "uncertain-card" if confidence < UNCERTAIN_THRESHOLD else "result-card"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="result-label">Prediction: {display_label}</div>
                    <div><b>Selected model:</b> {model_choice}</div>
                    <div><b>Confidence:</b> {confidence * 100:.1f}%</div>
                    <div><b>AI probability:</b> {probability_ai * 100:.1f}%</div>
                    <p class="small-note">{confidence_note(prediction, confidence)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

            st.subheader("Model agreement check")

            agreement_rows = []

            for name in ["BERT", "Random Forest", "Logistic Regression"]:
                pred, prob_ai, conf = get_prediction(name, tweet_text, lightweight_models)
                agreement_rows.append({
                    "Model": name,
                    "Prediction": final_display_label(pred, conf),
                    "AI probability": round(prob_ai * 100, 1),
                    "Confidence": round(conf * 100, 1)
                })

            agreement_df = pd.DataFrame(agreement_rows)
            st.dataframe(agreement_df, use_container_width=True)

            if agreement_df["Prediction"].nunique() > 1:
                st.warning(
                    "The models do not fully agree, so this example should be treated carefully."
                )

    st.divider()

    st.subheader("Batch Prediction")
    st.write("Upload a CSV file with a column named `text` to predict multiple tweets at once.")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        if "text" not in batch_df.columns:
            st.error("The CSV must contain a column named `text`.")
        else:
            text_values = batch_df["text"].fillna("").astype(str)

            predictions = []
            probabilities = []
            confidences = []

            for text in text_values:
                pred, prob_ai, conf = get_prediction(model_choice, text, lightweight_models)
                predictions.append(pred)
                probabilities.append(prob_ai)
                confidences.append(conf)

            batch_df["raw_prediction"] = predictions
            batch_df["prediction_label"] = [
                final_display_label(pred, conf)
                for pred, conf in zip(predictions, confidences)
            ]
            batch_df["ai_probability"] = [round(prob, 4) for prob in probabilities]
            batch_df["confidence"] = [round(conf, 4) for conf in confidences]

            st.success("Batch prediction complete.")
            st.dataframe(batch_df.head(20), use_container_width=True)

            st.download_button(
                "Download predictions",
                batch_df.to_csv(index=False).encode("utf-8"),
                "ai_tweet_predictions.csv",
                "text/csv"
            )


with tab2:
    st.subheader("Model Evaluation Results")

    display_results = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_results[col] = (display_results[col] * 100).round(2)

    best_model_row = display_results.sort_values("F1-score", ascending=False).iloc[0]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Best model", best_model_row["Model"])
    metric_cols[1].metric("Best accuracy", f"{best_model_row['Accuracy']:.2f}%")
    metric_cols[2].metric("Best recall", f"{best_model_row['Recall']:.2f}%")
    metric_cols[3].metric("Best F1-score", f"{best_model_row['F1-score']:.2f}%")

    st.dataframe(display_results, use_container_width=True)

    st.write(
        "BERT performs best overall because it captures more context from the text. "
        "Random Forest and Logistic Regression are lighter models, so they are easier to run in this app."
    )

    st.subheader("Confusion Matrix")

    matrix_choice = st.selectbox(
        "Choose a model",
        ["Random Forest", "Logistic Regression"],
        key="matrix_choice"
    )

    cm = pd.DataFrame(
        confusion_matrices[matrix_choice],
        index=["Actual Human", "Actual AI-generated"],
        columns=["Predicted Human", "Predicted AI-generated"]
    )

    st.dataframe(cm, use_container_width=True)


with tab3:
    st.subheader("Project Overview")

    st.write(
        "The app gives users a simple way to test the model without opening a notebook. "
        "A user can paste a tweet, choose a model, and receive a prediction with a confidence score."
    )

    st.subheader("Training Data Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total tweets", f"{len(df):,}")

    with col2:
        st.metric("Human-written", f"{(df['label'] == 0).sum():,}")

    with col3:
        st.metric("AI-generated", f"{(df['label'] == 1).sum():,}")

    st.subheader("Why the app is useful")

    st.write(
        "The app makes the model easier to understand because users can interact with it directly. "
        "It also shows model results and limitations in one place, which makes the project easier to explain during a demo."
    )

    st.subheader("Limitations and next steps")

    st.write(
        "The model can make mistakes, especially when human writing sounds polished or when AI-generated text sounds casual. "
        "A strong next step would be testing the model on more diverse tweets and improving how the app explains uncertain predictions."
    )
