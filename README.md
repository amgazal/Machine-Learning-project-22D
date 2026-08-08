# AI Tweet Detection — Group 22D

A machine learning project that classifies tweets as **human-written** or **AI-generated**, built for AI4ALL. The project combines two public datasets into one benchmark, explores the data, and compares three modeling approaches — Logistic Regression, Random Forest, and a fine-tuned BERT — with a Streamlit app for interactive demos.

## Project pipeline

The repo is organized as a sequence of notebooks, each feeding the next:

1. **[`AI_Tweet_Detection_Dataset.ipynb`](AI_Tweet_Detection_Dataset.ipynb)** — Dataset selection and merge. Loads TweepFake and ElectAI from their public GitHub repos, standardizes both onto one binary label (`0` = human, `1` = AI-generated), cleans the text (drops missing/empty rows and duplicate tweets), and saves the result to `combined_ai_tweet_detection_dataset.csv`.
2. **[`EDA.ipynb`](EDA.ipynb)** — Exploratory data analysis on the merged dataset: missing-value/duplicate checks, class balance, source and generator-type breakdowns, tweet length distributions, and sample tweets.
3. **Modeling notebooks** — three approaches trained and evaluated on the same merged dataset:
   - [`modeling.ipynb`](modeling.ipynb) — Logistic Regression on TF-IDF features (unigram vs. bigram comparison)
   - [`Random_Forest_AI_Tweet_Detection_Model.ipynb`](Random_Forest_AI_Tweet_Detection_Model.ipynb) — Random Forest on TF-IDF features, including a per-source performance breakdown and feature-importance analysis
   - [`BERT_model.ipynb`](BERT_model.ipynb) — fine-tuned `bert-base-uncased` transformer model
4. **[`streamlit_app/`](streamlit_app)** — an interactive Streamlit app for trying the models on custom text.

## Dataset

| Source | Description | Rows (after cleaning) |
|---|---|---|
| [TweepFake](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251415) | General human-vs-bot tweet benchmark; bot tweets generated via Markov chains, RNN, LSTM, and GPT-2 | 25,553 |
| [ElectAI](https://arxiv.org/abs/2404.16116) | Election-claims authorship dataset; AI tweets generated via Falcon, LLaMA, and Mistral | 9,890 |
| **Combined** | | **35,443** (44.2% human / 55.8% AI-generated) |

The merged dataset is saved as `combined_ai_tweet_detection_dataset.csv` and is the shared input for every notebook and the Streamlit app.

## Model results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 0.824 | 0.833 | 0.856 | 0.844 |
| Random Forest | 0.824 | 0.808 | 0.898 | 0.850 |
| BERT (fine-tuned) | **0.921** | **0.922** | **0.938** | **0.930** |

BERT outperforms both classical models across every metric. However, breaking Random Forest's performance down by source dataset reveals a generalization gap — 98.1% accuracy on ElectAI vs. 76.3% on TweepFake — with the model's top feature by importance being the literal string `https`, followed by election-topic words like `county` and `voter fraud`. This suggests part of the models' accuracy comes from topic/URL artifacts correlated with the label (ElectAI is mostly AI-labeled) rather than purely from learning AI-vs-human writing style — a limitation worth investigating further across all three models before treating the results as a robust style detector.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel torch transformers
```

Run any notebook with Jupyter, or execute one end-to-end from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

## Running the Streamlit app

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run streamlit_app.py
```

See [`streamlit_app/README.md`](streamlit_app/README.md) for what the app does.

## Repo structure

```
.
├── AI_Tweet_Detection_Dataset.ipynb   # Dataset load, merge, clean
├── EDA.ipynb                          # Exploratory data analysis
├── modeling.ipynb                     # Logistic Regression baseline
├── Random_Forest_AI_Tweet_Detection_Model.ipynb
├── BERT_model.ipynb                   # Fine-tuned BERT model
├── combined_ai_tweet_detection_dataset.csv
├── streamlit_app/                     # Interactive demo app
└── old_disaster_project/              # Earlier, unrelated project (disaster tweet classification)
```

## License

MIT — see [LICENSE](LICENSE).
