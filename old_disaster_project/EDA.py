%pip install kagglehub
%pip install wordcloud

# Install packages if needed:
# !pip install kagglehub pandas matplotlib scikit-learn wordcloud

import html
import re
import warnings
from collections import Counter
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud

warnings.filterwarnings("ignore")


# ============================================================
# 1. DOWNLOAD AND LOAD THE DATASET
# ============================================================

dataset_path = Path(
    kagglehub.dataset_download("vstepanenko/disaster-tweets")
)

print("Dataset path:", dataset_path)

# Locate all CSV files inside the downloaded directory
csv_files = list(dataset_path.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files were found in the dataset folder.")

print("\nCSV files found:")
for file in csv_files:
    print("-", file.name)


def select_dataset_file(files):
    """
    Select the CSV most likely to contain tweet text and class labels.
    """
    text_candidates = {"text", "tweet", "tweet_text", "content"}
    label_candidates = {"target", "label", "class", "disaster", "is_disaster"}

    best_file = None
    best_score = -1

    for file in files:
        try:
            sample = pd.read_csv(file, nrows=5)
            columns = {column.lower().strip() for column in sample.columns}

            score = 0
            score += 2 * len(columns.intersection(text_candidates))
            score += 2 * len(columns.intersection(label_candidates))

            if score > best_score:
                best_score = score
                best_file = file
        except Exception:
            continue

    return best_file


data_file = select_dataset_file(csv_files)

if data_file is None:
    raise ValueError("A readable dataset CSV could not be identified.")

df = pd.read_csv(data_file)

print("\nSelected file:", data_file.name)
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

display(df.head())


# ============================================================
# 2. IDENTIFY TEXT AND CLASS COLUMNS
# ============================================================

def find_column(dataframe, possible_names):
    """
    Find a column using a list of possible column names.
    """
    lower_to_original = {
        column.lower().strip(): column for column in dataframe.columns
    }

    for name in possible_names:
        if name in lower_to_original:
            return lower_to_original[name]

    return None


TEXT_COL = find_column(
    df,
    ["text", "tweet", "tweet_text", "content"]
)

LABEL_COL = find_column(
    df,
    ["target", "label", "class", "disaster", "is_disaster"]
)

KEYWORD_COL = find_column(df, ["keyword"])
LOCATION_COL = find_column(df, ["location"])

if TEXT_COL is None:
    raise ValueError(
        "The tweet text column could not be found. "
        "Set TEXT_COL manually using one of the printed column names."
    )

if LABEL_COL is None:
    raise ValueError(
        "The class label column could not be found. "
        "Set LABEL_COL manually using one of the printed column names."
    )

print("\nText column:", TEXT_COL)
print("Class column:", LABEL_COL)

if KEYWORD_COL:
    print("Keyword column:", KEYWORD_COL)

if LOCATION_COL:
    print("Location column:", LOCATION_COL)


# Remove rows without text or labels
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

# Create readable class names
normalized_labels = (
    df[LABEL_COL]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

df["class_name"] = normalized_labels.replace(
    {
        "0": "Not disaster (0)",
        "1": "Disaster (1)",
    }
)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\nDataset information:")
df.info()

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate tweets:", df[TEXT_COL].duplicated().sum())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\nClass counts:")
class_summary = (
    df["class_name"]
    .value_counts()
    .rename_axis("class")
    .reset_index(name="count")
)

class_summary["percentage"] = (
    class_summary["count"] / len(df) * 100
).round(2)

display(class_summary)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

class_counts = df["class_name"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values)

for position, count in enumerate(class_counts.values):
    plt.text(
        position,
        count,
        f"{count:,}",
        ha="center",
        va="bottom"
    )

plt.title("Number of Tweets in Each Class")
plt.xlabel("Class")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 5. CLEAN AND TOKENIZE THE TWEETS
# ============================================================

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_tweet(text):
    """
    Basic cleaning for exploratory text analysis.

    - Converts HTML entities
    - Removes URLs
    - Removes usernames
    - Keeps hashtag words but removes the # symbol
    - Removes punctuation and numbers
    - Converts text to lowercase
    """
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()


def tokenize_tweet(text):
    """
    Convert cleaned text into tokens and remove common stop words.
    """
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())

    return [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]


df["clean_text"] = df[TEXT_COL].apply(clean_tweet)
df["tokens"] = df["clean_text"].apply(tokenize_tweet)
df["analysis_text"] = df["tokens"].apply(" ".join)

print("\nExample tokenized tweets:")

for row_number in range(min(5, len(df))):
    print(f"\nOriginal: {df.iloc[row_number][TEXT_COL]}")
    print(f"Tokens:   {df.iloc[row_number]['tokens']}")


# ============================================================
# 6. TWEET AND TOKEN LENGTHS
# ============================================================

df["character_count"] = df[TEXT_COL].astype(str).str.len()
df["word_count"] = df["clean_text"].str.split().str.len()
df["token_count"] = df["tokens"].str.len()

length_summary = (
    df.groupby("class_name")[
        ["character_count", "word_count", "token_count"]
    ]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)

print("\nTweet-length summary by class:")
display(length_summary)


# Token-count distribution
plt.figure(figsize=(9, 5))

for class_name, class_data in df.groupby("class_name"):
    plt.hist(
        class_data["token_count"],
        bins=30,
        alpha=0.6,
        label=class_name
    )

plt.title("Token Count Distribution by Class")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Tweets")
plt.legend()
plt.tight_layout()
plt.show()


# Average token count
average_length = (
    df.groupby("class_name")["token_count"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 5))
plt.bar(average_length.index, average_length.values)
plt.title("Average Number of Tokens by Class")
plt.xlabel("Class")
plt.ylabel("Average Token Count")
plt.tight_layout()
plt.show()


# ============================================================
# 7. MOST COMMON INDIVIDUAL TOKENS
# ============================================================

all_tokens = [
    token
    for token_list in df["tokens"]
    for token in token_list
]

top_tokens = pd.DataFrame(
    Counter(all_tokens).most_common(20),
    columns=["token", "count"]
)

print("\nMost common tokens:")
display(top_tokens)

plt.figure(figsize=(9, 6))
plt.barh(
    top_tokens["token"][::-1],
    top_tokens["count"][::-1]
)
plt.title("20 Most Common Tokens")
plt.xlabel("Frequency")
plt.ylabel("Token")
plt.tight_layout()
plt.show()


# ============================================================
# 8. WORD CLOUDS
# ============================================================

def plot_wordcloud(text, title):
    """
    Create one word cloud.
    """
    if not text.strip():
        print(f"No words available for: {title}")
        return

    cloud = WordCloud(
        width=1200,
        height=600,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Overall word cloud
plot_wordcloud(
    " ".join(df["analysis_text"]),
    "Most Common Words in All Disaster Tweets Data"
)

# One word cloud for each class
for class_name, class_data in df.groupby("class_name"):
    plot_wordcloud(
        " ".join(class_data["analysis_text"]),
        f"Most Common Words: {class_name}"
    )


# ============================================================
# 9. UNIGRAM, BIGRAM, AND TRIGRAM ANALYSIS
# ============================================================

def get_top_ngrams(text_series, n=1, top_n=15):
    """
    Return the most frequent n-grams.

    n=1: unigram
    n=2: bigram
    n=3: trigram
    """
    valid_text = text_series[text_series.str.strip() != ""]

    if valid_text.empty:
        return pd.DataFrame(columns=["ngram", "count"])

    vectorizer = CountVectorizer(
        ngram_range=(n, n),
        min_df=2
    )

    try:
        matrix = vectorizer.fit_transform(valid_text)
    except ValueError:
        return pd.DataFrame(columns=["ngram", "count"])

    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    results = pd.DataFrame(
        {
            "ngram": terms,
            "count": frequencies
        }
    )

    return (
        results
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_top_ngrams(results, title):
    """
    Plot an n-gram frequency table.
    """
    if results.empty:
        print(f"No n-grams available for: {title}")
        return

    plot_data = results.sort_values("count")

    plt.figure(figsize=(10, 6))
    plt.barh(plot_data["ngram"], plot_data["count"])
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("N-gram")
    plt.tight_layout()
    plt.show()


ngram_names = {
    1: "Unigrams",
    2: "Bigrams",
    3: "Trigrams"
}

# Overall n-grams
for n in [1, 2, 3]:
    results = get_top_ngrams(
        df["analysis_text"],
        n=n,
        top_n=15
    )

    print(f"\nTop {ngram_names[n].lower()}:")
    display(results)

    plot_top_ngrams(
        results,
        f"Top 15 {ngram_names[n]} in All Tweets"
    )


# N-grams separated by class
for class_name, class_data in df.groupby("class_name"):
    for n in [1, 2, 3]:
        results = get_top_ngrams(
            class_data["analysis_text"],
            n=n,
            top_n=15
        )

        plot_top_ngrams(
            results,
            f"Top 15 {ngram_names[n]}: {class_name}"
        )


# ============================================================
# 10. OPTIONAL: MOST COMMON DISASTER KEYWORDS
# ============================================================

if KEYWORD_COL is not None:
    keyword_counts = (
        df[KEYWORD_COL]
        .dropna()
        .astype(str)
        .str.replace("%20", " ", regex=False)
        .value_counts()
        .head(20)
        .sort_values()
    )

    print("\nMost common dataset keywords:")
    display(
        keyword_counts
        .sort_values(ascending=False)
        .rename_axis("keyword")
        .reset_index(name="count")
    )

    plt.figure(figsize=(10, 7))
    plt.barh(keyword_counts.index, keyword_counts.values)
    plt.title("20 Most Common Disaster Keywords")
    plt.xlabel("Number of Tweets")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. OPTIONAL: LOCATION COVERAGE
# ============================================================

if LOCATION_COL is not None:
    location_present = df[LOCATION_COL].notna().value_counts()

    location_present.index = location_present.index.map(
        {
            True: "Location provided",
            False: "Location missing"
        }
    )

    plt.figure(figsize=(8, 5))
    plt.bar(location_present.index, location_present.values)
    plt.title("Tweet Location Availability")
    plt.xlabel("Location Status")
    plt.ylabel("Number of Tweets")
    plt.tight_layout()
    plt.show()


print("\nEDA complete.")
