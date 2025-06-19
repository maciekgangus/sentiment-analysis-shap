# utils/preprocessing.py
import re
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def combine_text(row):
    content = row['content'] if pd.notna(row['content']) else ''
    hashtags = row['hashtags'] if pd.notna(row['hashtags']) else ''
    mentions = row['mentions'] if pd.notna(row['mentions']) else ''
    return f"{content} {hashtags} {mentions}"

def clean_tweet_for_vader(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)  # usuwa linki
    text = re.sub(r"[@#]", "", text)  # usuwa tylko znaki @ i #, nie usuwa treści
    text = re.sub(r"[\"\']", "", text)  # usuwa tylko znaki " i '
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_sentiment(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def sentiment_to_number(label: str) -> int:
    mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
    return mapping.get(label, -1)

def preprocess_tweet_bert(text: str) -> str:
    text = re.sub(r"http\S+", "http", text)
    text = re.sub(r"pic\.twitter\.com/\S+", "<IMG>", text)
    text = re.sub(r"@\S+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_simple_tokenize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "<URL>", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)  # usuwa znaki specjalne
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_vader_analysis(df: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df["vader_clean"] = df["full_text"].apply(clean_tweet_for_vader)
    df["vader_scores"] = df["vader_clean"].apply(analyzer.polarity_scores)
    df = pd.concat([df, df["vader_scores"].apply(pd.Series)], axis=1)
    df["vader_sentiment_label"] = df["compound"].apply(classify_sentiment)
    df["vader_numeric"] = df["vader_sentiment_label"].map(sentiment_to_number)
    return df[["vader_clean", "vader_sentiment_label", "vader_numeric"]]

def run_bert_analysis(df: pd.DataFrame,
                      model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
                      device="mps") -> pd.DataFrame:
    df = df.copy()
    df["bert_clean"] = df["full_text"].apply(preprocess_tweet_bert)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    texts = df["bert_clean"].tolist()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tokenizer(batch, padding=True,
                            truncation=True,
                            return_tensors="pt",
                            max_length=128).to(device)
            out = model(**enc)
            probs = F.softmax(out.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    df["bert_sentiment_class"] = all_preds
    df["bert_sentiment_label"] = df["bert_sentiment_class"].map({0: "negative", 1: "neutral", 2: "positive"})
    df["bert_numeric"] = df["bert_sentiment_label"].map(sentiment_to_number)
    df["bert_probs"] = all_probs
    return df[["bert_clean", "bert_sentiment_label", "bert_numeric", "bert_probs"]]