# utils/preprocessing.py
import re
from collections import defaultdict, Counter
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import shap
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_STOPWORDS = set("""
i me my myself we our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs
themselves what which who whom this that these those am is are was were be
been being have has had having do does did doing a an the and but if or because
as until while of at by for with about against between into through during before
after above below to from up down in out on off over under again further then
once here there when where why how all any both each few more most other some
such no nor not only own same so than too very s t can will just don should now
""".split())

def combine_text(row):
    content = row['content'] if pd.notna(row['content']) else ''
    hashtags = row['hashtags'] if pd.notna(row['hashtags']) else ''
    mentions = row['mentions'] if pd.notna(row['mentions']) else ''
    return f"{content} {hashtags} {mentions}"



def clean_tweet_for_vader(text: str) -> str:
    text = re.sub(r"pic\.twitter\.com/\S+", "<IMG>", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"[\"\']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = ' '.join([word for word in text.split() if word.lower() not in DEFAULT_STOPWORDS])
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
    text = re.sub(r"[@#]", "", text)
    text = re.sub(r"[\"\']", "", text)
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


