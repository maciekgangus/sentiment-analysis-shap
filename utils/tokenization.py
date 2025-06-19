# utils/tokenization.py
import pickle
import torch
from transformers import PreTrainedTokenizer


def build_word2idx(tweets, pad_token='<pad>', unk_token='<unk>'):
    word2idx = {pad_token: 0, unk_token: 1}
    for tweet in tweets:
        for word in tweet.split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    return word2idx


def tokenize_and_pad(tweets, word2idx, pad_token='<pad>'):
    pad_idx = word2idx[pad_token]
    tokenized = [[word2idx.get(word, word2idx['<unk>']) for word in tweet.split()] for tweet in tweets]
    max_len = max(len(t) for t in tokenized)
    padded = [t + [pad_idx] * (max_len - len(t)) for t in tokenized]
    return torch.tensor(padded, dtype=torch.long)


def tokenize_and_save_stream(data, tokenizer: PreTrainedTokenizer, batch_size=32, file_path="tokenized_batches.pkl"):
    with open(file_path, "wb") as f:
        pass  # nadpisz
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        encoded_inputs = tokenizer(batch, padding=True, truncation=True, max_length=32, return_tensors="pt")
        batch_dict = {k: v.tolist() for k, v in encoded_inputs.items()}
        with open(file_path, "ab") as f:
            pickle.dump(batch_dict, f)


def load_tokenized_batches_stream(model, device, file_path="tokenized_batches.pkl"):
    model.eval()
    results = []
    with open(file_path, "rb") as f:
        while True:
            try:
                batch_dict = pickle.load(f)
                batch_tensors = {k: torch.tensor(v).to(device) for k, v in batch_dict.items()}
                with torch.no_grad():
                    outputs = model(**batch_tensors)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probs, dim=1)
                results.append((predicted_classes.cpu().numpy(), probs.cpu().numpy()))
            except EOFError:
                break
    return results
