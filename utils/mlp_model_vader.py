# utils/mlp_model.py
import torch
import torch.nn as nn

class MLPWithMeanPooling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dims, output_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        layers = []
        input_dim = embedding_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.6))
            input_dim = hdim

        layers.append(nn.Linear(input_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        mask = (x != self.embedding.padding_idx).unsqueeze(2)  # (batch, seq_len, 1)
        masked = embedded * mask
        pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # mean pooling
        return self.fc(pooled)


def train_mlp(model, train_loader, optimizer, criterion, device, l1_lambda=0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # L1 regularization
        if l1_lambda > 0:
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            loss += l1_lambda * l1_penalty

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total



def evaluate_mlp(model, data_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * x_batch.size(0)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels
