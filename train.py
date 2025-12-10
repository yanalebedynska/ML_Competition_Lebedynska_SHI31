import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed


def tokenize_texts(tokenizer, texts, max_len):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len
    )


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.enc = encodings
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx]).long()
        return item


def train_model(train_path, test_path, output_dir, model_name="xlm-roberta-base", max_len=256, epochs=2):

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    # LOAD DATA
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train["new_label"] = df_train["new_label"].astype(int)

    print("Label distribution:")
    print(df_train["new_label"].value_counts())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_train["cleaned_message"].astype(str).tolist(),
        df_train["new_label"].values,
        test_size=0.15,
        stratify=df_train["new_label"],
        random_state=42
    )

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenize_texts(tokenizer, train_texts, max_len)
    val_enc = tokenize_texts(tokenizer, val_texts, max_len)
    test_enc = tokenize_texts(tokenizer, df_test["cleaned_message"].astype(str).tolist(), max_len)

    train_ds = TextDataset(train_enc, train_labels)
    val_ds = TextDataset(val_enc, val_labels)
    test_ds = TextDataset(test_enc)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    # MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)

    class_counts = np.bincount(df_train["new_label"])
    weights = class_counts.sum() / (2 * class_counts)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # TRAIN LOOP
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()

        running_loss = 0
        progress = tqdm(train_loader, desc="Training", leave=False)

        for batch in progress:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            progress.set_postfix({"loss": loss.item()})

        print(f"Train Loss: {running_loss / len(train_loader):.4f}")

        # VALIDATION
        model.eval()
        val_preds, val_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        acc = accuracy_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds)

        print(f"Validation Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # TEST PREDICT
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())

    # SAVE RESULTS
    os.makedirs(output_dir, exist_ok=True)
    submission = pd.DataFrame({
        "row ID": df_test["row ID"],
        "new_label": all_preds
    })
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print("Saved:", os.path.join(output_dir, "submission.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
