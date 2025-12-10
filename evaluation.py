import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "xlm-roberta-base"

def evaluate(train_path, test_path):
    print("Завантаження даних…")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print("Приклад тексту:", df_test["cleaned_message"].iloc[0][:100])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("Токенізація…")

    enc = tokenizer(
        df_test["cleaned_message"].astype(str).tolist(),
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    print("Інференс (прогноз)…")

    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    submission = pd.DataFrame({
        "row ID": df_test["row ID"],
        "new_label": preds
    })

    submission.to_csv("submission_eval.csv", index=False)
    print("Файл submission_eval.csv створено!")

if __name__ == "__main__":
    evaluate("data/train.csv", "data/to_answer.csv")
