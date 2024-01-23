import json
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Features, Value
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def read_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            json_data = json.loads(line)
            data.append(json_data)
    return data


def get_pandas_dfs(train_path, val_path, train_sample=None, val_sample=None):
    train_data = read_jsonl(train_path)
    val_data = read_jsonl(val_path)

    train_df = pd.DataFrame(train_data).loc[:, ["text", "label"]]
    train_df["label"] = train_df["label"].map(lambda x: "LLM" if x==1 else "human")
    val_df = pd.DataFrame(val_data).loc[:, ["text", "label"]]
    val_df["label"] = val_df["label"].map(lambda x: "LLM" if x==1 else "human")
    
    if train_sample:
        train_df = train_df.sample(n=train_sample, random_state=42).reset_index(drop=True)
    if val_sample:
        val_df = val_df.sample(n=val_sample, random_state=42).reset_index(drop=True)
    
    return train_df, val_df


def prepare_datasets(train_df, val_df):
    class_names = ["human", "LLM"]
    features = Features({'text': Value('string'), 'label': ClassLabel(num_classes=2, names=class_names)})

    train_ds = Dataset.from_pandas(train_df, features=features)
    val_ds = Dataset.from_pandas(val_df, features=features)
    
    return train_ds, val_ds


def chunk_text(text, tokenizer, max_token_count=512):
    min_token_count = max_token_count // 8
    tokens = tokenizer.encode(text, add_special_tokens=True)
    chunks = [tokens[i:i+max_token_count] for i in range(0, len(tokens), max_token_count)]
    if len(chunks) > 1:
        chunks = list(filter(lambda x: len(x) >= min_token_count, chunks))

    chunked_text = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    return chunked_text


def extract_hidden_states(batch, tokenizer, model, device):
    inputs = tokenizer(batch["text"], padding=True, 
                       truncation=True, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
    mean_embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu()
    max_embedding, _ = torch.max(outputs.last_hidden_state, dim=1)
    max_embedding = max_embedding.cpu()
    embeddings = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)
    return {"embeddings": embeddings}


def compute_metrics(eval_preds):
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics_dict
    


    
