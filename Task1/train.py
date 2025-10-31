# train.py

import json
import torch
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.optim import AdamW  # FIX: AdamW is imported from torch.optim
from seqeval.metrics import classification_report
from tqdm import tqdm  # Use standard tqdm
import numpy as np
import sys

# --- 0. Configuration and Mappings ---

# Default values, can be overridden by command-line arguments
LABEL_LIST = ["O", "B-MOUNTAIN", "I-MOUNTAIN", "[CLS]", "[SEP]"]
label_map = {label: i for i, label in enumerate(LABEL_LIST)}
id_to_label = {i: label for label, i in label_map.items()}

# --- 1. Dataset Class (NERDataset) ---

class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_map, max_len):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        self.data = self._load_data(data_path)

        if not self.data:
            print(f"Critical Error: Dataset at path {data_path} is empty or not loaded.")
            raise ValueError(f"Failed to load data from {data_path}")

    def _load_data(self, data_path):
        if not os.path.exists(data_path):
            print(f"Warning: File not found at path: {data_path}")
            return []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} records from {data_path}")
            return data
        except Exception as e:
            print(f"Error reading JSON file {data_path}: {e}")
            return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        tokens = sentence["tokens"]
        labels = sentence["labels"]

        tokenized_input = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = tokenized_input["input_ids"].squeeze(0)
        attention_mask = tokenized_input["attention_mask"].squeeze(0)

        # Label Alignment
        word_ids = tokenized_input.word_ids()
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a new word
                label_ids.append(self.label_map.get(labels[word_idx], -100))
            else:
                # Subsequent subword tokens get -100
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels_tensor = torch.tensor(label_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor
        }

# --- 2. Evaluation Function (evaluate_model) ---

def evaluate_model(model, dataloader, device, id_to_label, phase="Evaluation"):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc=phase):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            labels = labels.cpu().numpy()

            # Collect true tags and predicted tags, ignoring -100
            for i in range(labels.shape[0]):
                pred_tags, true_tags = [], []
                for j in range(labels.shape[1]):
                    if labels[i, j] != -100:
                        pred_tags.append(id_to_label[preds[i, j]])
                        true_tags.append(id_to_label[labels[i, j]])
                if true_tags:
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)

    print(f"\n--- {phase} ---")
    if all_labels:
        print(classification_report(all_labels, all_preds, digits=4))
    else:
        print("No data to evaluate.")
    return all_labels, all_preds

# --- 3. Main Training Function ---

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # Initialize tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    
    model = BertForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_LIST),
        id2label=id_to_label,
        label2id=label_map
    ).to(DEVICE)

    # Freeze layers
    FREEZE_LAYERS = args.freeze_layers
    if FREEZE_LAYERS > 0:
        for name, param in model.bert.named_parameters():
            if "encoder.layer." in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < FREEZE_LAYERS:
                    param.requires_grad = False
        print(f"Froze {FREEZE_LAYERS} BERT layers.")
    else:
        print("All BERT layers are unfrozen.")

    # --- Data ---
    try:
        train_dataset = NERDataset(args.train_path, tokenizer, label_map, args.max_len)
        test_dataset = NERDataset(args.test_path, tokenizer, label_map, args.max_len)
    except ValueError as e:
        print(f"Error loading data: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Prepare optimizer (only for parameters that require gradients)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    print("--- Starting training ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Loss = {avg_loss:.4f}")
        
        # Evaluate at the end of each epoch
        evaluate_model(model, test_loader, DEVICE, id_to_label, f"Evaluation Epoch {epoch+1}")

    print("--- Training complete ---")

    # --- Save model ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a NER model for mountain recognition")
    
    # --- Path Arguments ---
    # Use 'default=' to set the path.
    # Use r"..." (raw string) for Windows paths.
    # Removed 'required=True' since 'default' is provided.
    
    parser.add_argument("--train_path", 
                        type=str, 
                        default=r"C:\Users\User\Desktop\Task1\Data\Train_data.json",
                        help="Path to the training data JSON file")
    
    parser.add_argument("--test_path", 
                        type=str, 
                        default=r"C:\Users\User\Desktop\Task1\Data\Test_data.json",
                        help="Path to the test data JSON file")
    

    parser.add_argument("--output_dir", type=str, default="mountain_ner_model", help="Directory to save the trained model")
    
    # --- Model & Training Hyperparameters ---
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", help="Base BERT model name")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--freeze_layers", type=int, default=8, help="Number of BERT layers to freeze")

    args = parser.parse_args()
    main(args)