# inference.py

import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# --- 0. Configuration and Label Mapping ---

LABEL_LIST = ["O", "B-MOUNTAIN", "I-MOUNTAIN", "[CLS]", "[SEP]"]
label_map = {label: i for i, label in enumerate(LABEL_LIST)}
id_to_label = {i: label for label, i in label_map.items()}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Inference Function ---

def predict_mountain_names(text: str, model, tokenizer):
    model.eval()
    tokenized = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokenized)

    preds = torch.argmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"].squeeze(0).cpu().numpy())

    current_mountain_tokens = []
    results = []

    for token, pred_id in zip(tokens, preds):
        label = id_to_label.get(pred_id, "O")

        if token in tokenizer.all_special_tokens:
            continue

        if label.startswith("B-MOUNTAIN"):
            if current_mountain_tokens:
                results.append(" ".join(current_mountain_tokens))
            current_mountain_tokens = [token]
        elif label.startswith("I-MOUNTAIN"):
            current_mountain_tokens.append(token)
        else:
            if current_mountain_tokens:
                results.append(" ".join(current_mountain_tokens))
                current_mountain_tokens = []

    if current_mountain_tokens:
        results.append(" ".join(current_mountain_tokens))

    cleaned_results = [
        r.replace(" ##", "")
        for r in results if r.strip()
    ]

    return list(set(cleaned_results))

# --- 2. Main Execution ---

if __name__ == "__main__":
    MODEL_PATH = "mountain_ner_model"  # Path to your trained model
    TEXT_TO_PREDICT = (
        "Mount Everest and K2 are the highest mountains in the world. "
        "Mount Elbrus and Aconcagua are also very tall peaks."
    )

    print(f"🔹 Loading model from: {MODEL_PATH}")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = BertForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit()

    print(f"\n📝 Input text: {TEXT_TO_PREDICT}")
    mountains = predict_mountain_names(TEXT_TO_PREDICT, model, tokenizer)
    print(f"\n🏔️ Detected mountain names: {mountains}")
