# Mountain Name Recognition (NER) with BERT

This repository contains a focused Named Entity Recognition (NER) solution that identifies **mountain names** in text using a fine‑tuned BERT model. Entities are labeled using standard BIO tags: `B-MOUNTAIN`, `I-MOUNTAIN`.

---



### 1. Install dependencies
Install all required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Train the model
Train the model from scratch using the default data paths:

```bash
python train.py
```

The trained model and tokenizer will be saved to `mountain_ner_model/`.

### 3. Run inference
Test the trained model on new sentences:

```bash
# Run interactive inference
python inference.py

# Or pass text directly
python inference.py --text "I want to climb Mount Everest and K2."
```

---

##  Solution — Deep Dive

### 1. Dataset — Synthetic Generation Approach
A clean, pre-labeled corpus specifically for mountain-only NER is not readily available. To solve this, we built a **synthetic dataset**:

- **Positive samples:** Collected a comprehensive list of mountain names (e.g., from Wikipedia) and generated diverse, natural sentences containing those names.
- **Negative samples:** Created sentences that include other geographic entities — **cities, islands, rivers, countries**, etc. — but **no mountains**.

This negative-sampling strategy trains the model to learn contextual cues that distinguish mountains from other entities, rather than memorizing a finite list of names.

### 2. Model architecture
We fine-tune **`bert-base-multilingual-cased`** for token-level classification.

Why this model:
- **Context-aware** transformers (BERT) capture surrounding text signals that define entity boundaries.
- **Multilingual** coverage supports mountain names from many languages and scripts.
- **Cased** model preserves capitalization, a helpful cue for proper nouns.

### 3. Training strategy
- Fine-tuned for a small number of epochs (commonly **3–5**) to avoid overfitting on the small, synthetic dataset.
- Evaluated using **Precision, Recall, and F1-score** on a validation split.
- Best checkpoints and tokenizer are saved automatically during training.

Short training is deliberate: the dataset is task-specific and small; longer training tends to overfit and degrade generalization.