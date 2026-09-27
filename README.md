# Emotion Classification NLP

A deep learning project that reads a sentence and predicts the emotion behind it — sadness, joy, love, anger, fear, or surprise. Started out as a straightforward BiGRU classifier, but while testing it with my own sentences I noticed the model did well on the benchmark but kept getting confused on normal, real-world text. Most of this repo (and this README) is about tracking down why that was happening and actually fixing it, not just reporting a high accuracy number.

## What it does

* Takes a sentence and returns the predicted emotion + a confidence score + probabilities for all 6 classes
* Served through a FastAPI backend with a small web UI to try it live
* Swagger docs auto-generated at `/docs`

## The model pipeline

```text
Input Text
    ↓
Text Preprocessing (lowercase, strip punctuation/apostrophes)
    ↓
Tokenization + Padding
    ↓
Embedding (initialized with GloVe, fine-tuned during training)
    ↓
Bidirectional GRU (x2)
    ↓
Bahdanau Attention layer
    ↓
Dense + Softmax
    ↓
Emotion Prediction
```

## Why it looks like this (the actual story)

The dataset is [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) — about 16k short, informal, Twitter-style sentences labeled with one of six emotions. A plain BiGRU trained on this got **92% test accuracy**, which looked great, but it kept messing up on sentences that weren't written like tweets. Things like *"I really appreciate everything you have done for me"* would get classified as sadness instead of love.

Test accuracy alone hid this because the test set is pulled from the same narrow, informal distribution as the training data — so of course a model trained on tweets does fine on more tweets. To actually check generalization, I wrote a separate set of 24 hand-written, normal-sounding sentences (not from the dataset at all) and started measuring accuracy on that too. That's where the real problems showed up.

From there I ran through a few things to try and actually fix it, not just the model:

**1. Tried adding a self-attention layer.** My theory was the BiGRU only really uses its last hidden state to decide, which loses information. Added a Bahdanau-style attention layer on top so the model could weigh every word instead. Result: barely moved the needle, sometimes even slightly worse. Turned out the issue wasn't *how* the model reads the sentence.

**2. Tried GloVe pretrained embeddings.** The actual problem looked more like a representation issue — the embedding layer was learning word meanings from scratch on only ~16k examples, so it never really learned what words like "grateful" or "furious" mean in general. Swapped the embedding layer to start from GloVe (300d, trained on billions of words) instead of random init, fine-tuned during training. This one actually helped.

**3. Fixed the class imbalance.** `love` and `surprise` only had 1,304 and 572 examples in training (vs. 5,362 for `joy`), and the dataset's own definition of `love`/`surprise` was pretty narrow and informal. Wrote and collected ~200 additional, more natural examples for these two classes and added them to training. This is what fixed most of the "surprise" over-prediction problem (the model used to guess "surprise" way too often to compensate for its low representation).

**4. Ran everything 3 times with different fixed random seeds** (42, 123, 2024) before drawing any conclusions, because the first couple of times I compared models the "winner" kept changing just from random initialization luck, not anything real. Averaging across all 3 seeds gave a trustworthy answer.

### Final comparison (averaged across 3 random seeds)

| Model | Test Accuracy | Macro F1 | Real-world (OOD) accuracy |
|---|---|---|---|
| Baseline BiGRU | 92.00% | 0.875 | 65.3% |
| BiGRU + Attention | 91.88% | 0.872 | 63.9% |
| **BiGRU + Attention + GloVe** | **92.85%** | **0.883** | **68.1%** |

The attention-only version never consistently beat the plain baseline — on any single run it might look better or worse, but averaged out it wasn't a real improvement. GloVe was the only change that helped consistently across all 3 seeds, both on the standard test set and on the hand-written real-world sentences. That's the version deployed here.

Honestly, the biggest single improvement didn't come from a fancier architecture at all — it came from fixing the imbalanced, narrow training data. Worth remembering next time before reaching for a bigger model.

## Tech Stack

* Python, TensorFlow / Keras
* FastAPI + Pydantic + Uvicorn
* HTML / CSS / JS (vanilla, no framework)

## Project Structure

```text
emotion-classification-nlp/
│
├── artifacts/
│   ├── glove.6B.300d.txt        (not included in repo, see setup below)
│   └── models/
│       ├── BiGRU_Attention_GloVe_Model.keras
│       └── tokenizer.pkl
├── notebooks/
│   ├── emotion_prediction.ipynb
│   ├── train.csv                 (original dair-ai/emotion training split)
│   ├── train_augmented_v2.csv    (train.csv + extra love/surprise examples — what's actually used for training)
│   ├── val.csv
│   └── test.csv
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── main.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Ashish800935/emotion-classification-nlp.git
cd emotion-classification-nlp
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download GloVe embeddings (needed only if you want to retrain from the notebook)

The trained model + tokenizer are already committed in `artifacts/models/`, so you don't need this just to run the app. If you want to re-run the notebook from scratch though:

* Download `glove.6B.zip` from [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/)
* Unzip it, take `glove.6B.300d.txt` out
* Put it directly in `artifacts/` (it's too big for git, so it's gitignored)

### 5. Run the server

```bash
uvicorn main:app --reload
```

App: `http://127.0.0.1:8000/`
Swagger docs: `http://127.0.0.1:8000/docs`

## API

### `GET /health`
Returns whether the server and model are up.

### `POST /predict`

```json
{
  "text": "I am extremely happy today!"
}
```

Returns the predicted emotion, a confidence score, and the probability for all 6 classes.

## Notes / things I'd still improve

* The OOD test set is only 24 sentences — enough to catch obvious generalization problems, but too small to fully trust small percentage differences between models. A bigger, more diverse hand-labeled eval set would make future comparisons more reliable.
* `love` and `surprise` are still the weakest classes (see the per-class report in the notebook) even after augmentation — the dataset's definition of these emotions is narrower than how they show up in normal writing.
* Haven't tried a transformer-based encoder (like DistilBERT) yet — logged it as a possible next step, but wanted to properly understand and be able to explain attention mechanisms first before jumping to a pretrained transformer.
