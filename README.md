<div align="center">

# 🧠 AI Lecture Summarizer
### BiGRU + DistilBERT · Extractive Summarization · End-to-End Deep Learning

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_App-orange?style=for-the-badge)](https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel)
[![Open In Colab](https://img.shields.io/badge/Open_In-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1j7eD76UCqm51LafeweFfwSbL7SmHmb2T?usp=sharing)
[![Dataset](https://img.shields.io/badge/Google_Drive-Dataset-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1z0RDMLAN-x9iaMUAE80eTrsCTEQXS9Ac?usp=sharing)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Post-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](YOUR_LINKEDIN_POST_URL)

*Paste any lecture transcript → get the most important sentences as a clean bullet-point summary*

</div>

---

## 🎯 What This Project Does

This project builds a **complete extractive summarization system** from scratch — no pre-built summarizer, no GPT wrapper. Every sentence in a document is scored by a custom-trained neural network and the top-K most important ones are returned in reading order.

**Works on:**
- ✅ Normal English articles
- ✅ YouTube auto-captions (even without punctuation)
- ✅ Urdu · Punjabi · English mixed transcripts
- ✅ Any text over 30 words

> **Try it live → [huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel](https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel)**

---

## 📊 Results

| Metric | Score |
|:---|:---|
| 🎯 Training Accuracy | **89.53%** |
| ✅ Validation Accuracy | **87.48%** |
| 📝 Total Sentences Trained | **16,173** |
| 🎬 Training Videos | **52+ YouTube lectures** |
| 💾 Model Size | **3.44 MB** |
| 🌍 Languages | **Urdu · Punjabi · English** |
| ⚡ Training Hardware | **Google Colab T4 GPU** |

---

## 🖼️ Proof — Model Architecture & Training

<img width="1357" height="764" alt="image" src="https://github.com/user-attachments/assets/1c845671-6003-48bc-8a44-79f91689de88" />


<img width="1359" height="767" alt="image" src="https://github.com/user-attachments/assets/ffd41190-1469-4631-97cf-e48293e57002" />


---

## 🏗️ Full Pipeline

```
Raw YouTube Videos (52+ lectures)
        │
        ▼
YouTube Transcript API  ──►  Raw transcripts (Urdu/Punjabi/English)
        │
        ▼
Sentence Tokenization  ──►  16,173 individual sentences extracted
        │
        ▼
Label Generation  ──►  TF-IDF cosine similarity → binary important/not labels
        │
        ▼
Feature Extraction (per sentence)
  ├── DistilBERT CLS embedding  → 768-dim semantic vector
  └── TF-IDF vector             → 232-dim lexical vector
        │ concatenate → 1000-dim fused vector
        ▼
BiGRU + Attention Model
  ├── Input:  (batch, 1, 1000)
  ├── BiGRU:  256 units bidirectional
  ├── Self-Attention layer
  ├── GlobalAveragePooling1D
  ├── Dense(128, relu) + Dropout(0.3)
  └── Dense(1, sigmoid) → importance score
        │
        ▼
Top-K Sentence Selection  ──►  Clean bullet-point summary
```

---

## 🧱 Model Architecture Detail

| Layer | Output Shape | Params | Connected To |
|:---|:---|---:|:---|
| `fused_input` (InputLayer) | (None, 1, 1000) | 0 | — |
| `bi_gru` (Bidirectional) | (None, 1, 256) | 867,840 | fused_input |
| `self_attention` (Attention) | (None, 1, 256) | 0 | bi_gru |
| `avg_pool` (GlobalAveragePooling1D) | (None, 256) | 0 | self_attention |
| `fc1` (Dense, relu) | (None, 128) | 32,896 | avg_pool |
| `dropout` (Dropout 0.3) | (None, 128) | 0 | fc1 |
| `importance_score` (Dense, sigmoid) | (None, 1) | 129 | dropout |

**Total params: 900,865 (3.44 MB)**

---

## 📁 Project Structure

```
GRUDeepLeraningmodel/
├── app.py                      # Gradio UI + inference pipeline
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── summarizer_model.keras      # Trained BiGRU model (3.44 MB)
├── summarizer_vectorizer.pkl   # Fitted TF-IDF vectorizer
└── assets/
    ├── model_architecture.png  # Screenshot of model summary
    └── training_results.png    # Screenshot of training logs
```

---

## 🚀 Run Locally

```bash
git clone https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel
cd GRUDeepLeraningmodel
pip install -r requirements.txt
python app.py
```

---

## 🧰 Tech Stack

| Component | Technology |
|:---|:---|
| Sentence Embeddings | `DistilBERT-base-uncased` |
| Lexical Features | `TF-IDF (scikit-learn)` |
| Sequence Model | `Bidirectional GRU + Attention` |
| Framework | `Keras 3 · TensorFlow 2.15` |
| UI | `Gradio 6.14` |
| Hosting | `Hugging Face Spaces` |
| Training | `Google Colab T4 GPU` |

---
<img width="1360" height="2258" alt="screencapture-huggingface-co-spaces-Maliktg5-GRUDeepLeraningmodel-2026-05-24-17_06_03" src="https://github.com/user-attachments/assets/fc8cc8ac-1866-4579-beac-a1ca1dad465c" />


## ✍️ Author

**Muhammad Mudassar**  
Built end-to-end: data collection → labeling → training → deployment

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](YOUR_LINKEDIN_POST_URL)

[![HuggingFace](https://img.shields.io/badge/🤗-Maliktg5-yellow)](https://huggingface.co/Maliktg5)

[![Colab Notebook](https://img.shields.io/badge/Colab-Training_Notebook-F9AB00?style=flat&logo=googlecolab)](https://colab.research.google.com/drive/1j7eD76UCqm51LafeweFfwSbL7SmHmb2T?usp=sharing)

[![Dataset](https://img.shields.io/badge/Dataset-Google_Drive-4285F4?style=flat&logo=googledrive)](https://drive.google.com/drive/folders/1z0RDMLAN-x9iaMUAE80eTrsCTEQXS9Ac?usp=sharing)

