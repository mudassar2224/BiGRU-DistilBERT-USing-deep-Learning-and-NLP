


```markdown
# 🧠 AI Lecture Summarizer — BiGRU + DistilBERT

> Extractive text summarization using DistilBERT + Bidirectional GRU,
> trained on 52+ multilingual YouTube lecture transcripts
> (Urdu, Punjabi, English) — built and deployed end-to-end from scratch.

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_App-orange)](https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j7eD76UCqm51LafeweFfwSbL7SmHmb2T?usp=sharing)
[![Dataset](https://img.shields.io/badge/Google_Drive-Dataset-blue)](https://drive.google.com/drive/folders/1z0RDMLAN-x9iaMUAE80eTrsCTEQXS9Ac?usp=sharing)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Post-0077B5)](YOUR_LINKEDIN_POST_URL)

---

## 🎯 What This Project Does

Paste any lecture transcript or article → get the most important sentences
as a clean bullet-point summary.

**Works on:**
- ✅ Normal English articles
- ✅ YouTube auto-captions (even without punctuation)
- ✅ Urdu, Punjabi, English mixed transcripts
- ✅ Any text over 30 words

**Try it live → [huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel](https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel)**

---

## 📊 Results

| Metric | Score |
|---|---|
| Training Accuracy | **89.53%** |
| Validation Accuracy | **87.48%** |
| Total Sentences Trained | **16,173** |
| Training Videos | 52+ YouTube lectures |
| Model Size | 3.44 MB |
| Languages | Urdu · Punjabi · English |
| Training Hardware | Google Colab T4 GPU |

---

## 🏗️ Full Pipeline

```
📥 YouTube Videos (52+ lectures)
         ↓
🎙️ OpenAI Whisper → JSON Transcripts
         ↓
✂️ NLTK Sentence Tokenizer
         ↓
🏷️ TF-IDF Pseudo-Labeling
   (Top 20% = Label 1, Rest = Label 0)
         ↓
🔀 Feature Fusion per Sentence:
   DistilBERT CLS Token  →  768 dims
   +
   TF-IDF Vector         →  232 dims
   =
   Fused Vector          → 1000 dims
         ↓
🧠 Bidirectional GRU (128 units × 2)
         ↓
🎯 Self-Attention Layer
         ↓
📊 GlobalAveragePooling
         ↓
💡 Dense(128, ReLU) → Dropout(0.3)
         ↓
📈 Dense(1, Sigmoid) → Importance Score
         ↓
📋 Top K sentences returned in reading order
```

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| Source | YouTube lectures (manually selected) |
| Videos | 52+ across multiple topics |
| Topics | OOP · Flutter · ML · Data Structures · CS basics |
| Languages | Urdu · Punjabi · English (code-switched) |
| Transcription | OpenAI Whisper |
| Labeling | Automatic — TF-IDF pseudo-labeling |
| Sentences | 16,173 labeled sentences |
| Human annotation | ❌ Not required |

📂 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1z0RDMLAN-x9iaMUAE80eTrsCTEQXS9Ac?usp=sharing)**

---

## 🛠️ Tech Stack

| Component | Tool / Library |
|---|---|
| Speech-to-Text | OpenAI Whisper |
| Semantic Embeddings | DistilBERT (`distilbert-base-uncased`) |
| Training Framework | TensorFlow / Keras |
| Inference Framework | PyTorch (HF Spaces compatible) |
| Sequence Model | Bidirectional GRU + Self-Attention |
| Lexical Features | TF-IDF (scikit-learn) |
| Sentence Tokenization | NLTK |
| Web App | Gradio |
| Deployment | HuggingFace Spaces |
| Training Hardware | Google Colab T4 GPU |

---

## 🚀 How To Use

### ▶️ Option 1 — Live App (No Setup)
**[👉 Click here to use the app instantly](https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel)**

### 📓 Option 2 — Run the Training Notebook
**[👉 Open in Google Colab](https://colab.research.google.com/drive/1j7eD76UCqm51LafeweFfwSbL7SmHmb2T?usp=sharing)**

Steps inside the notebook:
1. Mount Google Drive
2. Install dependencies
3. Load Whisper transcripts
4. Generate pseudo-labels
5. Build BERT + TF-IDF feature matrix
6. Train BiGRU model
7. Deploy Gradio app

### 💻 Option 3 — Run Locally

```bash
git clone https://github.com/mudassar2224/BiGRU-DistilBERT-USing-deep-Learning-and-NLP
cd BiGRU-DistilBERT-USing-deep-Learning-and-NLP
pip install torch transformers nltk scikit-learn gradio numpy huggingface_hub
python app.py
```

---

## 📁 Repository Structure

```
├── GRU_MODEL_TRAINED_ON_PERSONAL_DATASET.ipynb  ← Full training notebook
├── app.py                                        ← Gradio web app
├── requirements.txt                              ← Dependencies
└── README.md
```

> Model weights (`model_weights.npz`) and vectorizer
> (`summarizer_vectorizer.pkl`) are hosted on HuggingFace
> and downloaded automatically at runtime — no manual setup needed.

---

## 💡 Key Challenges Solved

**1. No labeled data** → Solved with TF-IDF pseudo-labeling. No human
annotation needed — the model learns to replicate and improve TF-IDF scoring.

**2. Unpunctuated transcripts** → YouTube auto-captions have no full stops.
Fixed by forcing 35-word chunk splits when sentence count is too low.

**3. Multilingual mixed text** → DistilBERT handles English tokens;
TF-IDF handles Urdu/Punjabi lexical patterns. The fusion of both works
better than either alone on code-switched text.

**4. TF + PyTorch conflict on HF Spaces** → Solved by exporting Keras
weights to NumPy and implementing the GRU forward pass in pure NumPy.
Zero TensorFlow needed at inference time.

---

## 🔗 All Links

| Resource | Link |
|---|---|
| 🤗 Live App | [HuggingFace Spaces](https://huggingface.co/spaces/Maliktg5/GRUDeepLeraningmodel) |
| 📓 Training Notebook | [Google Colab](https://colab.research.google.com/drive/1j7eD76UCqm51LafeweFfwSbL7SmHmb2T?usp=sharing) |
| 📂 Dataset | [Google Drive](https://drive.google.com/drive/folders/1z0RDMLAN-x9iaMUAE80eTrsCTEQXS9Ac?usp=sharing) |
| 💻 GitHub | [This Repository](https://github.com/mudassar2224/BiGRU-DistilBERT-USing-deep-Learning-and-NLP) |

---

## 👤 Author

Built solo by **Malik Taimoor Ghazanfar**

🤗 [HuggingFace Profile](https://huggingface.co/Maliktg5)  
💼 [LinkedIn](YOUR_LINKEDIN_POST_URL)  
💻 [GitHub](https://github.com/mudassar2224)

---

⭐ **If this project helped you or you find it interesting, please give it a star!**
```

---

## Two Things To Do Now

**1.** In the GitHub README, replace `YOUR_LINKEDIN_POST_URL` with your LinkedIn post URL after you post it.

**2.** Add these 4 files to your GitHub repo before sharing:
- `GRU_MODEL_TRAINED_ON_PERSONAL_DATASET.ipynb` ← you have this
- `app.py` ← copy from your HF Space files tab
- `requirements.txt` ← copy from your HF Space files tab
- `README.md` ← paste the above

