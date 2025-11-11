# 🧠 Fake News Detection using Transformer Models  
### *A Comparative Study of Human vs AI Judgment*

---

## 📄 Project Overview
This project investigates how **Artificial Intelligence** compares to **human intuition** in detecting fake news.  
A transformer-based NLP model (**DistilBERT**) was fine-tuned on English news articles to classify them as *real* or *fake*, and its performance was evaluated against a human participant’s manual judgments.

---

## 🎯 Objectives
- Build a reproducible end-to-end fake news detection pipeline.  
- Clean and normalize the Kaggle *Fake/Real News* dataset.  
- Establish a **TF-IDF + Logistic Regression** baseline.  
- Fine-tune **DistilBERT** and compare its accuracy to human evaluation.  
- Analyze misclassification patterns and discuss implications for AI trust and interpretability.

---

## 📚 Verified Literature (2023–2025)

1. **Ramzan, A., Ali, R. H., Ali, N., & Khan, A. (2024).** *Enhancing Fake News Detection Using BERT: A Comparative Analysis of Logistic Regression, RFC, LSTM and BERT.* In 2024 International Conference on IT and Industrial Technologies (ICIT). IEEE. DOI: [10.1109/ICIT63607.2024.10859673](https://doi.org/10.1109/ICIT63607.2024.10859673)

2. **Kitanovski, M., & Mitrevski, P. (2023).** *DistilBERT and RoBERTa Models for Identification of Fake News.* 46th MIPRO ICT and Electronics Convention. IEEE. DOI: [10.23919/MIPRO57284.2023.10159740](https://doi.org/10.23919/MIPRO57284.2023.10159740)

3. **Saadi, A., Belhadef, H., Guessas, A., & Hafirassou, O. (2025).** *Enhancing Fake News Detection with Transformer Models and Summarization.* *Engineering, Technology & Applied Science Research, 15*(3), 23253–23259. DOI: [10.48084/etasr.10678](https://doi.org/10.48084/etasr.10678)

*Dataset citation:* Kaggle. *Fake and Real News Dataset.* Retrieved 2025. [Link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📊 Dataset Collection & Preparation

**Source:** Kaggle – *Fake and Real News Dataset*  
**Files used:** `Fake.csv` (23,481 rows), `True.csv` (21,417 rows)  
**Total combined:** 44,898 records  

### 🧹 Cleaning & Normalization Steps
| Step | Description |
|------|--------------|
| Merging & Labeling | Added label column (0 = Fake, 1 = Real). |
| Duplicate Removal | Dropped 6,252 duplicates. |
| Short Text Filter | Removed 144 rows under 50 chars. |
| Text Normalization | Removed URLs, special symbols, and extra whitespace. |
| Column Merge | Combined *title* + *text* into `content` column. |

📦 **Final Shape:** 38,502 rows × 6 columns  
🗂 **Cleaned Dataset:** `data/processed/cleaned_combined.csv`

**Label Distribution**
| Class | Label | Proportion |
|--------|--------|-------------|
| Real News | 1 | 55% |
| Fake News | 0 | 45% |

---

## ⚙️ Model Architecture & Workflow

### 1️⃣ Baseline Model
- **TF-IDF + Logistic Regression**
- Metrics: Accuracy = 0.89, Precision = 0.88, Recall = 0.87  
- Served as interpretability baseline.

### 2️⃣ Fine-Tuned Model
- **DistilBERT Base Uncased** (Hugging Face)
- Optimizer: AdamW, Learning Rate: 2e-5, Batch Size: 16  
- Training Epochs: 3  
- Final Accuracy: **0.942** on validation set  
- Metrics visualization: `figures/metrics_comparison_bar.png`, `figures/confusion_matrices_comparison.png`

### 3️⃣ Explainability
- **SHAP** used to identify influential tokens.
- Highlighted linguistic cues and emotional patterns driving misclassifications.

---

## 🧩 Project Structure

```
B198project/
│
├── app.py                           # Flask app for inference
├── Requirements.txt
├── README.md
│
├── data/
│   └── processed/
│       └── cleaned_combined.csv
│
├── datasets/
│   ├── Fake.csv
│   └── True.csv
│
├── figures/
│   ├── metrics_comparison_bar.png
│   ├── confusion_matrices_comparison.png
│   └── total_misclassifications.png
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_distilbert_finetuning.ipynb
│   └── 04_evaluation_and_results.ipynb
│
└── trained_distilbert_fake_news/
    ├── config.json
    ├── model.safetensors
    └── training_args.bin
```

---

## ⚠️ Note on Git LFS Files
This repository uses **Git Large File Storage (LFS)** for large model and dataset files.

Tracked via LFS:
- `data/processed/cleaned_combined.csv`
- `datasets/Fake.csv`
- `datasets/True.csv`
- `trained_distilbert_fake_news/model.safetensors`

If cloning the repo, run:

```bash
git lfs install
git lfs pull
```

---

## 🧰 Tools & Libraries
| Category | Libraries |
|-----------|------------|
| Core | Pandas, NumPy, scikit-learn |
| NLP | Transformers, Datasets, Tokenizers |
| ML | PyTorch, Accelerate, Safetensors |
| Visualization | Matplotlib |
| Explainability | SHAP |
| App Layer | Flask |
| Dev Tools | JupyterLab, Git, Git LFS |

---

## 🚀 Quick Setup & Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/erenbg1/B198_project.git
cd B198_project
```

### 2️⃣ Install dependencies
```bash
pip install -r Requirements.txt
```

### 3️⃣ Pull LFS files (if needed)
```bash
git lfs install
git lfs pull
```

### 4️⃣ Run app
```bash
python app.py
```

or open individual notebooks inside `/notebooks/`.

---

## 🧠 Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|------------|---------|-----------|
| TF-IDF + Logistic Regression | 0.89 | 0.88 | 0.87 | 0.88 |
| DistilBERT Fine-Tuned | **0.94** | **0.93** | **0.94** | **0.94** |

Misclassification analysis revealed higher confusion in **neutral-toned articles**, aligning with prior research on human cognitive bias in misinformation detection.

---

## 👤 Author
**Eren Burak Gökpınar**  
GISMA University of Applied Sciences  
**Module:** B198 End-to-End Project

---

## 🏁 License
This project is distributed for educational and research purposes under the MIT License.  
See the full license text in `LICENSE` if provided.
