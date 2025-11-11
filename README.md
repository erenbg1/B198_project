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

1. **Ramzan, A., Ali, R. H., Ali, N., & Khan, A. (2024).** *Enhancing Fake News Detection Using BERT: A Comparative Analysis of Logistic Regression, RFC, LSTM and BERT.* In 2024 International Conference on IT and Industrial Technologies (ICIT). IEEE. DOI: [10.1109/ICIT63607.2024.10859673](https://doi.org/10.1109/ICIT63607.2024.10859673)  
2. **Kitanovski, M., & Mitrevski, P. (2023).** *DistilBERT and RoBERTa Models for Identification of Fake News.* 46th MIPRO ICT and Electronics Convention. IEEE. DOI: [10.23919/MIPRO57284.2023.10159740](https://doi.org/10.23919/MIPRO57284.2023.10159740)  
3. **Saadi, A., Belhadef, H., Guessas, A., & Hafirassou, O. (2025).** *Enhancing Fake News Detection with Transformer Models and Summarization.* *Engineering, Technology & Applied Science Research, 15*(3), 23253–23259. DOI: [10.48084/etasr.10678](https://doi.org/10.48084/etasr.10678)

*Dataset citation:* Kaggle. *Fake and Real News Dataset.* Retrieved 2025.

---

## 📊 Dataset Collection & Preparation

**Source:** Kaggle – *Fake and Real News Dataset*  
**Files used:** `Fake.csv` (23,481 rows), `True.csv` (21,417 rows)  
**Total combined:** 44,898 records  

### Columns
1. `title`  
2. `text`  
3. `subject`  
4. `date`  
5. `label`  
6. `content`  

The `content` column merges **title** and **text** to create richer contextual input for modeling.

Cleaning highlights:
- Removed duplicates
- Filtered very short texts (<50 characters)
- Normalized text (URLs, symbols, whitespace)

---

## ⚙️ Model Architecture & Workflow

### 1️⃣ Baseline Model
- **TF-IDF + Logistic Regression**
- Metrics (validated from notebooks & figures):  
  | Metric | TF-IDF + Logistic Regression | DistilBERT (Fine-Tuned) |
  |:--|:--:|:--:|
  | Accuracy | **0.9856** | **0.9987** |
  | Precision | **0.9818** | **0.9987** |
  | Recall | **0.9922** | **0.9987** |
  | F1-score | **0.9870** | **0.9987** |
- Total misclassifications: **111** (LogReg) vs **10** (DistilBERT).  
- Confusion matrices and metric bars are available under `figures/`.

### 2️⃣ Fine-Tuned Model
- **Model:** DistilBERT Base Uncased (Hugging Face)
- **Optimizer:** AdamW (Hugging Face Trainer)
- **Learning Rate:** 2e-5
- **Epochs:** 2
- **Batch Size:** 8 (train & eval)
- **Warmup Steps:** 100
- **Weight Decay:** 0.01
- **Evaluation Strategy:** Per epoch
- **Save Strategy:** Per epoch
- **Final Accuracy:** **0.9987** (validation set)
- **Metrics Visualization:** `figures/metrics_comparison_bar.png`, `figures/confusion_matrices_comparison.png`, `figures/total_misclassifications.png`

### 3️⃣ Explainability
**SHAP (SHapley Additive Explanations)** will be integrated in a future version to interpret token-level importance and explain model decisions.

---

## 🧩 Project Structure

```
├── README.md
├── Requirements.txt
├── app.py
├── data
│   └── processed
│       └── cleaned_combined.csv
├── datasets
│   ├── Fake.csv
│   └── True.csv
├── figures
│   ├── confusion_matrices_comparison.png
│   ├── metrics_comparison_bar.png
│   ├── model_comparison_metrics.csv
│   └── total_misclassifications.png
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_distilbert_finetuning.ipynb
│   └── 04_evaluation_and_results.ipynb
└── trained_distilbert_fake_news
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
| Explainability | SHAP (planned) |
| Dashboard | Streamlit |
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

### 4️⃣ Run the Streamlit dashboard
```bash
streamlit run app.py
```
The dashboard allows users to test news articles in real time and view prediction confidence.

---

## 🧠 Results Summary
The TF-IDF baseline achieved **98.56% accuracy**, while the fine-tuned **DistilBERT reached 99.87%** on the validation set.  
Removing very short texts (<50 characters) improved overall consistency and model focus.

---

## 🔮 Future Work
- Add SHAP explainability  
- Add multilingual dataset 
- Deploy as online verification tool  

---

## 👤 Author
**Eren Burak Gökpınar**  
GISMA University of Applied Sciences  
**Module:** B198 End-to-End Project  

---

## 🏁 License
This project is distributed for educational and research purposes under the MIT License.  
See the full license text in `LICENSE` if provided.
