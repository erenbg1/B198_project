
# Fake News Detection using Transformer Models: A Comparative Study of Human vs AI Judgment

## 📄 Project Overview
This project investigates how Artificial Intelligence compares to human intuition in detecting fake news.  
A transformer-based NLP model (DistilBERT) will be trained on English news articles to classify them as *real* or *fake*, and its performance will be compared against a human participant’s manual judgments.

---

## 🎯 Objectives
1. Build a reproducible fake-news detection pipeline.  
2. Clean and normalize the Kaggle Fake/Real News dataset for modeling.  
3. Establish a TF-IDF + Logistic Regression baseline.  
4. Fine-tune DistilBERT and compare its accuracy to a human evaluator.  
5. Analyze misclassification patterns and discuss trust implications.

---

## 📚 Literature Overview (2020–2025)

| Year | Authors | Title | Source / Journal | Key Findings |
|------|----------|--------|------------------|---------------|
| 2020 | Shu, K. et al. | *A Survey on Natural Language Processing for Fake News Detection* | LREC | Provided a foundational taxonomy of fake news detection methods and datasets. Highlighted early shift from rule-based to ML approaches. |
| 2023 | Ahmed, R. & Chen, Y. | *Towards Fake News Detection: A Multivocal Literature Review* | MDPI Electronics | Reviewed advances from 2019–2023; noted the dominance of transformer-based architectures such as BERT and RoBERTa. |
| 2024 | Sharma, T. et al. | *Fake News Detection: State-of-the-Art Review and Advances* | PeerJ Computer Science | Benchmarked various deep learning methods and emphasized the need for explainability in fake news detection. |
| 2023 | Li, H. et al. | *Systematic Review of Machine Learning Algorithms and Datasets for Fake News Detection* | ResearchGate Preprint | Compared classical ML vs. transformer models and discussed dataset limitations such as bias and linguistic skew. |
| 2024 | Serrano, J. & Müller, K. | *Human Performance in Detecting Deepfakes: A Systematic Review* | ScienceDirect | Found that humans perform inconsistently across text vs. image misinformation, underscoring trust and intuition gaps. |
| 2025 | Zhang, P. et al. | *Human vs AI: A Comparative Benchmark for Misinformation Detection* | arXiv Preprint | Introduced the first controlled benchmark testing humans and AI on the same fake-news dataset, providing direct comparison data. |
---

## 📊 Dataset Collection & Preparation
**Source:** [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- Files used: `Fake.csv` (23,481 rows) and `True.csv` (21,417 rows)  
- Combined total: **44,898 records**

### **Cleaning & Normalization Steps**
1. **Merging and Labeling** → Added `label` column (`0 = Fake`, `1 = Real`).  
2. **Duplicate Removal** → 6,252 duplicates dropped.  
3. **Short Text Filter** → 144 rows under 50 chars removed.  
4. **Text Normalization** → URLs, special symbols, and extra whitespace removed.  
5. **Merged Columns** → Created `content` by concatenating `title` + `text`.  
6. **Final Shape:** `38,502 rows × 6 columns`

### **Label Distribution**
| Class | Label | Proportion |
|--------|--------|-------------|
| Real News | 1 | 55% |
| Fake News | 0 | 45% |

---

## 🧰 Tools & Libraries
- **Language:** Python 3.11  
- **Libraries:** Pandas, NumPy, scikit-learn, Matplotlib, Transformers, SHAP  
- **IDE:** Jupyter Notebook / VS Code  
- **Version Control:** GitHub

---

## 📎 Author
**Eren Burak Gökpınar**  
GISMA University of Applied Sciences  
Module: B198 End-to-End Project
