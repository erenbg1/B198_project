# Fake News Detection using Machine Learning and Transformer Models

## Project Overview
This project focuses on detecting fake news articles using natural language processing techniques. Two different modelling approaches were implemented and compared on the same dataset:

- A traditional machine learning baseline using TF-IDF and Logistic Regression
- A transformer-based model using fine-tuned DistilBERT

The main objective of the project is to evaluate how a simple baseline model performs compared to a more advanced transformer-based model under similar conditions.

---

## Objectives
- Build an end-to-end fake news detection pipeline
- Preprocess and clean a large-scale news dataset
- Establish a baseline using TF-IDF and Logistic Regression
- Fine-tune a DistilBERT model for text classification
- Compare model performance using standard evaluation metrics

---

## Dataset
Source: Kaggle – Fake and Real News Dataset  
Link: [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]

Files used:
- Fake.csv
- True.csv

The dataset contains approximately 45,000 English news articles with binary labels (Fake / Real).
To provide richer contextual information, the title and main text of each article were merged into a single text field before training.

Preprocessing steps include:
- Removing duplicate entries
- Filtering very short or uninformative texts
- Basic text normalization

---

## Models and Methodology

### Baseline Model
- Text representation: TF-IDF
- Classifier: Logistic Regression

This approach is computationally efficient, easy to interpret, and fast to train. It serves as a strong baseline for comparison.

### Transformer-Based Model
- Model: DistilBERT (distilbert-base-uncased)
- Approach: Fine-tuning using Hugging Face Transformers
- Training epochs: 2
- Batch size: 8

DistilBERT was selected due to its balance between performance and computational efficiency, making it suitable for training on standard hardware.

---

## Results Summary
Both models were evaluated using accuracy, precision, recall, and F1-score.

- TF-IDF + Logistic Regression achieved an accuracy of approximately 98.56%.
- Fine-tuned DistilBERT achieved an accuracy of approximately 99.87%.

The DistilBERT model showed higher overall performance and fewer misclassifications, particularly in recall and F1-score.

---

## Figures and Visualizations
All figures generated during the evaluation process are stored in the `figures/` directory. These include:

- Class distribution of fake and real news articles
- Confusion matrix comparison between models
- Metric comparison bar charts
- Total misclassification comparison

These visualizations support the quantitative evaluation discussed in the report.

---

## Project Structure
```
├── README.md
├── requirements.txt
├── app.py
├── data
│   └── processed
│       └── cleaned_combined.csv
├── datasets
│   ├── Fake.csv
│   └── True.csv
├── figures
│   ├── class_distribution.png
│   ├── confusion_matrices_comparison.png
│   ├── metrics_comparison_bar.png
│   ├── model_comparison_metrics.csv
│   └── total_misclassifications.png
├── notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_distilbert_finetuning.ipynb
│   └── 04_evaluation_and_results.ipynb
└── trained_model
    ├── config.json
    ├── model.safetensors
    └── training_args.bin
```

---

## How to Run
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/erenbg1/B198_project.git
cd B198_project
pip install -r requirements.txt
```

(Optional, for large files)
```bash
git lfs install
git lfs pull
```

To run the demo application:
```bash
streamlit run app.py
```

---

## Tools and Libraries
- Python
- Pandas
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- Matplotlib
- Streamlit

---

## Author
Eren Burak Gökpınar  
GISMA University of Applied Sciences  
Module: B198 – End-to-End Project

---

## License
This project is shared for educational purposes.
