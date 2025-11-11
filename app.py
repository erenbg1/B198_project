# app.py — Final (fixed static examples, no randomness)
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import os

# --- Page setup ---
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.abspath("notebooks/trained_distilbert_fake_news")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model.config.id2label = {0: "Fake", 1: "Real"}
    model.config.label2id = {"Fake": 0, "Real": 1}
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

# --- Load dataset ---
@st.cache_data
def load_dataset():
    dataset_path = os.path.abspath("data/processed/cleaned_combined.csv")
    df = pd.read_csv(dataset_path)
    return df

df = load_dataset()

# --- Title ---
st.title("🕵️ Fake News Detection App")
st.write("This app uses our fine-tuned **DistilBERT model** to classify news as real or fake.")
st.markdown("---")
st.markdown("<h4 style='color:#0A66C2'>📈 Model Accuracy: 99.87%</h4>", unsafe_allow_html=True)

# --- Static examples (2 real + 2 fake) ---
st.subheader("🧩 Example News Samples")

# pick stable rows (use fixed indexes)
real_examples = df[df["label"] == 1].iloc[[10, 25]]   # 2 real
fake_examples = df[df["label"] == 0].iloc[[5, 15]]    # 2 fake

examples_dict = {
    f"🟢 Real – {real_examples.iloc[0]['title'][:80]}...": real_examples.iloc[0]["content"],
    f"🟢 Real – {real_examples.iloc[1]['title'][:80]}...": real_examples.iloc[1]["content"],
    f"🔴 Fake – {fake_examples.iloc[0]['title'][:80]}...": fake_examples.iloc[0]["content"],
    f"🔴 Fake – {fake_examples.iloc[1]['title'][:80]}...": fake_examples.iloc[1]["content"],
}

# --- Display examples vertically ---
for title, text in examples_dict.items():
    if st.button(title, key=title):
        st.session_state["selected_text"] = text
        st.session_state["selected_title"] = title

# --- Text area ---
selected_title = st.session_state.get("selected_title", "")
text_input = st.text_area(
    "🗞️ Paste or type a news article:",
    value=st.session_state.get("selected_text", ""),
    height=200,
    placeholder="Enter a headline or paragraph..."
)

if selected_title:
    st.caption(f"Example selected: {selected_title}")

# --- Prediction ---
if st.button("🔍 Predict"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        st.subheader("Result:")
        if pred == 0:
            st.error(f"🔴 **Fake News Detected!**  \nConfidence: {confidence*100:.2f}%")
        else:
            st.success(f"🟢 **Real News Detected!**  \nConfidence: {confidence*100:.2f}%")

        st.progress(confidence)

# --- Footer ---
st.markdown("---")
st.caption("Built for the B198 Fake News Detection Project | Fine-tuned DistilBERT Model")
