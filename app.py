from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator
import streamlit as st
import sacrebleu
import requests
import zipfile
import os

DATA_URL = "https://github.com/halleyIsStressed/Translation_Method_Comparitor/releases/download/v1.0/data.zip"
DATA_DIR = "./data" 
NMT_MODEL = os.path.join(DATA_DIR, "fine_tuned_marian")

# -----------------------------
# Setup / Download data
# -----------------------------
@st.cache_resource
def setup_data():
    if not os.path.exists(DATA_DIR):
        r = requests.get(DATA_URL, stream=True)
        zip_path = "data.zip"
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
    return DATA_DIR

setup_data()

# -----------------------------
# Load NMT model
# -----------------------------
@st.cache_resource
def load_nmt_model(model_dir=NMT_MODEL):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    return tokenizer, model

nmtTokenizer, nmtModel = load_nmt_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Practical Comparison: NMT vs. Google Translate")

input_sentence = st.text_input("Please enter an English sentence:")
reference = st.text_input("Enter the Reference Translation:")

translate_btn = st.button("Translate")

if translate_btn and input_sentence:
    # --- NMT Translation ---
    nmt_inputs = nmtTokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = nmtModel.generate(**nmt_inputs)
    nmt_output = nmtTokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # --- Google Translate ---
    google_output = GoogleTranslator(source="en", target="zh-CN").translate(input_sentence)
    
    # --- Display Results ---
    st.markdown("### Translation Results")
    st.markdown(f"**NMT Output:** {nmt_output}")
    st.markdown(f"**Google Output:** {google_output}")
    
    # --- BLEU Scores if reference provided ---
    if reference:
        nmt_bleu = sacrebleu.sentence_bleu(nmt_output, [reference], tokenize='zh') 
        google_bleu = sacrebleu.sentence_bleu(google_output, [reference], tokenize='zh')
        
        st.markdown("### BLEU Scores")
        st.markdown(f"- **NMT BLEU:** {nmt_bleu.score:.2f}")
        st.markdown(f"- **Google BLEU:** {google_bleu.score:.2f}")
    else:
        st.markdown("No reference provided, BLEU Score calculation skipped.")
