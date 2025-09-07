import os
import zipfile
import requests
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator
import sacrebleu

# ------------------------------
# Configuration
# ------------------------------
DATA_URL = "https://github.com/halleyIsStressed/Translation_Method_Comparitor/releases/download/v1.0/data.zip"
DATA_DIR = "./data"
NMT_MODEL = os.path.join(DATA_DIR, "fine_tuned_marian")

# ------------------------------
# Download and unzip dataset/model if not present
# ------------------------------
def download_and_extract(url, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to, exist_ok=True)
        zip_path = os.path.join(extract_to, "data.zip")
        # Download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # Unzip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)

download_and_extract(DATA_URL, DATA_DIR)

# ------------------------------
# Load NMT model safely from local folder
# ------------------------------
@st.cache_resource
def load_nmt_model(model_dir=NMT_MODEL):
    tokenizer = MarianTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = MarianMTModel.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, model

nmtTokenizer, nmtModel = load_nmt_model()

# ------------------------------
# Streamlit Interface
# ------------------------------
st.title("Practical Comparison: MarianMT vs. Google Translate")

input_sentence = st.text_input("Please enter an English sentence:")
reference = st.text_input("Enter the Reference Translation (optional):")

if st.button("Translate") and input_sentence:
    # ------------------------------
    # NMT Translation
    # ------------------------------
    inputs = nmtTokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = nmtModel.generate(**inputs)
    nmt_output = nmtTokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # ------------------------------
    # Google Translation
    # ------------------------------
    google_output = GoogleTranslator(source="en", target="zh-CN").translate(input_sentence)

    # ------------------------------
    # Display translation results
    # ------------------------------
    st.markdown("### Translation Results")
    st.markdown(f"**NMT Output:** {nmt_output}")
    st.markdown(f"**Google Output:** {google_output}")

    # ------------------------------
    # BLEU Score Calculation
    # ------------------------------
    if reference:
        nmt_bleu = sacrebleu.sentence_bleu(nmt_output, [reference], tokenize='zh')
        google_bleu = sacrebleu.sentence_bleu(google_output, [reference], tokenize='zh')

        st.markdown("### BLEU Scores")
        st.markdown(f"- **NMT BLEU:** {nmt_bleu.score:.2f}")
        st.markdown(f"- **Google BLEU:** {google_bleu.score:.2f}")
    else:
        st.markdown("No reference provided, BLEU Score calculation skipped.")
