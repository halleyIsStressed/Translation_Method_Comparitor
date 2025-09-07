from transformers import MarianMTModel, MarianTokenizer
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from transformers import pipeline

import streamlit as st
import itertools
import sacrebleu
import requests
import zipfile
import joblib
import nltk
import math
import json
import re
import os

DATA_URL = "https://github.com/halleyIsStressed/Translation_Method_Comparitor/releases/download/v1.0/data.zip"
DATA_DIR = "./data" 
NMT_MODEL = os.path.join(DATA_DIR, "fine_tuned_marian")


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

input_sentence = ""

# Loading the NMT model
@st.cache_resource
def load_nmt_model(model_dir=NMT_MODEL):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    return tokenizer, model

nmtTokenizer, nmtModel = load_nmt_model()

st.title("Practical Comparison: RBMT vs. SMT vs. NMT vs. Google Translate")
input_sentence = st.text_input("Please enter an English sentence:")
reference = st.text_input("Enter the Reference Translation:")


if translate_btn and input_sentence:
    nmt_input = input_sentence
    google_input = input_sentence

    # Running NMT Translation
    inputs = nmtTokenizer(nmt_input, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = nmtModel.generate(**inputs)
    nmt_output = nmtTokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Running Google Translation
    google_output = GoogleTranslator(source="en", target="zh-CN").translate(google_input)
    
    # Display translation results 
    st.markdown("### Translation Results")
    
    st.markdown(f"**NMT Output:** {nmt_output}")
    st.markdown(f"**Google Output:** {google_output}")
    
    if reference:
        nmt_bleu = sacrebleu.sentence_bleu(nmt_output, [reference], tokenize='zh') 
        google_bleu = sacrebleu.sentence_bleu(google_output, [reference], tokenize='zh')
        
        st.markdown("### BLEU Scores")
        st.markdown(f"- **NMT BLEU:** {nmt_bleu.score:.2f}")
        st.markdown(f"- **Google BLEU:** {google_bleu.score:.2f}")
    else:
        st.markdown("No reference provided, BLEU Score calculation skipped.")
