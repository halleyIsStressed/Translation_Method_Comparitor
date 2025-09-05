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
NLTK_DIR = "./nltk"
FILTERED_JSON = os.path.join(DATA_DIR, "filtered.json")
NMT_MODEL = os.path.join(DATA_DIR, "fine_tuned_marian")
LM_FILE = os.path.join(DATA_DIR, "chinese_lm.joblib")
PROB_FILE = os.path.join(DATA_DIR, "en_cn_probs.joblib")


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

nltk.data.path.append(NLTK_DIR)  # point NLTK to pre-downloaded folder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()

# Call setup_data first
data_dir = setup_data()
dict_path = os.path.join(data_dir, "filtered.json")

@st.cache_resource
def load_smt_resources(lm_path=LM_FILE, probs_path=PROB_FILE):
    # Ensure data folder exists (optional if already handled)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Memory-mapped loading for large objects
    lm_dict = joblib.load(lm_path, mmap_mode='r')
    en_cn_probs = joblib.load(probs_path, mmap_mode='r')

    return lm_dict, en_cn_probs 

# Load JSON -> Dictionary
@st.cache_resource
def load_dataset_dict(file_path=FILTERED_JSON):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset_dict = {}
    for entry in raw_data:
        eng = entry.get("word", "").strip().lower()
        trans_list = entry.get("translation", [])
        if isinstance(trans_list, list) and len(trans_list) > 0:
            dataset_dict[eng] = trans_list[0]
    return dataset_dict

dataset_dict = load_dataset_dict()

# -----------------------------
# Utility functions
# -----------------------------
ignore_words = ["to", "a", "an", "the"]  # 冠词/虚词忽略

contractions = {
    "it's":"it is",
    "i'm":"i am",
    "you're":"you are",
    "we're":"we are",
    "they're":"they are",
    "i've":"i have",
    "you've":"you have",
    "isn't":"is not",
    "aren't":"are not",
    "don't":"do not",
    "doesn't":"does not",
}

def expand_contractions(sentence):
    for k, v in contractions.items():
        sentence = re.sub(r"\b"+k+r"\b", v, sentence, flags=re.IGNORECASE)
    return sentence

def _strip_brackets(s: str) -> str:
    s = re.sub(r"【[^】]*】", "", s)
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"（[^）]*）", "", s)
    s = re.sub(r"\[[^\]]*\]", "", s)
    return s

def clean_translation_text(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    if not isinstance(text, str):
        return ""
    candidates = re.split(r"[;,；，、/]| or ", text)
    pos_pat = re.compile(
        r"^(?:n|v|vi|vt|adj|adv|prep|conj|pron|abbr|int|interj|art|aux|num|det|modal|phr|idiom)\.\s*",
        re.IGNORECASE
    )
    for cand in candidates:
        cand = pos_pat.sub("", cand.strip())
        cand = _strip_brackets(cand).strip()
        if re.search(r"[\u4e00-\u9fff]", cand):
            return cand
    return pos_pat.sub("", candidates[0].strip()) if candidates else ""

# -----------------------------
# Word Lookup
# -----------------------------
subjects = {"i":"我","you":"你","he":"他","she":"她","we":"我们","they":"他们","it":"它"}
possessives = {"my":"我的","your":"你的","his":"他的","her":"她的","their":"他们的","our":"我们的"}
verbs = {"eat":"吃","drink":"喝","go":"去","play":"玩","like":"喜欢","see":"看","know":"知道","say":"说","love":"爱","want":"想","do":"做","make":"做/制作","take":"拿","get":"得到","have":"有","check":"检查","ensure":"确定"}
adjectives = {"happy":"快乐","total":"全体的","maximum":"极点","specific":"特定"}
time_words = {"today":"今天","tomorrow":"明天","yesterday":"昨天","now":"现在","morning":"早上","afternoon":"下午","evening":"晚上","then":"然后"}
connectors = {"and":"和","or":"或","but":"但是","to":"","for":"为","in":"在"}
be_verbs = {"am":"是","is":"是","are":"是","was":"是","were":"是","be":"是"}
negations = {"not":"不","don't":"不","doesn't":"不","didn't":"没"}
question_words = {"what":"什么","who":"谁","where":"哪里","when":"什么时候","why":"为什么","how":"怎么"}
others = {"ok":"好","nice":"美好","hi":"嗨","hello":"你好","friend":"朋友","book":"书","phone":"电话","seat":"座","availability":"有效性","status":"状态","capacity":"容量","class":"班级","verifying":"证明"}

fixed_combinations = {
    ("thank","you"):"谢谢",
    ("thanks",):"谢谢",
    ("good","morning"):"早上好",
    ("good","afternoon"):"下午好",
    ("good","evening"):"晚上好",
    ("good","night"):"晚安",
    ("be","ok"):"很好",
    ("play","game"):"玩游戏",
    ("eat","apple"):"吃苹果",
    ("drink","water"):"喝水",
    ("how","are","you"):"你好吗",
    ("see","you"):"再见",
    ("year","old"):"岁"
}

def lookup(word: str) -> str:
    lw = word.lower()
    if lw in ignore_words:
        return ""

    for dic in [subjects, possessives, verbs, adjectives, time_words, connectors, be_verbs, negations, question_words, others]:
        if lw in dic:
            return dic[lw]

    if lw in dataset_dict:
        trans = dataset_dict[lw]
        return clean_translation_text(trans)

    return "未知词"

# -----------------------------
# Sentence preprocessing
# -----------------------------
def preprocess_sentence(sentence: str):
    sentence = expand_contractions(sentence)
    sentence = re.sub(r"[^\w\s]", "", sentence)
    sentence = sentence.lower()
    words = sentence.strip().split()
    words = [w for w in words if w not in ignore_words]
    return words

# -----------------------------
# RBMT Translation
# -----------------------------
def generate_translation(sentence: str) -> str:
    words = preprocess_sentence(sentence)
    result = []
    i = 0
    while i < len(words):
        w = words[i].lower()
        if w in ignore_words:
            i += 1
            continue

        # 三词固定搭配
        if i+2 < len(words) and (words[i].lower(), words[i+1].lower(), words[i+2].lower()) in fixed_combinations:
            result.append(fixed_combinations[(words[i].lower(), words[i+1].lower(), words[i+2].lower())])
            i += 3
            continue

        # 两词固定搭配
        if i+1 < len(words) and (w, words[i+1].lower()) in fixed_combinations:
            result.append(fixed_combinations[(w, words[i+1].lower())])
            i += 2
            continue

        # 时间词
        if w in time_words:
            result.append(time_words[w])
            i += 1
            continue

        # 疑问词
        if w in question_words:
            result.append(question_words[w])
            i += 1
            continue

        # 主语
        if w in subjects:
            result.append(subjects[w])
            i += 1
            continue

        # be 动词
        if w in be_verbs:
            if i+1 < len(words) and words[i+1].lower() == "ok":
                result.append("很好")
                i += 2
                continue
            if i+1 < len(words):
                nxt = words[i+1].lower()
                if nxt in adjectives:
                    result.append("很" + adjectives[nxt])
                    i += 2
                    continue
            result.append(be_verbs[w])
            i += 1
            continue

        # 否定
        if w in negations:
            result.append(negations[w])
            i += 1
            continue

        # 所有格 + 名词
        if w in possessives and i+1 < len(words):
            nxt = words[i+1].lower()
            obj = lookup(nxt)
            if obj:
                result.append(possessives[w] + obj)
                i += 2
                continue

        # 动词 + 宾语
        if w in verbs:
            verb_trans = verbs[w]
            obj = ""
            if i+1 < len(words):
                nxt = words[i+1].lower()
                if nxt not in connectors and nxt not in ignore_words:
                    obj = lookup(nxt)
                    if obj:
                        result.append(verb_trans + obj)
                        i += 2
                        continue
            result.append(verb_trans)
            i += 1
            continue

        # 形容词
        if w in adjectives:
            result.append(adjectives[w])
            i += 1
            continue

        # 连词
        if w in connectors:
            result.append(connectors[w])
            i += 1
            continue

        # fallback
        trans = lookup(w)
        if trans:
            result.append(trans)
        i += 1

    return "".join(result)



input_sentence = ""

# Loading the NMT model
@st.cache_resource
def load_nmt_model(model_dir=NMT_MODEL):
    tokenizer = MarianTokenizer.from_pretrained(model_dir)
    model = MarianMTModel.from_pretrained(model_dir)
    return tokenizer, model

nmtTokenizer, nmtModel = load_nmt_model()

# ------------------------
# Preload NLTK data (if needed)
# ------------------------
NLTK_DIR = "./nltk"
nltk.data.path.append(NLTK_DIR)

def ngram_lm_score(sentence, lm_dict):
    words = sentence.split()
    score = 0.0
    for i, word in enumerate(words):
        score += lm_dict[1].get(word, -10.0)
        if i > 0:
            bigram = f"{words[i-1]} {words[i]}"
            score += lm_dict[2].get(bigram, -5.0)
        if i > 1:
            trigram = f"{words[i-2]} {words[i-1]} {words[i]}"
            score += lm_dict[3].get(trigram, -2.5)
    return score

def translate_word(english_word, en_cn_probs, probability_threshold=0.0):
    english_word = english_word.lower()
    translations = {
        zh_word: prob
        for zh_word, prob in en_cn_probs.get(english_word, {}).items()
        if prob >= probability_threshold
    }
    return dict(sorted(translations.items(), key=lambda item: item[1], reverse=True))

def sentence_decode(eng_sentence, lm_dict, en_cn_probs):
    tokens = nltk.word_tokenize(eng_sentence.lower())
    u_tokens = sorted(set(tokens), key=tokens.index)
    translations = {token: list(translate_word(token, en_cn_probs).keys()) for token in u_tokens}

    # Generate all translation combinations
    translation_options = [translations.get(token, []) for token in u_tokens]
    translation_iterations = {" ".join(comb) for comb in itertools.product(*translation_options)}

    # Remove repeated words
    new_iterations = set()
    for sentence in translation_iterations:
        words = sentence.split()
        counts = defaultdict(int)
        for word in words:
            counts[word] += 1
        for word, count in counts.items():
            if count > 1:
                new_words = list(words)
                new_words.remove(word)
                new_iterations.add(" ".join(new_words))
    translation_iterations |= new_iterations

    # Score candidates
    scored_sentences = []
    for sentence in sorted(translation_iterations):
        words = sentence.split()
        translation_prob = 1.0
        for en_word, zh_word in zip(u_tokens, words):
            prob = en_cn_probs.get(en_word, {}).get(zh_word, 0.0)
            translation_prob *= prob if prob > 0 else 1e-9
        log_prob = math.log(translation_prob)
        lm_score_val = ngram_lm_score(sentence, lm_dict)
        final_score = ALPHA * log_prob + BETA * lm_score_val
        scored_sentences.append((sentence, final_score))

    # Return top candidate
    top1 = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:1]
    for sentence, _ in top1:
        return sentence.replace(" ", "")


st.title("Practical Comparison: RBMT vs. SMT vs. NMT vs. Google Translate")

input_sentence = st.text_input("Please enter an English sentence:")
reference = st.text_input("Enter the Reference Translation:")

# Create two columns for the buttons
col1, col2 = st.columns(2)

with col1:
    translate_btn = st.button("Translate")  # RBMT/NMT/Google

with col2:
    smt_btn = st.button("Run SMT method (Warning, could crash!)")

if translate_btn and input_sentence:
    rbmt_input = input_sentence
    nmt_input = input_sentence
    google_input = input_sentence

    # Running RBMT Translation
    def split_into_clauses(sentence):
        clauses = re.split(r'[,.!?;]| and | or ', sentence)
        return [c.strip() for c in clauses if c.strip()]

    rbmt_input = expand_contractions(rbmt_input)
    words = preprocess_sentence(rbmt_input)

    if len(words) <= 8:
        rbmt_output = generate_translation(rbmt_input)
    else:
        clauses = split_into_clauses(rbmt_input)
        translated_clauses = [generate_translation(c) for c in clauses]
        rbmt_output = "，".join(translated_clauses)


    # Running NMT Translation
    inputs = nmtTokenizer(nmt_input, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = nmtModel.generate(**inputs)
    nmt_output = nmtTokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    # Running Google Translation
    google_output = GoogleTranslator(source="en", target="zh-CN").translate(google_input)
    
    # Display translation results 
    st.markdown("### Translation Results")
    
    st.markdown(f"**RBMT Output:** {rbmt_output}")
    st.markdown(f"**NMT Output:** {nmt_output}")
    st.markdown(f"**Google Output:** {google_output}")
    
    if reference:
        rbmt_bleu = sacrebleu.sentence_bleu(rbmt_output, [reference], tokenize='zh') 
        nmt_bleu = sacrebleu.sentence_bleu(nmt_output, [reference], tokenize='zh') 
        google_bleu = sacrebleu.sentence_bleu(google_output, [reference], tokenize='zh')
        
        st.markdown("### BLEU Scores")
        st.markdown(f"- **RBMT BLEU:** {rbmt_bleu.score:.2f}")
        st.markdown(f"- **NMT BLEU:** {nmt_bleu.score:.2f}")
        st.markdown(f"- **Google BLEU:** {google_bleu.score:.2f}")
    else:
        st.markdown("No reference provided, BLEU Score calculation skipped.")

# Running SMT Translation
if smt_btn and input_sentence:
    lm_dict, en_cn_probs = load_smt_resources()  # Lazy load
    smt_input = re.sub(r'[^\w\s]', '', input_sentence)
    smt_output = sentence_decode(smt_input, lm_dict, en_cn_probs)

    # Display SMT results 
    st.markdown("### SMT Translation Result")
    
    smt_output = sentence_decode(smt_input, lm_dict, en_cn_probs)
    st.markdown(f"**SMT Output:** {smt_output}")
    
    if reference:
        smt_bleu = sacrebleu.sentence_bleu(smt_output, [reference], tokenize='zh')
        st.markdown(f"**SMT BLEU:** {smt_bleu.score:.2f}")
    else:
        st.markdown("No reference provided, BLEU Score calculation skipped.")



