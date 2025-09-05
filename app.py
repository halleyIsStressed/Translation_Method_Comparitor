from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline
from google.colab import files
from google.colab import drive
from googletrans import Translator as gt
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

import itertools
import sacrebleu
import requests
import kenlm
import nltk
import math
import json
import re
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Load JSON -> Dictionary
def load_dictionary(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dictionary = {}
    for entry in raw_data:
        eng = entry.get("word", "").strip().lower()
        trans_list = entry.get("translation", [])
        if isinstance(trans_list, list) and len(trans_list) > 0:
            dictionary[eng] = trans_list[0]
    return dictionary

dict_path = "/content/data/MyDrive/AI Assignment Colab/data/filtered.json"
dictionary = load_dictionary(dict_path)
print(f"✅ Dictionary loading success, total number of entries: {len(dictionary)}")

# 3. Clean the translated text

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

def lookup(word: str) -> str:
    word = word.lower()
    raw = dictionary.get(word, "")
    return clean_translation_text(raw)

# NLTK Initialisation & Participle + Word Form Reduction
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

def analyze(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    words = sentence.split()
    words = [lemmatizer.lemmatize(w, pos='v') for w in words]
    words = [lemmatizer.lemmatize(w, pos='n') for w in words]
    return words

# RBMT generates translation (最新版)

# -----------------------------
# 规则字典
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

# -----------------------------
# 固定搭配短语
# -----------------------------
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

# -----------------------------
# 缩写字典
# -----------------------------
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

ignore_words = ["to", "a", "an", "the"]  # 冠词/虚词忽略

def expand_contractions(sentence):
    for k,v in contractions.items():
        sentence = re.sub(r"\b"+k+r"\b", v, sentence, flags=re.IGNORECASE)
    return sentence

# -----------------------------
# 载入 JSON dataset
# -----------------------------
dict_path = "/content/data/MyDrive/AI Assignment Colab/data/filtered.json"  # 改成你的路径
with open(dict_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

dataset_dict = {}
for entry in raw_data:
    eng = entry.get("word","").strip().lower()
    trans_list = entry.get("translation", [])
    if isinstance(trans_list, list) and len(trans_list) > 0:
        dataset_dict[eng] = trans_list[0]

# -----------------------------
# clean translation
# -----------------------------
def clean_translation_text(text):
    if isinstance(text, list):
        text = text[0] if text else ""
    if not isinstance(text, str):
        return ""
    candidates = re.split(r"[;,；，、/]| or ", text)
    for cand in candidates:
        cand = re.sub(r"^(?:n|v|vi|vt|adj|adv|prep|conj|pron|abbr|int|interj|art|aux|num|det|modal|phr|idiom)\.\s*", "", cand, flags=re.IGNORECASE)
        cand = re.sub(r"【[^】]*】|\([^)]*\)|（[^）]*）|\[[^\]]*\]", "", cand).strip()
        if re.search(r"[\u4e00-\u9fff]", cand):
            return cand
    return candidates[0].strip() if candidates else ""

# -----------------------------
# lookup 函数
# -----------------------------
def lookup(word):
    lw = word.lower()
    if lw in ignore_words:
        return ""   # 冠词忽略

    for dic in [subjects, possessives, verbs, adjectives, time_words, connectors, be_verbs, negations, question_words, others]:
        if lw in dic:
            return dic[lw]

    if lw in dataset_dict:
        trans = dataset_dict[lw]
        return clean_translation_text(trans)

    # fallback，保证永远返回完整中文
    return "未知词"

# -----------------------------
# 预处理句子，去掉标点，分词
# -----------------------------
def preprocess_sentence(sentence):
    # 统一替换缩写
    sentence = expand_contractions(sentence)
    # 去掉所有标点符号，包括句号、逗号等
    sentence = re.sub(r"[^\w\s]", "", sentence)
    # 小写化
    sentence = sentence.lower()
    # 分词
    words = sentence.strip().split()
    # 忽略冠词/虚词
    words = [w for w in words if w not in ignore_words]
    return words


# -----------------------------
# generate 函数（接收完整句子）
# -----------------------------
def generate_translation(sentence):
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
nmtTokenizer = MarianTokenizer.from_pretrained("./data/MyDrive/AI Assignment Colab/data/fine_tuned_marian")
nmtModel = MarianMTModel.from_pretrained("./data/MyDrive/AI Assignment Colab/data/fine_tuned_marian")

# Loading the SMT model
model = kenlm.Model('./data/MyDrive/AI Assignment Colab/data/Chinese_LM.arpa')
alpha = 1
beta = 1
with open("./data/MyDrive/AI Assignment Colab/data/En_Cn_Probs.json", "r", encoding="utf-8") as f:
    en_cn_probs = json.load(f)

def translate_word(english_word, probability_threshold=0.0):
    english_word = english_word.lower()

    with open("./data/MyDrive/AI Assignment Colab/data/En_Cn_Probs.json", "r", encoding="utf-8") as f:
        word_probabilities = json.load(f)

    translations = {
        zh_word: prob
        for zh_word, prob in word_probabilities.get(english_word, {}).items()
        if prob >= probability_threshold
    }

    return dict(sorted(translations.items(), key=lambda item: item[1], reverse=True))

def sentence_decode(eng_sentence):
    tokens = nltk.word_tokenize(eng_sentence.lower())

    # Generate all possible translation combinations
    u_tokens = sorted(set(tokens), key=tokens.index)
    translations = {token: list(translate_word(token).keys()) for token in u_tokens}

    # Generate combinations of Chinese words
    translation_options = [translations.get(token, []) for token in u_tokens]
    translation_iterations = {" ".join(combination) for combination in itertools.product(*translation_options)}

    # Handle repeated words by creating variations with fewer duplicates
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

    # Save all translation candidates
    with open("Trans_Chinese_Iter.txt", "w", encoding="utf-8") as f:
        for i, sentence in enumerate(sorted(translation_iterations), 1):
            f.write(f"{sentence}\n")

    # Score candidates with KenLM + word translation probabilities
    scored_sentences = []
    with open("Trans_Iter_Score.txt", "w", encoding="utf-8") as f:
        for sentence in sorted(translation_iterations):
            words = sentence.split()

            translation_prob = 1.0
            for en_word, zh_word in zip(u_tokens, words):
                prob = en_cn_probs.get(en_word, {}).get(zh_word, 0.0)
                translation_prob *= prob if prob > 0 else 1e-9
            log_prob = math.log(translation_prob)

            lm_score = model.score(sentence)
            final_score = alpha * log_prob + beta * lm_score
            scored_sentences.append((sentence, final_score))

            f.write(f"{sentence}\t{final_score:.6f}\n")

    # Print top 5 by final score
    top_5 = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:1]
    for sentence, score in top_5:
        return sentence.replace(" ", "")



input_sentence = input("Please enter an English sentence:")
reference = input("Enter the Reference Translation: ")
print("")

rbmt_input = input_sentence
smt_input = input_sentence
nmt_input = input_sentence
google_input = input_sentence

# Running RBMT Translator
def split_into_clauses(sentence):
    clauses = re.split(r'[,.!?;]| and | or ', sentence)
    return [c.strip() for c in clauses if c.strip()]

rbmt_input = expand_contractions(rbmt_input)
words = analyze(rbmt_input)

if len(words) <= 8:
  rbmt_output = generate_translation(rbmt_input)
else:
  clauses = split_into_clauses(rbmt_input)
  translated_clauses = [generate_translation(c) for c in clauses]
  rbmt_output = "，".join(translated_clauses)

print(f"RBMT Output\t: {rbmt_output}")


# Running SMT Translator
smt_input = re.sub(r'[^\w\s]', '', smt_input)
print(f"SMT Output\t: {sentence_decode(smt_input)}")


# Running NMT Translator
inputs = nmtTokenizer(nmt_input, return_tensors="pt", padding=True, truncation=True)
translated_tokens = nmtModel.generate(**inputs)
nmt_output = nmtTokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print(f"NMT Output\t: {nmt_output}")


# Running Google Translator
google_translator = gt()
google_output = await google_translator.translate(google_input, dest="zh-cn")
print(f"Google Output\t: {google_output.text}")

rbmt_bleu   = sacrebleu.sentence_bleu(rbmt_output, [reference], tokenize='zh')
smt_bleu    = sacrebleu.sentence_bleu(sentence_decode(smt_input), [reference], tokenize='zh')
nmt_bleu    = sacrebleu.sentence_bleu(nmt_output, [reference], tokenize='zh')
google_bleu = sacrebleu.sentence_bleu(google_output.text, [reference], tokenize='zh')

print(f"\nRBMT BLEU \t: {rbmt_bleu.score:.2f}")
print(f"SMT BLEU \t: {smt_bleu.score:.2f}")
print(f"NMT BLEU \t: {nmt_bleu.score:.2f}")
print(f"Google BLEU \t: {google_bleu.score:.2f}")

