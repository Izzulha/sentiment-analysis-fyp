# =====================================================
# IMPORTS
# =====================================================
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = Path(r"C:\Users\Admin\Desktop\FYP\model_savee")
DATA_PATH = Path(r"C:\Users\Admin\Desktop\FYP\deploy.xlsx")
TEXT_COLUMN = "text"
MAX_LENGTH = 128
BATCH_SIZE = 16

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# LOAD TOKENIZER & MODEL (LOCAL)
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
).to(device)

model.eval()
print("Model & tokenizer loaded successfully ✅")

# =====================================================
# LABEL MAP
# =====================================================
label_map = {0: "Negative", 1: "Positive"}

# =====================================================
# LOAD EXCEL DATA
# =====================================================
df = pd.read_excel(DATA_PATH)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found in Excel file")

texts = df[TEXT_COLUMN].dropna().astype(str).tolist()
print("Total texts loaded:", len(texts))

# =====================================================
# MALAY STOPWORDS (FROM WORD CLOUD ANALYSIS)
# =====================================================
MALAY_STOPWORDS = {
     "yang","dan","itu","ini","ada","pun","lah","sahaja","juga","untuk",
    "dengan","dari","pada","sebagai","kerana","bila","kalau","sebab","maka","atau",
    "saya","aku","kita","kami","mereka","dia","kau","anda","ni","tu","je",
    "dah","tak","tidak","bukan","la","kan","kat","macam","tau",
    "orang","kerja","pekerja","negara","malaysia","lagi","lebih","masa","hari","semua","sama","sendiri","lain","dalam",
    "buat","boleh","dapat","nak","pernah","masih","mula","sekarang","sejak",
    "jika","hingga","tanpa","oleh","tentang","terhadap","seperti","selain",
    "jangan","sedang","telah","akan","baru","setiap","ramai","banyak",
    "kena","sampai","cukup","terlalu","terus","balik","perlu","mesti",
    "jg","lg","dgn","dlm","pd","sy","ko","korang","aq","org","dh","nk","blh","bkn","depa","tapi","nya","apa","mau","tapi","sya","pon","klau","plak","plak","macam","ko","kau","sub","la","jim","abang","jer","bang"
    "tu","ke","laa","dia","saja","dok","apa","16 jam","jer","bro","klu","dorang","diaorang",
    "orang","ni","pun","lah","gak","sini","sana",
    "mana","semua","satu","hari","jam","bulan","tahun","lama","sikit",
    "buat","cakap","dan","yang",
    "untuk","dari","dengan","pada","bila","kalau","je","S","memang","ini","rm9k","ini","pula","itu","ke","dont","apa","lah","pun","bang","siapa","itu","abang"
}

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict(texts):
    results = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        for j, text in enumerate(batch):
            results.append({
                "text": text,
                "label": label_map[preds[j].item()],
                "confidence": round(probs[j][preds[j]].item(), 4)
            })

    return results

# =====================================================
# WORD CLOUD FUNCTION (STOPWORD-FILTERED)
# =====================================================
def generate_wordcloud(texts, title):
    cleaned_words = []

    for text in texts:
        words = text.lower().split()
        words = [w for w in words if w not in MALAY_STOPWORDS and len(w) > 2]
        cleaned_words.extend(words)

    if not cleaned_words:
        print(f"No valid words for {title}")
        return

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        colormap="viridis"
    ).generate(" ".join(cleaned_words))

    img = wc.to_image()

    plt.figure(figsize=(11, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    predictions = predict(texts)
    result_df = pd.DataFrame(predictions)

    result_df.to_excel("sentiment_results4.xlsx", index=False)
    print("Saved: sentiment_results4.xlsx")

    positive_texts = result_df[result_df["label"] == "Positive"]["text"].tolist()
    negative_texts = result_df[result_df["label"] == "Negative"]["text"].tolist()

    if positive_texts:
        generate_wordcloud(positive_texts, "Positive Sentiment Word Cloud")

    if negative_texts:
        generate_wordcloud(negative_texts, "Negative Sentiment Word Cloud")
