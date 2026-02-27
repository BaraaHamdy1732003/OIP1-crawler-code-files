import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem


# -----------------------------
# REMOVE HTML TAGS
# -----------------------------
def remove_html_tags(text: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", text)
    return clean


# -----------------------------
# CLEAN TOKEN
# -----------------------------
RUS_LETTERS_RE = re.compile(r"[^а-яё]")

def clean_token(token: str) -> str:
    token = token.lower().strip()
    token = RUS_LETTERS_RE.sub("", token)
    return token


# -----------------------------
# PROCESS ONE FILE
# -----------------------------
def process_file(filepath: str, output_folder: str, russian_stopwords: set, mystem: Mystem):

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 1️⃣ Remove HTML
    text = remove_html_tags(text)

    # 2️⃣ Extract Russian words only
    raw_tokens = re.findall(r"[А-Яа-яЁё]+", text)

    tokens_set = set()

    for token in raw_tokens:
        cleaned = clean_token(token)

        if not cleaned:
            continue
        if len(cleaned) < 2:
            continue
        if cleaned in russian_stopwords:
            continue

        tokens_set.add(cleaned)

    tokens_sorted = sorted(tokens_set)

    filename = os.path.splitext(os.path.basename(filepath))[0]

    # -----------------------------
    # SAVE TOKENS
    # -----------------------------
    tokens_path = os.path.join(output_folder, f"{filename}_tokens.txt")

    with open(tokens_path, "w", encoding="utf-8") as f:
        for token in tokens_sorted:
            f.write(token + "\n")

    # -----------------------------
    # GROUP BY LEMMAS
    # -----------------------------
    lemma_dict = defaultdict(set)

    for token in tokens_sorted:
        lemma = mystem.lemmatize(token)[0].strip()
        if lemma:
            lemma_dict[lemma].add(token)

    lemmas_path = os.path.join(output_folder, f"{filename}_lemmas.txt")

    with open(lemmas_path, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_dict.keys()):
            words = " ".join(sorted(lemma_dict[lemma]))
            f.write(f"{lemma} {words}\n")


# -----------------------------
# MAIN
# -----------------------------
def main():

    input_folder = "D:\OIP\OIP1-crawler-code-files\pages"      # folder with 117 txt files
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    nltk.download("stopwords", quiet=True)
    russian_stopwords = set(stopwords.words("russian"))

    mystem = Mystem()

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])

    processed = 0

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        process_file(filepath, output_folder, russian_stopwords, mystem)
        processed += 1

    print("\n✅ DONE")
    print(f"Processed files: {processed}")
    print(f"Generated {processed} token files")
    print(f"Generated {processed} lemma files")

    if processed != 117:
        print("⚠ WARNING: Number of processed files is not 117!")
        print("Check your input folder.")


if __name__ == "__main__":
    main()