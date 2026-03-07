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
    return re.sub(r"<[^>]+>", " ", text)


# -----------------------------
# CLEAN TOKEN
# -----------------------------
RUS_LETTERS_RE = re.compile(r"[^а-яё]")

def clean_token(token: str) -> str:
    token = token.lower()
    token = RUS_LETTERS_RE.sub("", token)
    return token


# -----------------------------
# PROCESS ONE FILE
# -----------------------------
def process_file(filepath: str, output_folder: str, russian_stopwords: set, mystem: Mystem):

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # 1️⃣ remove html
    text = remove_html_tags(text)

    # 2️⃣ extract russian tokens
    raw_tokens = re.findall(r"[А-Яа-яЁё]+", text)

    tokens = []

    for token in raw_tokens:

        cleaned = clean_token(token)

        if not cleaned:
            continue
        if len(cleaned) < 2:
            continue
        if cleaned in russian_stopwords:
            continue

        tokens.append(cleaned)

    # sort tokens alphabetically
    tokens_sorted = sorted(tokens)

    filename = os.path.splitext(os.path.basename(filepath))[0]

    # -----------------------------
    # SAVE TOKENS
    # -----------------------------
    tokens_path = os.path.join(output_folder, f"{filename}_tokens.txt")

    with open(tokens_path, "w", encoding="utf-8") as f:
        for token in tokens_sorted:
            f.write(token + "\n")

    # -----------------------------
    # LEMMATIZATION (batch)
    # -----------------------------
    lemmas = mystem.lemmatize(" ".join(tokens))

    lemma_dict = defaultdict(set)

    token_index = 0

    for lemma in lemmas:

        lemma = lemma.strip()

        if not lemma:
            continue

        if token_index >= len(tokens):
            break

        token = tokens[token_index]
        lemma_dict[lemma].add(token)

        token_index += 1

    # -----------------------------
    # SAVE LEMMAS
    # -----------------------------
    lemmas_path = os.path.join(output_folder, f"{filename}_lemmas.txt")

    with open(lemmas_path, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_dict.keys()):
            words = " ".join(sorted(lemma_dict[lemma]))
            f.write(f"{lemma} {words}\n")


# -----------------------------
# MAIN
# -----------------------------
def main():

    input_folder = r"D:\OIP\OIP1-crawler-code-files\pages"
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