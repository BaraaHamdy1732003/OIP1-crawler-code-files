import os
import re
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem


# -----------------------------
# TOKEN CLEANING
# -----------------------------
RUS_LETTERS_RE = re.compile(r"[^а-яё]")

def clean_token(token: str) -> str:
    token = token.lower().strip()
    token = RUS_LETTERS_RE.sub("", token)  # keep only Russian letters
    return token


# -----------------------------
# PROCESS ONE FILE
# -----------------------------
def process_file(filepath: str, output_folder: str, russian_stopwords: set[str], mystem: Mystem) -> None:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Extract only Russian-letter sequences (prevents digits/markup fragments)
    raw_tokens = re.findall(r"[А-Яа-яЁё]+", text)

    # Unique tokens per page (no duplicates)
    tokens_set: set[str] = set()

    for tok in raw_tokens:
        cleaned = clean_token(tok)

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
    # SAVE TOKENS FILE (per page)
    # -----------------------------
    tokens_output_path = os.path.join(output_folder, f"{filename}_tokens.txt")
    with open(tokens_output_path, "w", encoding="utf-8") as f:
        for token in tokens_sorted:
            f.write(token + "\n")

    # -----------------------------
    # SAVE LEMMAS FILE (per page)
    # lemma line format: <lemma> <token1> <token2> ... <tokenN>
    # -----------------------------
    lemma_dict: defaultdict[str, set[str]] = defaultdict(set)

    for token in tokens_sorted:
        lemma = mystem.lemmatize(token)[0].strip()
        if lemma:
            lemma_dict[lemma].add(token)

    lemmas_output_path = os.path.join(output_folder, f"{filename}_lemmas.txt")
    with open(lemmas_output_path, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_dict.keys()):
            words = " ".join(sorted(lemma_dict[lemma]))
            f.write(f"{lemma} {words}\n")


# -----------------------------
# MAIN
# -----------------------------
def main():
    # ✅ Your folder with 1.txt ... 117.txt
    input_folder = "/Users/amyargotti/PycharmProjects/OIP1-crawler-code-files/task02/input_texts/117_pages"
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    nltk.download("stopwords", quiet=True)
    russian_stopwords = set(stopwords.words("russian"))

    mystem = Mystem()

    # Sort files numerically if names are 1.txt, 2.txt, ...
    def sort_key(fn: str):
        base = os.path.splitext(fn)[0]
        return int(base) if base.isdigit() else base

    files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".txt")],
        key=sort_key
    )

    processed_files = 0

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        process_file(filepath, output_folder, russian_stopwords, mystem)
        processed_files += 1

    print("\n✅ DONE")
    print(f"Processed files: {processed_files}")
    print(f"Generated {processed_files} token files.")
    print(f"Generated {processed_files} lemma files.")

    if processed_files != 117:
        print("⚠ WARNING: The number of processed files is NOT 117!")
        print("Check your input folder contents (missing/extra .txt files).")


if __name__ == "__main__":
    main()