import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem


def clean_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^а-яё]", "", token)
    return token


def is_noise(token: str) -> bool:
    # remove tokens containing latin letters or digits
    return bool(re.search(r"[0-9a-zA-Z]", token))


def process_file(filepath, output_folder, russian_stopwords, mystem):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    raw_tokens = re.findall(r"[А-Яа-яЁё]+", text)

    tokens_set = set()

    for tok in raw_tokens:
        cleaned = clean_token(tok)

        if not cleaned:
            continue
        if len(cleaned) < 2:
            continue
        if cleaned in russian_stopwords:
            continue
        if is_noise(cleaned):
            continue

        tokens_set.add(cleaned)

    tokens_list = sorted(tokens_set)

    filename = os.path.splitext(os.path.basename(filepath))[0]

    # -------------------
    # SAVE TOKENS FILE
    # -------------------
    tokens_output_path = os.path.join(output_folder, f"{filename}_tokens.txt")

    with open(tokens_output_path, "w", encoding="utf-8") as f:
        for token in tokens_list:
            f.write(token + "\n")

    # -------------------
    # LEMMATIZATION
    # -------------------
    lemma_dict = defaultdict(list)

    for token in tokens_list:
        lemma = mystem.lemmatize(token)[0].strip()
        if lemma:
            lemma_dict[lemma].append(token)

    lemmas_output_path = os.path.join(output_folder, f"{filename}_lemmas.txt")

    with open(lemmas_output_path, "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_dict.keys()):
            words = " ".join(sorted(lemma_dict[lemma]))
            f.write(f"{lemma} {words}\n")


def main():
    input_folder = "input_texts"
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    nltk.download("stopwords")
    russian_stopwords = set(stopwords.words("russian"))

    mystem = Mystem()

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            process_file(filepath, output_folder, russian_stopwords, mystem)

    print("\n✅ Done! Each document now has its own tokens and lemmas files.")


if __name__ == "__main__":
    main()