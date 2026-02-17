import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem


def read_all_txt_files(folder_path: str) -> str:
    full_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                full_text += f.read() + "\n"
    return full_text


def clean_token(token: str) -> str:
    token = token.lower().strip()
    token = re.sub(r"[^а-яё]", "", token)
    return token


def is_noise(token: str) -> bool:
    if re.search(r"[0-9a-zA-Z]", token):
        return True
    return False


def main():
    input_folder = "input_texts"
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    nltk.download("stopwords")
    russian_stopwords = set(stopwords.words("russian"))

    mystem = Mystem()

    text = read_all_txt_files(input_folder)

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

    # Save tokens.txt
    with open(os.path.join(output_folder, "tokens.txt"), "w", encoding="utf-8") as f:
        for token in tokens_list:
            f.write(token + "\n")

    # Lemmatization + grouping
    lemma_dict = defaultdict(list)

    for token in tokens_list:
        lemma = mystem.lemmatize(token)[0].strip()
        if lemma:
            lemma_dict[lemma].append(token)

    # Save lemmas.txt
    with open(os.path.join(output_folder, "lemmas.txt"), "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_dict.keys()):
            words = " ".join(sorted(lemma_dict[lemma]))
            f.write(f"{lemma} {words}\n")

    print("✅ Done! Output saved in output/ folder.")


if __name__ == "__main__":
    main()
