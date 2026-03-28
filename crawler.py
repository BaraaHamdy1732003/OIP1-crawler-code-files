import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import spacy


def read_all_txt_files(folder_path: str) -> str:
    full_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                full_text += f.read() + "\n"
    return full_text


def is_noise(token: str) -> bool:
    # remove tokens containing digits or latin letters
    if re.search(r"[0-9a-zA-Z]", token):
        return True
    return False


def main():
    input_folder = "input_texts"
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    nltk.download("stopwords")
    russian_stopwords = set(stopwords.words("russian"))

    # load Russian model
    nlp = spacy.load("ru_core_news_sm")

    text = read_all_txt_files(input_folder)

    doc = nlp(text)

    tokens_set = set()

    for token in doc:
        if token.is_punct or token.is_space:
            continue

        word = token.text.lower().strip()

        # keep only Russian letters
        word = re.sub(r"[^а-яё]", "", word)

        if not word:
            continue

        if len(word) < 2:
            continue

        if word in russian_stopwords:
            continue

        if is_noise(word):
            continue

        tokens_set.add(word)

    tokens_list = sorted(tokens_set)

    # save tokens.txt
    with open(os.path.join(output_folder, "tokens.txt"), "w", encoding="utf-8") as f:
        for t in tokens_list:
            f.write(t + "\n")

    # group by lemma
    lemma_dict = defaultdict(list)

    for t in tokens_list:
        lemma = nlp(t)[0].lemma_.lower()
        lemma_dict[lemma].append(t)

    # save lemmas.txt
    with open(os.path.join(output_folder, "lemmas.txt"), "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_dict.keys()):
            words = " ".join(sorted(lemma_dict[lemma]))
            f.write(f"{lemma} {words}\n")

    print("Done! Output saved in output/ folder.")


if __name__ == "__main__":
    main()
