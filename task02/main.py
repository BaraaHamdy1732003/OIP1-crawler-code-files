import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -----------------------------
# DOWNLOAD NLTK DATA
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# -----------------------------
# REMOVE HTML TAGS
# -----------------------------
def remove_html_tags(text):
    return re.sub(r"<[^>]+>", " ", text)


# -----------------------------
# CLEAN TOKEN
# -----------------------------
def clean_token(token):
    token = token.lower()
    token = re.sub(r"[^a-z]", "", token)
    return token


# -----------------------------
# PROCESS ONE FILE
# -----------------------------
def process_file(filepath, output_folder, stop_words, lemmatizer):

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # remove html
    text = remove_html_tags(text)

    # extract English tokens
    raw_tokens = re.findall(r"[A-Za-z]+", text)

    tokens = []

    for token in raw_tokens:

        cleaned = clean_token(token)

        if not cleaned:
            continue

        if len(cleaned) < 2:
            continue

        if cleaned in stop_words:
            continue

        tokens.append(cleaned)

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
    # GROUP BY LEMMAS
    # -----------------------------
    lemma_dict = defaultdict(set)

    for token in tokens:

        lemma = lemmatizer.lemmatize(token)

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

    input_folder = r"D:\OIP\OIP1-crawler-code-files\pages"
    output_folder = "output"

    os.makedirs(output_folder, exist_ok=True)

    stop_words = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()

    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".txt")])

    processed = 0

    for filename in files:

        filepath = os.path.join(input_folder, filename)

        print(f"Processing {filename}...")

        process_file(filepath, output_folder, stop_words, lemmatizer)

        processed += 1

    print("\nDONE")
    print(f"Processed files: {processed}")
    print(f"Generated {processed} token files")
    print(f"Generated {processed} lemma files")

    if processed != 117:
        print("WARNING: Expected 117 files!")


if __name__ == "__main__":
    main()