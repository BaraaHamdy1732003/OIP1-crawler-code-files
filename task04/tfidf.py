import os
import math
import re
from collections import Counter, defaultdict
from pymystem3 import Mystem


INPUT_FOLDER = r"D:\OIP\OIP1-crawler-code-files\task02\input_texts"
OUTPUT_TERMS_FOLDER = "tfidf_terms"
OUTPUT_LEMMAS_FOLDER = "tfidf_lemmas"


def tokenize(text):
    return re.findall(r"[A-Za-zА-Яа-яЁё]+", text.lower())


def read_documents():
    docs = {}
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as f:
                docs[filename] = f.read()
    return docs


def compute_idf(documents_tokens):
    N = len(documents_tokens)
    df = defaultdict(int)

    for tokens in documents_tokens.values():
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1

    idf = {}
    for term, doc_count in df.items():
        idf[term] = math.log(N / doc_count)

    return idf


def compute_tf(tokens):
    total_terms = len(tokens)
    counts = Counter(tokens)
    tf = {}

    for term, count in counts.items():
        tf[term] = count / total_terms

    return tf


def save_results(doc_name, tf_dict, idf_dict, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, doc_name)

    with open(output_path, "w", encoding="utf-8") as f:
        for term in sorted(tf_dict.keys()):
            idf_value = idf_dict.get(term, 0)
            tfidf = tf_dict[term] * idf_value
            f.write(f"{term} {idf_value:.6f} {tfidf:.6f}\n")


def main():
    documents = read_documents()

    # Tokenize documents
    documents_tokens = {
        name: tokenize(text)
        for name, text in documents.items()
    }

    # Compute IDF for terms
    idf_terms = compute_idf(documents_tokens)

    # Save term TF-IDF
    for name, tokens in documents_tokens.items():
        tf = compute_tf(tokens)
        save_results(name, tf, idf_terms, OUTPUT_TERMS_FOLDER)

    # Lemmatization
    mystem = Mystem()
    documents_lemmas = {}

    for name, tokens in documents_tokens.items():
        lemmas = []
        for token in tokens:
            lemma = mystem.lemmatize(token)[0].strip()
            if lemma:
                lemmas.append(lemma)
        documents_lemmas[name] = lemmas

    # Compute IDF for lemmas
    idf_lemmas = compute_idf(documents_lemmas)

    # Save lemma TF-IDF
    for name, lemmas in documents_lemmas.items():
        tf = compute_tf(lemmas)
        save_results(name, tf, idf_lemmas, OUTPUT_LEMMAS_FOLDER)

    print("✅ TF-IDF calculation completed.")


if __name__ == "__main__":
    main()
