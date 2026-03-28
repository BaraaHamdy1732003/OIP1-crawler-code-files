import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
def load_documents(folder_path):
    documents = []
    filenames = []

    files = sorted(os.listdir(folder_path), key=lambda x: int(x.split(".")[0]))

    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                documents.append(text)
                filenames.append(file)

    return documents, filenames


# -----------------------------
# LOAD INDEX FILE (doc_id → URL)
# -----------------------------
def load_index(index_path):
    index = {}

    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                doc_id, url = parts
                index[int(doc_id)] = url

    return index


# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search(query, vectorizer, doc_vectors, filenames, index, top_k=5):
    query_vec = vectorizer.transform([query])

    # cosine similarity
    similarities = (doc_vectors * query_vec.T).toarray().flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for i in top_indices:
        doc_name = filenames[i]
        doc_id = int(doc_name.split(".")[0])
        url = index.get(doc_id, "No URL")

        results.append((doc_name, url, similarities[i]))

    return results


# -----------------------------
# MAIN
# -----------------------------
def main():
    documents_folder = r"D:\OIP\OIP1-crawler-code-files\pages"
    index_file = r"D:\OIP\OIP1-crawler-code-files\index.txt"

    print("Loading documents...")
    documents, filenames = load_documents(documents_folder)

    print("Loading index...")
    index = load_index(index_file)

    print("Building TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    doc_vectors = vectorizer.fit_transform(documents)

    print("\n✅ Search Engine Ready!\n")

    while True:
        query = input("Enter your search query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = search(query, vectorizer, doc_vectors, filenames, index)

        print("\nTop results:")
        for doc, url, score in results:
            print(f"{doc} | {url} | score={score:.4f}")

        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()