import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

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
                documents.append(f.read())
                filenames.append(file)

    return documents, filenames


# -----------------------------
# LOAD INDEX
# -----------------------------
def load_index(index_path):
    index = {}

    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                index[int(parts[0])] = parts[1]

    return index


# -----------------------------
# INIT DATA (RUN ONCE)
# -----------------------------
documents, filenames = load_documents("pages")
index = load_index("index.txt")

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
doc_vectors = vectorizer.fit_transform(documents)


# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search(query, top_k=10):
    query_vec = vectorizer.transform([query])
    similarities = (doc_vectors * query_vec.T).toarray().flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for i in top_indices:
        doc_id = int(filenames[i].split(".")[0])
        url = index.get(doc_id, "No URL")

        results.append({
            "doc": filenames[i],
            "url": url,
            "score": float(similarities[i])
        })

    return results


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_api():
    query = request.json.get("query")
    results = search(query)
    return jsonify(results)


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)