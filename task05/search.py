import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Sample Documents
# -----------------------------
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks.",
    "Python is a popular programming language.",
    "Vector search is used in semantic search systems.",
    "FAISS is a library for efficient similarity search."
]

# -----------------------------
# 2. Load Embedding Model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert documents to embeddings
doc_embeddings = model.encode(documents)

# Convert to numpy float32 (required by FAISS)
doc_embeddings = np.array(doc_embeddings).astype("float32")

# -----------------------------
# 3. Build FAISS Index
# -----------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(doc_embeddings)

print(f"Indexed {index.ntotal} documents.")

# -----------------------------
# 4. Search Function
# -----------------------------
def search(query, top_k=3):
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    distances, indices = index.search(query_vector, top_k)

    print("\nQuery:", query)
    print("\nTop results:\n")

    for i, idx in enumerate(indices[0]):
        print(f"Rank {i+1}:")
        print(f"Document: {documents[idx]}")
        print(f"Distance: {distances[0][i]}")
        print("-" * 40)

# -----------------------------
# 5. Example Query
# -----------------------------
search("What is artificial intelligence?")