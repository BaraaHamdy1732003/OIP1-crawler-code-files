import os
import re
import json
from collections import defaultdict


def tokenize(text: str):
    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", text.lower())
    return words


def build_inverted_index(folder_path: str):
    index = defaultdict(set)

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            tokens = tokenize(text)

            for token in tokens:
                index[token].add(filename)

    # convert sets to sorted lists (JSON compatible)
    index_json = {term: sorted(list(files)) for term, files in index.items()}
    return index_json


def main():
    input_folder = "input_texts"
    output_file = "inverted_index.json"

    index = build_inverted_index(input_folder)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"✅ Inverted index saved to {output_file}")


if __name__ == "__main__":
    main()
