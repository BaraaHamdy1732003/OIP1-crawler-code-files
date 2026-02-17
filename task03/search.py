import json
import re


def tokenize_query(query: str):
    query = query.replace("(", " ( ").replace(")", " ) ")
    return query.split()


def precedence(op):
    if op == "NOT":
        return 3
    if op == "AND":
        return 2
    if op == "OR":
        return 1
    return 0


def to_postfix(tokens):
    output = []
    stack = []

    for token in tokens:
        token_upper = token.upper()

        if token_upper in ("AND", "OR", "NOT"):
            while stack and stack[-1] != "(" and precedence(stack[-1]) >= precedence(token_upper):
                output.append(stack.pop())
            stack.append(token_upper)

        elif token == "(":
            stack.append(token)

        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            stack.pop()

        else:
            output.append(token.lower())

    while stack:
        output.append(stack.pop())

    return output


def evaluate_postfix(postfix, index, all_docs):
    stack = []

    for token in postfix:
        if token == "AND":
            b = stack.pop()
            a = stack.pop()
            stack.append(a & b)

        elif token == "OR":
            b = stack.pop()
            a = stack.pop()
            stack.append(a | b)

        elif token == "NOT":
            a = stack.pop()
            stack.append(all_docs - a)

        else:
            stack.append(set(index.get(token, [])))

    return stack.pop() if stack else set()


def boolean_search(query, index):
    all_docs = set()
    for docs in index.values():
        all_docs.update(docs)

    tokens = tokenize_query(query)
    postfix = to_postfix(tokens)
    result_docs = evaluate_postfix(postfix, index, all_docs)

    return sorted(result_docs)


def main():
    with open("inverted_index.json", "r", encoding="utf-8") as f:
        index = json.load(f)

    query = input("Enter Boolean query: ")

    results = boolean_search(query, index)

    if results:
        print("\n✅ Documents found:")
        for doc in results:
            print(doc)
    else:
        print("\n❌ No documents matched.")


if __name__ == "__main__":
    main()
