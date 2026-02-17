import subprocess
import sys


def main():
    print("1) Building inverted index...")
    subprocess.run([sys.executable, "build_index.py"])

    print("\n2) Boolean search ready.")
    subprocess.run([sys.executable, "search.py"])


if __name__ == "__main__":
    main()
