import os
import time
import requests

URLS_FILE = "urls.txt"
OUTPUT_DIR = "pages"
INDEX_FILE = "index.txt"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SimpleCrawler/1.0)"
}

TIMEOUT = 15
DELAY_SECONDS = 1


def read_urls(filename):
    with open(filename, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


def download_page(url):
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    return response.status_code, response.text


def main():
    urls = read_urls(URLS_FILE)

    if len(urls) < 100:
        print("ERROR: urls.txt must contain at least 100 URLs")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    index_lines = []
    success_count = 0

    for i, url in enumerate(urls, start=1):
        print(f"[{i}] Downloading: {url}")

        try:
            status_code, html = download_page(url)

            if status_code != 200:
                print(f"   Failed (status code {status_code})")
                continue

            file_path = os.path.join(OUTPUT_DIR, f"{i}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html)

            index_lines.append(f"{i} {url}")
            success_count += 1

            print("   Saved successfully.")

        except Exception as e:
            print(f"   Error: {e}")

        time.sleep(DELAY_SECONDS)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines))

    print("\nDONE!")
    print(f"Downloaded pages: {success_count}")
    print(f"Index file saved as: {INDEX_FILE}")


if __name__ == "__main__":
    main()
