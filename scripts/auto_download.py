import gzip
import shutil
import time
from pathlib import Path

import requests

RAW_DIR = Path("raw_data")
RAW_DIR.mkdir(exist_ok=True)

# jmcauley.ucsd.edu URL is permanently down; Stanford SNAP is the working mirror
AMAZON_URLS = [
    "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz",
]

GOODREADS_URLS = {
    "ratings.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
    "books.csv": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv",
}

CHUNK_SIZE = 65536
REPORT_EVERY = 10 * 1024 * 1024  # 10 MB
CONNECT_TIMEOUT = 15  # seconds — fail fast on dead hosts
READ_TIMEOUT = 120    # seconds per chunk — generous for slow servers


def _download_single(url: str, dest: Path) -> bool:
    """Download a single URL to dest with chunked reads. Returns True on success."""
    headers = {"User-Agent": "Mozilla/5.0"}
    with requests.get(url, headers=headers, stream=True,
                      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = REPORT_EVERY
        with open(dest, "wb") as out:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(f"    Downloaded {downloaded / 1024 / 1024:.1f} MB", flush=True)
                    next_report += REPORT_EVERY
    if total_size and downloaded < total_size:
        raise RuntimeError(
            f"Download truncated: received {downloaded / 1024 / 1024:.1f} MB "
            f"of {total_size / 1024 / 1024:.1f} MB"
        )
    print(f"    Finished: {dest.name} ({downloaded / 1024 / 1024:.1f} MB)")
    return True


def download_with_retry(urls, dest: Path) -> bool:
    """Try each URL in order, 3 attempts each, with exponential backoff."""
    for url in urls:
        print(f"  Trying {url}")
        for attempt in range(1, 4):
            try:
                return _download_single(url, dest)
            except Exception as e:
                print(f"    Attempt {attempt}/3 failed: {e}")
                if dest.exists():
                    dest.unlink()  # remove partial file before retry
                if attempt < 3:
                    time.sleep(2 ** attempt)
    return False


def main():
    amazon_ok = False
    amazon_dir = RAW_DIR / "amazon"
    amazon_dir.mkdir(parents=True, exist_ok=True)
    amazon_gz = amazon_dir / "reviews_Movies_and_TV_5.json.gz"
    amazon_json = amazon_dir / "reviews_Movies_and_TV_5.json"

    if amazon_json.exists():
        print("Amazon JSON already exists, skipping download.")
        amazon_ok = True
    else:
        print("Downloading Amazon reviews (Movies & TV 5-core)...")
        if download_with_retry(AMAZON_URLS, amazon_gz):
            print("Extracting Amazon reviews...")
            try:
                with gzip.open(amazon_gz, "rb") as f_in:
                    with open(amazon_json, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print("Extraction done.")
                amazon_ok = True
            except Exception as e:
                print(f"[WARN] Extraction failed: {e}")
                # Remove partial output so the next run re-downloads cleanly
                for stale in (amazon_json, amazon_gz):
                    if stale.exists():
                        stale.unlink()
                amazon_ok = False
        else:
            print("[WARN] Amazon failed after all retries. Continuing with Goodreads + Nigerian only.")
            amazon_ok = False

    goodreads_dir = RAW_DIR / "goodreads"
    goodreads_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in GOODREADS_URLS.items():
        dest = goodreads_dir / filename
        if dest.exists():
            print(f"Goodreads {filename} already exists, skipping.")
            continue
        print(f"Downloading Goodreads {filename}...")
        ok = download_with_retry([url], dest)
        if not ok:
            raise RuntimeError(f"Failed to download Goodreads {filename} after all retries.")

    print("All available datasets downloaded.")
    return amazon_ok


if __name__ == "__main__":
    main()
