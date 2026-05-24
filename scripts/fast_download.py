"""
Parallel chunk downloader. Splits the file into N chunks and downloads
each in its own thread, then reassembles. Much faster than single-connection
wget when the server throttles per-connection.
"""
import os
import sys
import threading
import time
import requests
from pathlib import Path

URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz"
DEST = Path("raw_data/amazon/reviews_Movies_and_TV_5.json.gz")
CONNECTIONS = 8
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120


def get_file_size(url):
    r = requests.head(url, timeout=(CONNECT_TIMEOUT, 30), headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    size = int(r.headers.get("Content-Length", 0))
    if not size:
        raise RuntimeError("Server did not return Content-Length — cannot chunk.")
    return size


def download_chunk(url, start, end, buf, idx, progress):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Range": f"bytes={start}-{end}",
    }
    r = requests.get(url, headers=headers, stream=True,
                     timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()
    pos = start
    for chunk in r.iter_content(chunk_size=65536):
        if chunk:
            buf[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
            progress[idx] = pos - start
    progress[idx] = end - start + 1


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checking file size...")
    total = get_file_size(URL)
    print(f"File size: {total / 1024 / 1024:.1f} MB — splitting into {CONNECTIONS} chunks")

    chunk_size = total // CONNECTIONS
    ranges = []
    for i in range(CONNECTIONS):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < CONNECTIONS - 1 else total - 1
        ranges.append((start, end))

    buf = bytearray(total)
    progress = [0] * CONNECTIONS
    threads = []

    start_time = time.time()

    for i, (s, e) in enumerate(ranges):
        t = threading.Thread(target=download_chunk, args=(URL, s, e, buf, i, progress), daemon=True)
        t.start()
        threads.append(t)

    print("Downloading... (press Ctrl+C to cancel)")
    while any(t.is_alive() for t in threads):
        downloaded = sum(progress)
        elapsed = time.time() - start_time
        speed = downloaded / elapsed if elapsed > 0 else 0
        pct = downloaded / total * 100
        eta = (total - downloaded) / speed if speed > 0 else 0
        print(f"\r  {pct:.1f}%  {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB  "
              f"{speed/1024:.1f} KB/s  ETA: {eta/60:.1f} min   ", end="", flush=True)
        time.sleep(2)

    for t in threads:
        t.join()

    downloaded = sum(progress)
    if downloaded < total:
        print(f"\nERROR: Only got {downloaded}/{total} bytes.")
        sys.exit(1)

    print(f"\nWriting {total / 1024 / 1024:.1f} MB to {DEST}...")
    with open(DEST, "wb") as f:
        f.write(buf)
    print("Done.")


if __name__ == "__main__":
    main()
