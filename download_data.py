"""Download the pinned public Parquet mirror of the Home Credit competition tables."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "Lixiaoxiofficial/home-credit-default-risk"
REVISION = "b79b5e7c5bdd2f7ff4a7d53ef4b44bcfdc88823e"
TABLES = (
    "application_train_dated",
    "bureau",
    "bureau_balance",
    "previous_application",
    "installments_payments",
    "credit_card_balance",
    "POS_CASH_balance",
)


def open_url(url):
    return urlopen(
        Request(url, headers={"User-Agent": "home-credit-portfolio/1.0"}), timeout=95
    )


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_table(directory, table):
    listing = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{table}"
    with open_url(listing) as response:
        shards = [f for f in json.load(response) if f["path"].endswith(".parquet")]
    if not shards:
        raise RuntimeError(f"No Parquet shards for {table}")
    for shard in shards:
        target = directory / table / Path(shard["path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = shard["lfs"]["oid"]
        if (
            target.exists()
            and target.stat().st_size == shard["size"]
            and file_hash(target) == expected
        ):
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(shard['path'])}"
        temporary = target.with_suffix(".part")
        for attempt in range(3):
            try:
                with open_url(url) as source, temporary.open("wb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)
                if (
                    temporary.stat().st_size != shard["size"]
                    or file_hash(temporary) != expected
                ):
                    raise OSError("Checksum or size mismatch")
                temporary.replace(target)
                break
            except OSError:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    print(f"{table}: ready", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("data"))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for _ in pool.map(lambda name: fetch_table(args.directory, name), TABLES):
            pass


if __name__ == "__main__":
    main()
