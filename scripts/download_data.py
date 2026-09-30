"""Cache Fashion-MNIST from its authors' GitHub, checking torchvision's MD5s."""
import gzip
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path
from torchvision.datasets import FashionMNIST

raw = Path("data/FashionMNIST/raw")
raw.mkdir(parents=True, exist_ok=True)
base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
for name, expected in FashionMNIST.resources:
    target = raw / name
    if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != expected:
        temporary = target.with_suffix(".download")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(base + name, timeout=30) as src, temporary.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                print("Retry", name, flush=True)
                time.sleep(2)
        assert hashlib.md5(temporary.read_bytes()).hexdigest() == expected, name
        temporary.replace(target)
    with gzip.open(target, "rb") as src, target.with_suffix("").open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(name, "checksum OK", flush=True)
for train in (True, False):
    ds = FashionMNIST("data", train=train, download=False)
    print("train" if train else "test", len(ds))
