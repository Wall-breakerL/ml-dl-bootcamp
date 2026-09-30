"""Only bookkeeping lives here; training remains visible in each lab."""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fixed seeds reduce variation; they do not guarantee cross-device bit equality.


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def save_run(name, args, metrics, history=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path("runs") / f"{stamp}-{name}"
    out.mkdir(parents=True)
    record = {"experiment": name, "args": vars(args), "metrics": metrics,
              "python": sys.version.split()[0], "torch": torch.__version__,
              "cuda_runtime": torch.version.cuda, "utc": stamp}
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    (out / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    if history:
        (out / "history.json").write_text(json.dumps(history, indent=2))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key in history[0]:
            if key != "epoch":
                plt.plot([h["epoch"] for h in history], [h[key] for h in history], label=key)
        plt.xlabel("epoch / step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "curves.png", dpi=140)
        plt.close()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("saved:", out)
    return out
