"""Validate imports AND a CUDA forward/backward pass, without a training job."""
import json
import platform
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision


def main():
    assert torch.cuda.is_available(), "This server check requires a working CUDA GPU."
    x = torch.randn(32, 8, device="cuda")
    model = torch.nn.Linear(8, 1).cuda()
    loss = model(x).square().mean()
    loss.backward()
    assert torch.isfinite(model.weight.grad).all()
    torch.cuda.synchronize()
    print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_forward_backward": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
