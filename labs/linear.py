"""D1: manual MSE gradient -> autograd check -> SGD -> least-squares reference.

Teaching adaptation of D2L 2.5, 3.1-3.3. No d2l helper functions.
"""
import argparse
import torch
from .common import seed_all, save_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    seed_all(a.seed)
    # X:[N,2], w:[2,1], b:[1]; y must remain [N,1] to avoid broadcasting bugs.
    X = torch.randn(1000, 2, dtype=torch.float64)
    true_w = torch.tensor([[2.0], [-3.4]], dtype=X.dtype)
    y = X @ true_w + 4.2 + 0.01 * torch.randn(1000, 1, dtype=X.dtype)
    train_X, val_X = X[:800], X[800:]
    train_y, val_y = y[:800], y[800:]
    w = torch.zeros(2, 1, dtype=X.dtype, requires_grad=True)
    b = torch.zeros(1, dtype=X.dtype, requires_grad=True)
    loss = ((train_X @ w + b - train_y) ** 2).mean()
    loss.backward()
    manual = 2 * train_X.T @ (train_X @ w + b - train_y) / len(train_X)
    grad_error = (manual - w.grad).abs().max().item()
    assert grad_error < 1e-10
    w.grad.zero_(); b.grad.zero_()
    history = []
    for step in range(a.steps):
        loss = ((train_X @ w + b - train_y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= a.lr * w.grad
            b -= a.lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
            val_loss = ((val_X @ w + b - val_y) ** 2).mean().item()
        history.append({"epoch": step + 1, "train_mse": loss.item(), "val_mse": val_loss})
    design = torch.cat([train_X, torch.ones(800, 1, dtype=X.dtype)], dim=1)
    reference = torch.linalg.lstsq(design, train_y).solution
    assert torch.isfinite(w).all(), "Divergence: inspect learning rate."
    save_run("linear", a, {"w": w.detach().flatten().tolist(), "b": b.item(),
        "val_mse": val_loss, "manual_autograd_max_error": grad_error,
        "least_squares": reference.flatten().tolist()}, history)


if __name__ == "__main__":
    main()
