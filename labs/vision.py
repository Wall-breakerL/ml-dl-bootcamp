"""D2/D3: same split and training loop for softmax, MLP, LeNet, a tiny residual CNN.

Fashion-MNIST, 54k train / 6k validation. Test set is only opened with --test.
These are teaching adaptations, not claims of paper-level reproductions.
"""
import argparse
import copy
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from .common import seed_all, device_for, save_run


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return torch.relu(x + self.branch(x))


def make_model(name, dropout=0.0):
    if name == "softmax":
        # Return logits. CrossEntropyLoss includes log_softmax; no extra softmax.
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                             nn.Dropout(dropout), nn.Linear(256, 10))
    if name == "lenet":
        # 28 -> 28 -> 14 -> 10 -> 5; D2L's sigmoid/average-pooling form.
        return nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2),
            nn.Flatten(), nn.Linear(16*5*5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
    return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), Residual(16),
                         nn.AvgPool2d(2), Residual(16), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Linear(16, 10))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    targets, predictions = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        count += len(y)
        targets.extend(y.cpu().tolist()); predictions.extend(pred.cpu().tolist())
    return loss_sum/count, correct/count, targets, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["softmax", "mlp", "lenet", "residual"], default="mlp")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--train-size", type=int, default=0, help="0 uses full train split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="512 train/256 val, 1 epoch; not a quality result")
    p.add_argument("--test", action="store_true", help="Only after model/hyperparameters are frozen")
    a = p.parse_args()
    if a.smoke and a.test:
        p.error("Smoke runs cannot evaluate the final test set.")
    if a.epochs < 1 or a.train_size < 0 or a.train_size > 54000:
        p.error("epochs >= 1 and 0 <= train-size <= 54000 required")
    seed_all(a.seed)
    device = device_for(a.device)
    data = datasets.FashionMNIST("data", train=True, download=False, transform=transforms.ToTensor())
    train, val = random_split(data, [54000, 6000], generator=torch.Generator().manual_seed(42))
    if a.train_size:
        train = Subset(train, range(a.train_size))
    if a.smoke:
        train = Subset(train, range(min(512, len(train))))
        val = Subset(val, range(256))
        a.epochs = 1
    train_loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=a.batch_size, num_workers=0)
    model = make_model(a.model, a.dropout).to(device)
    cls = torch.optim.SGD if a.optimizer == "sgd" else torch.optim.Adam
    optimizer = cls(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    history, best_state, best_val = [], None, -1
    for epoch in range(a.epochs):
        model.train()
        total_loss = correct = count = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = nn.functional.cross_entropy(logits, y)
            assert torch.isfinite(loss), "Non-finite loss: check learning rate and data."
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(y)
            correct += (logits.argmax(1) == y).sum().item(); count += len(y)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch+1, "train_loss": total_loss/count,
            "val_loss": val_loss, "train_acc": correct/count, "val_acc": val_acc})
        print(json.dumps(history[-1]), flush=True)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    metrics = {"validation_accuracy": best_val, "n_train": len(train), "n_val": len(val),
        "device": str(device), "split_seed": 42, "smoke_only": a.smoke,
        "parameters": sum(p.numel() for p in model.parameters())}
    if a.test:
        from sklearn.metrics import confusion_matrix, f1_score
        test = datasets.FashionMNIST("data", train=False, download=False, transform=transforms.ToTensor())
        test_loss, test_acc, y, pred = evaluate(model, DataLoader(test, batch_size=256), device)
        metrics.update(test_loss=test_loss, test_accuracy=test_acc,
            test_macro_f1=f1_score(y, pred, average="macro"),
            confusion_matrix=confusion_matrix(y, pred).tolist())
    out = save_run("vision-"+a.model, a, metrics, history)
    torch.save({"model": a.model, "dropout": a.dropout,
                "state_dict": {k:v.cpu() for k,v in best_state.items()}}, out/"model.pt")


if __name__ == "__main__":
    main()
