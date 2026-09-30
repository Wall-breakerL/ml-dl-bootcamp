"""D5: RNN/GRU on chronological synthetic signal; compare with last-value baseline."""
import argparse
import copy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from .common import seed_all, device_for, save_run


def windows(series, width=24):
    # Split RAW series first so windows cannot share observations across splits.
    chunks = series.unfold(0, width+1, 1)
    return chunks[:, :-1, None].contiguous(), chunks[:, -1, None].contiguous()


class Forecast(nn.Module):
    def __init__(self, kind):
        super().__init__()
        cls = nn.GRU if kind == "gru" else nn.RNN
        self.rnn = cls(input_size=1, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        hidden, _ = self.rnn(x)
        return self.head(hidden[:, -1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rnn", "gru"], default="gru")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--test", action="store_true")
    a = p.parse_args(); seed_all(a.seed)
    if a.smoke and a.test: p.error("Smoke runs cannot evaluate the test split")
    if a.epochs < 1: p.error("epochs >= 1 required")
    device = device_for(a.device)
    t = torch.arange(6000, dtype=torch.float32)
    signal = torch.sin(t*0.035) + 0.3*torch.sin(t*0.08) + torch.randn(6000)*0.04
    X, y = windows(signal[:3600]); Xv, yv = windows(signal[3600:4800])
    if a.smoke: X,y = X[:256],y[:256]; a.epochs = 1
    loader = DataLoader(TensorDataset(X,y), batch_size=128, shuffle=True)
    Xv,yv = Xv.to(device),yv.to(device)
    model = Forecast(a.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    history, best, state = [], float("inf"), None
    for epoch in range(a.epochs):
        model.train(); total = 0
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = (model(xb)-yb).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()*len(yb)
        model.eval()
        with torch.no_grad(): val = (model(Xv)-yv).square().mean().item()
        history.append({"epoch":epoch+1,"train_mse":total/len(y),"val_mse":val})
        if val < best: best,state=val,copy.deepcopy(model.state_dict())
    metrics={"validation_rmse":best**0.5,"smoke_only":a.smoke,"device":str(device),
             "validation_last_value_rmse":(Xv[:,-1]-yv).square().mean().sqrt().item()}
    if a.test:
        model.load_state_dict(state)
        Xt,yt=windows(signal[4800:]); Xt,yt=Xt.to(device),yt.to(device)
        with torch.no_grad(): metrics["test_rmse"]=(model(Xt)-yt).square().mean().sqrt().item()
        metrics["test_last_value_rmse"]=(Xt[:,-1]-yt).square().mean().sqrt().item()
    save_run("sequence-"+a.model,a,metrics,history)


if __name__ == "__main__":
    main()
