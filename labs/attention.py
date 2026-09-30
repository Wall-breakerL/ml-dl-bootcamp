"""D5: manual causal attention + numerical check + tiny Transformer retrieval task.

The task is 'read a random token sequence, return its first token at the final
query position'. This is NOT language modelling or D2L machine translation.
"""
import argparse
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import seed_all, device_for, save_run


def attention(q, k, v):
    # [B,H,T,D] -> scores [B,H,T,T] -> weighted values [B,H,T,D]
    scores = q @ k.transpose(-2,-1) / math.sqrt(q.shape[-1])
    future = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).triu(1)
    weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return weights @ v, weights


class TinyTransformer(nn.Module):
    def __init__(self, width=32, heads=4, length=9):
        super().__init__()
        self.heads = heads
        self.embedding = nn.Embedding(11, width)
        self.position = nn.Parameter(torch.randn(1,length,width)*0.02)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width,width*3)
        self.out = nn.Linear(width,width)
        self.ffn = nn.Sequential(nn.Linear(width,width*2),nn.ReLU(),nn.Linear(width*2,width))
        self.classifier = nn.Linear(width,10)

    def forward(self, tokens):
        x = self.embedding(tokens) + self.position[:,:tokens.shape[1]]
        b,t,d = x.shape
        q,k,v = self.qkv(self.norm1(x)).chunk(3,dim=-1)
        q,k,v = [z.reshape(b,t,self.heads,d//self.heads).transpose(1,2) for z in (q,k,v)]
        context,_ = attention(q,k,v)
        x = x + self.out(context.transpose(1,2).reshape(b,t,d))
        x = x + self.ffn(self.norm2(x))
        return self.classifier(x[:,-1])


def make_data(n, generator, device):
    tokens = torch.randint(0,10,(n,8),generator=generator)
    targets = tokens[:,0].clone()
    tokens = torch.cat([tokens,torch.full((n,1),10)],dim=1)
    return tokens.to(device),targets.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="auto")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--test",action="store_true")
    a=p.parse_args(); seed_all(a.seed)
    if a.smoke and a.test: p.error("Smoke runs cannot evaluate the test split")
    if a.steps < 1: p.error("steps >= 1 required")
    device=device_for(a.device)
    q,k,v=[torch.randn(2,2,6,8,device=device) for _ in range(3)]
    custom,weights=attention(q,k,v)
    builtin=F.scaled_dot_product_attention(q,k,v,is_causal=True)
    error=(custom-builtin).abs().max().item()
    assert torch.allclose(custom,builtin,atol=2e-5,rtol=2e-5)
    changed_k,changed_v=k.clone(),v.clone()
    changed_k[:,:,3:]+=100; changed_v[:,:,3:]+=100
    changed,_=attention(q,changed_k,changed_v)
    assert torch.allclose(custom[:,:,:3],changed[:,:,:3],atol=1e-6)
    assert torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)))
    generator=torch.Generator().manual_seed(123)
    X,y=make_data(2048,generator,device); Xv,yv=make_data(512,generator,device)
    model=TinyTransformer().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.003)
    history,best,state=[], -1, None
    if a.smoke: a.steps=3
    for step in range(a.steps):
        model.train()
        idx=torch.randint(len(y),(64,),device=device)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(X[idx]),y[idx])
        loss.backward(); opt.step()
        if step%20==0 or step==a.steps-1:
            model.eval()
            with torch.no_grad(): acc=(model(Xv).argmax(-1)==yv).float().mean().item()
            history.append({"epoch":step+1,"train_loss":loss.item(),"val_acc":acc})
            if acc>best: best,state=acc,copy.deepcopy(model.state_dict())
    metrics={"manual_sdpa_max_error":error,"causal_perturbation":"PASS",
             "validation_accuracy":best,"chance_accuracy":0.1,"device":str(device),
             "task":"first_token_retrieval_not_language_modelling","smoke_only":a.smoke}
    if a.test:
        model.load_state_dict(state)
        Xt,yt=make_data(512,generator,device)
        with torch.no_grad(): metrics["test_accuracy"]=(model(Xt).argmax(-1)==yt).float().mean().item()
    save_run("attention",a,metrics,history)


if __name__=="__main__":
    main()
