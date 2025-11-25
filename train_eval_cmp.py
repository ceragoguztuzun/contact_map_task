#!/usr/bin/env python3
"""
train_eval_cmp.py - Minimal DNALongBench Contact Map trainer/evaluator

Usage:
  # Train and evaluate
  python train_eval_cmp.py --model "my_models:MyTransformer" --steps 1000
  
  # Eval only (load checkpoint)
  python train_eval_cmp.py --model "my_models:MyTransformer" --eval_only --checkpoint model.pt
"""
import argparse
import importlib
import torch
import torch.nn as nn
from dnalongbench.utils import load_data


# === Metrics ===

def pearson_corr(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Pearson correlation averaged across batch."""
    pred = pred.float() - pred.mean(dim=1, keepdim=True)
    true = true.float() - true.mean(dim=1, keepdim=True)
    num = (pred * true).sum(dim=1)
    den = pred.norm(dim=1) * true.norm(dim=1) + 1e-8
    return (num / den).mean().item()

def scc_hicrep_like(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Distance-stratified correlation (HiCRep-inspired)."""
    pred, true = pred.float(), true.float()
    
    triu = torch.triu_indices(448, 448, offset=2, device=pred.device)
    dists = triu[1] - triu[0]
    
    corrs, weights = [], []
    for d in range(2, 448):
        mask = (dists == d)
        n_d = mask.sum().item()
        
        # Skip if too few elements (need at least 2 for correlation)
        if n_d < 2:
            continue
        
        pd = pred[:, mask]
        td = true[:, mask]
        
        # Center
        pd = pd - pd.mean(dim=1, keepdim=True)
        td = td - td.mean(dim=1, keepdim=True)
        
        # Pearson correlation
        pd_norm = pd.norm(dim=1)
        td_norm = td.norm(dim=1)
        
        # Skip if either has zero variance
        if (pd_norm < 1e-10).any() or (td_norm < 1e-10).any():
            r = torch.zeros(pred.shape[0], device=pred.device)
        else:
            r = (pd * td).sum(dim=1) / (pd_norm * td_norm)
        
        # Ranks for weighting
        pr = pd.argsort(dim=1).argsort(dim=1).float()
        tr = td.argsort(dim=1).argsort(dim=1).float()
        
        # Normalize ranks to [0, 1]
        if n_d > 1:
            pr = pr / (n_d - 1)
            tr = tr / (n_d - 1)
        
        # Weight: variance in ranks * number of pairs
        w = pr.std(dim=1) * tr.std(dim=1) * n_d
        
        corrs.append(r)
        weights.append(w)
    
    if not corrs:
        return float('nan')
    
    corrs = torch.stack(corrs)  # (K, B)
    weights = torch.stack(weights)  # (K, B)
    
    # Weighted average across distances, then across batch
    weights_sum = weights.sum(0)  # (B,)
    
    # Handle case where all weights are zero
    if (weights_sum < 1e-10).all():
        return 0.0
    
    result = (weights * corrs).sum(0) / (weights_sum + 1e-10)  # (B,)
    return result.mean().item()

# === Training ===

def train(model, train_loader, valid_loader, test_loader, args):
    """Train model and evaluate."""
    device = args.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    print(f"\n=== Training for {args.steps} steps ===")
    
    train_iter = iter(train_loader)
    for step in range(args.steps):
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x = x.to(device)
        y = y.to(device).float()
        
        # Forward + backward
        model.train()
        pred = model(x)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log
        if step % args.log_every == 0:
            with torch.no_grad():
                pcc = pearson_corr(pred, y)
                scc = scc_hicrep_like(pred, y)
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | PCC: {pcc:.4f} | SCC: {scc:.4f}")
    
    # Save checkpoint
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"\n✓ Saved model to {args.save}")
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    eval_model(model, valid_loader, test_loader, device, args.eval_batches)


@torch.no_grad()
def eval_model(model, valid_loader, test_loader, device, num_batches=None):
    """Evaluate model on valid and test sets."""
    model.eval()
    
    for name, loader in [("Valid", valid_loader), ("Test", test_loader)]:
        pccs, sccs = [], []
        for i, (x, y) in enumerate(loader):
            if num_batches and i >= num_batches:
                break
            x = x.to(device)
            y = y.to(device).float()
            pred = model(x)
            pccs.append(pearson_corr(pred, y))
            sccs.append(scc_hicrep_like(pred, y))
        
        pcc = sum(pccs) / len(pccs) if pccs else 0
        scc = sum(sccs) / len(sccs) if sccs else 0
        print(f"{name:5s} | PCC: {pcc:.4f} | SCC: {scc:.4f}")


# === Main ===

def load_model(spec: str) -> nn.Module:
    """Load model from 'module.path:ClassName' string."""
    if ':' not in spec:
        raise ValueError(f"Model must be 'module:Class', got: {spec}")
    mod_name, cls_name = spec.split(':', 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)()


def main():
    parser = argparse.ArgumentParser(description='Train/Eval CMP model')
    
    # Model
    parser.add_argument('--model', required=True, help='module:ClassName')
    parser.add_argument('--checkpoint', help='Load weights from checkpoint')
    parser.add_argument('--save', default='model.pt', help='Save checkpoint path')
    
    # Data
    parser.add_argument('--root', default='/usr/homes/cxo147/DNALongBench/dnalongbench_data/contact_map_prediction/')
    parser.add_argument('--subset', default='HFF')
    parser.add_argument('--batch_size', type=int, default=1)
    
    # Training
    parser.add_argument('--eval_only', action='store_true', help='Skip training')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_every', type=int, default=50)
    
    # Eval
    parser.add_argument('--eval_batches', type=int, default=None, help='Limit eval batches')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load data
    root = args.root if args.root.endswith('/') else args.root + '/'
    print(f"Loading {args.subset} data from {root}")
    train_loader, valid_loader, test_loader = load_data(
        root=root,
        task_name='contact_map_prediction',
        subset=args.subset,
        batch_size=args.batch_size,
    )
    
    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model).to(args.device)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    
    # Shape check
    x, y = next(iter(train_loader))
    pred = model(x.to(args.device))
    print(f"✓ Input: {tuple(x.shape)}, Output: {tuple(pred.shape)}")
    assert pred.shape == (args.batch_size, 99681), f"Wrong output shape: {pred.shape}"
    
    # Train or eval
    if args.eval_only:
        print("\n=== Evaluation Only ===")
        eval_model(model, valid_loader, test_loader, args.device, args.eval_batches)
    else:
        train(model, train_loader, valid_loader, test_loader, args)


if __name__ == '__main__':
    main()