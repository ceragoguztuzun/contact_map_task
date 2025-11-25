import torch
from dnalongbench.utils import load_data

ROOT = "/usr/homes/cxo147/DNALongBench/dnalongbench_data/contact_map_prediction/"

def pcc(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> float:
    # pred/true: (B, D)
    pred = pred.float()
    true = true.float()
    pred = pred - pred.mean(dim=1, keepdim=True)
    true = true - true.mean(dim=1, keepdim=True)
    num = (pred * true).sum(dim=1)
    den = pred.norm(dim=1) * true.norm(dim=1) + eps
    return (num / den).mean().item()

def main():
    train_loader, valid_loader, test_loader = load_data(
        root=ROOT,
        task_name="contact_map_prediction",
        subset="HFF",
        batch_size=1,
    )

    x, y = next(iter(test_loader))
    print("x:", x.shape, x.dtype)
    print("y:", y.shape, y.dtype)

    # dummy predictions: all zeros, same shape as y
    pred = torch.zeros_like(y)

    print("Dummy PCC:", pcc(pred, y))

if __name__ == "__main__":
    main()