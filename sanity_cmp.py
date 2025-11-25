from dnalongbench.utils import load_data

root = "/usr/homes/cxo147/DNALongBench/dnalongbench_data/contact_map_prediction/"

train_loader, valid_loader, test_loader = load_data(
    root=root,
    task_name="contact_map_prediction",
    subset="HFF",
    batch_size=1,
)

x, y = next(iter(train_loader))
print("Train batch:")
print("  x:", x.shape, x.dtype)
print("  y:", y.shape, y.dtype)

xv, yv = next(iter(valid_loader))
print("Valid batch:")
print("  x:", xv.shape, xv.dtype)
print("  y:", yv.shape, yv.dtype)

xt, yt = next(iter(test_loader))
print("Test batch:")
print("  x:", xt.shape, xt.dtype)
print("  y:", yt.shape, yt.dtype)
