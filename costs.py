import numpy as np

import common as C

BYTES_PER_WINDOW = 150 * 3 * 2


def macs(model, length=150):
    total = 0
    for conv in (model.c1, model.c2, model.c3):
        length = (length + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        total += conv.in_channels * conv.out_channels * conv.kernel_size[0] * length
    return total + model.head.in_features * model.head.out_features


def main():
    m = C.GateCNN()
    n = sum(p.numel() for p in m.parameters())
    print(f"parameters {n}  fp32 bytes {4 * n}  MACs per window {macs(m):,}")
    print(f"bytes per participant, up + down: 15 rounds {15 * 2 * 4 * n / 1e6:.2f} MB, 60 rounds {60 * 2 * 4 * n / 1e6:.2f} MB")
    for name, dev in (("unimib", None), ("umafall", None), ("fallalld", 1), ("kfall", None)):
        D = C.load_site(name, dev)
        counts = np.bincount(D["subj"])
        counts = counts[counts > 0]
        print(f"{name}{'' if dev is None else '_D' + str(dev)}: median triggered windows per user {np.median(counts):.0f}, "
              f"median raw upload {np.median(counts) * BYTES_PER_WINDOW / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
