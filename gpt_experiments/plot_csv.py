import argparse
import csv
from typing import List, Tuple
import os

import matplotlib.pyplot as plt


def read_csv(path: str):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    data = rows[1:]
    return header, data


def parse_values(header: List[str], data: List[List[str]]):
    # Supports either [idx, mse_baseline, mse_after_delta]
    # or [layer, idx, mse_baseline, mse_after_delta]
    has_layer = header[0].strip().lower() == "layer"
    idx_col = 1 if has_layer else 0
    base_col = 2 if has_layer else 1
    after_col = 3 if has_layer else 2
    layers = []
    idxs = []
    baseline = []
    after = []
    for r in data:
        if not r:
            continue
        if has_layer:
            try:
                layers.append(int(r[0]))
            except Exception:
                continue
        try:
            idxs.append(int(r[idx_col]))
            baseline.append(float(r[base_col]))
            after.append(float(r[after_col]))
        except Exception:
            continue
    return has_layer, layers, idxs, baseline, after


def plot_per_prompt(idxs: List[int], baseline: List[float], after: List[float], out_path: str, title: str):
    plt.figure(figsize=(10, 4))
    plt.plot(idxs, baseline, label="baseline", marker="o", linewidth=1)
    plt.plot(idxs, after, label="after ΔW", marker="o", linewidth=1)
    plt.xlabel("prompt_idx")
    plt.ylabel("logits_mse")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_hist_drop(baseline: List[float], after: List[float], out_path: str, title: str):
    import numpy as np
    drops = np.array(baseline) - np.array(after)
    plt.figure(figsize=(6, 4))
    plt.hist(drops, bins=20, color="#4e79a7")
    plt.xlabel("MSE drop (baseline - after)")
    plt.ylabel("count")
    plt.title(title + " — drops")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Context vs ΔW (logits MSE)")
    args = ap.parse_args()

    header, data = read_csv(args.input)
    has_layer, layers, idxs, baseline, after = parse_values(header, data)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plot_per_prompt(idxs, baseline, after, args.out, args.title)
    base, ext = os.path.splitext(args.out)
    plot_hist_drop(baseline, after, base + "_drops" + ext, args.title)


if __name__ == "__main__":
    main()
