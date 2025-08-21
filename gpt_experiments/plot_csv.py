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


def plot_scatter(baseline: List[float], after: List[float], out_path: str, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(baseline, after, s=24, alpha=0.8, edgecolors='none')
    mn = min(min(baseline), min(after))
    mx = max(max(baseline), max(after))
    pad = 0.05 * (mx - mn if mx > mn else 1.0)
    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], 'r--', linewidth=1, label='y = x')
    plt.xlabel("baseline logits MSE")
    plt.ylabel("after ΔW logits MSE")
    plt.title(title + " — scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_layer_sweep(layers: List[int], baseline: List[float], after: List[float], out_path: str, title: str):
    import numpy as np
    if not layers:
        return
    layers_arr = np.array(layers)
    base = np.array(baseline)
    aft = np.array(after)
    drops = base - aft
    uniq = np.unique(layers_arr)
    mean_drop = []
    pct_imp = []
    for L in uniq:
        dL = drops[layers_arr == L]
        mean_drop.append(dL.mean() if dL.size else 0.0)
        pct_imp.append(100.0 * (dL > 0).mean() if dL.size else 0.0)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    ax1.plot(uniq, mean_drop, '-o', color="#4e79a7", label='mean drop')
    ax2.plot(uniq, pct_imp, '-s', color="#f28e2b", label='% improved')
    ax1.set_xlabel('layer')
    ax1.set_ylabel('mean MSE drop', color="#4e79a7")
    ax2.set_ylabel('% improved', color="#f28e2b")
    ax1.set_title(title + " — layer sweep")
    if len(mean_drop):
        best_idx = int(np.argmax(mean_drop))
        best_layer = int(uniq[best_idx])
        ax1.axvline(best_layer, color='gray', linestyle='--', linewidth=1)
        ax1.text(best_layer, mean_drop[best_idx], f" best {best_layer}", va='bottom')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def compute_stats(baseline: List[float], after: List[float]):
    import numpy as np
    base = np.array(baseline)
    aft = np.array(after)
    drops = base - aft
    n = drops.size
    pct_improved = 100.0 * (drops > 0).mean() if n else 0.0
    med = float(np.median(drops)) if n else 0.0
    mean = float(drops.mean()) if n else 0.0
    best = float(drops.max()) if n else 0.0
    return {
        "n": int(n),
        "pct_improved": pct_improved,
        "median": med,
        "mean": mean,
        "best": best,
    }


def plot_summary_card(stats: dict, out_path: str, title: str):
    plt.figure(figsize=(8, 4))
    plt.axis('off')
    lines = [
        title,
        f"n = {stats['n']} prompts",
        f"{stats['pct_improved']:.0f}% improved",
        f"median drop = {stats['median']:.1f}",
        f"mean drop = {stats['mean']:.1f}",
        f"best drop = {stats['best']:.1f}",
    ]
    y = 0.9
    sizes = [14, 11, 16, 12, 12, 11]
    weights = ['bold', 'normal', 'bold', 'normal', 'normal', 'normal']
    for i, text in enumerate(lines):
        plt.text(0.05, y, text, fontsize=sizes[i], fontweight=weights[i], va='top')
        y -= 0.15 if i == 1 else 0.12
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_perprompt(idxs: List[int], baseline: List[float], after: List[float], out_path: str, title: str):
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


def plot_dashboard(baseline: List[float], after: List[float], out_path: str, title: str):
    import numpy as np
    drops = np.array(baseline) - np.array(after)
    improved = (drops > 0)
    pct_improved = 100.0 * improved.mean() if drops.size > 0 else 0.0
    med_drop = float(np.median(drops)) if drops.size > 0 else 0.0
    mean_drop = float(drops.mean()) if drops.size > 0 else 0.0

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Scatter baseline vs after
    ax = axs[0]
    ax.scatter(baseline, after, s=18, alpha=0.8, edgecolors='none')
    if baseline and after:
        mn = min(min(baseline), min(after))
        mx = max(max(baseline), max(after))
    else:
        mn, mx = 0.0, 1.0
    pad = 0.05 * (mx - mn if mx > mn else 1.0)
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], 'r--', linewidth=1, label='y = x')
    ax.set_xlabel('baseline MSE'); ax.set_ylabel('after ΔW MSE')
    ax.set_title('Scatter (lower = better)')
    ax.legend()

    # 2) Sorted drops
    ax = axs[1]
    order = np.argsort(drops)[::-1]
    drops_sorted = drops[order]
    ax.bar(range(len(drops_sorted)), drops_sorted, color="#4e79a7")
    ax.axhline(med_drop, color='orange', linestyle='--', linewidth=1, label=f'median {med_drop:.1f}')
    ax.set_xlabel('prompts (sorted)'); ax.set_ylabel('MSE drop')
    ax.set_title('Improvements (baseline − after)')
    ax.legend()

    # 3) ECDF of drops
    ax = axs[2]
    x = np.sort(drops)
    y = np.arange(1, len(x) + 1) / len(x) if len(drops) > 0 else np.array([0])
    if len(x) > 0:
        ax.plot(x, y, color="#4e79a7")
        ax.axvline(med_drop, color='orange', linestyle='--', linewidth=1)
        ax.text(med_drop, 0.5, f'median={med_drop:.1f}', rotation=90, va='center', ha='right', fontsize=8)
    ax.set_xlabel('MSE drop'); ax.set_ylabel('ECDF')
    ax.set_title(f'ECDF — {pct_improved:.0f}% improved, mean={mean_drop:.1f}')

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ecdf_drops(baseline: List[float], after: List[float], out_path: str, title: str):
    import numpy as np
    drops = np.array(baseline) - np.array(after)
    x = np.sort(drops)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color="#4e79a7")
    plt.xlabel("MSE drop (baseline - after)")
    plt.ylabel("ECDF")
    plt.title(title + " — ECDF of drops")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_sortedbar_drops(baseline: List[float], after: List[float], out_path: str, title: str):
    import numpy as np
    drops = np.array(baseline) - np.array(after)
    order = np.argsort(drops)[::-1]
    drops_sorted = drops[order]
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(drops_sorted)), drops_sorted, color="#4e79a7")
    plt.xlabel("prompts (sorted by improvement)")
    plt.ylabel("MSE drop")
    plt.title(title + " — sorted drops")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Context vs ΔW (logits MSE)")
    ap.add_argument("--style", choices=["perprompt", "scatter", "sorted"], default="perprompt")
    args = ap.parse_args()

    header, data = read_csv(args.input)
    has_layer, layers, idxs, baseline, after = parse_values(header, data)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Main figure style
    if args.style == "perprompt":
        plot_perprompt(idxs, baseline, after, args.out, args.title)
    elif args.style == "scatter":
        plot_scatter(baseline, after, args.out, args.title)
    else:
        plot_sortedbar_drops(baseline, after, args.out, args.title)
    base, ext = os.path.splitext(args.out)
    plot_ecdf_drops(baseline, after, base + "_ecdf" + ext, args.title)
    plot_dashboard(baseline, after, base + "_dashboard" + ext, args.title)
    stats = compute_stats(baseline, after)
    plot_summary_card(stats, base + "_summary" + ext, args.title)
    if has_layer:
        plot_layer_sweep(layers, baseline, after, base + "_layers" + ext, args.title)


if __name__ == "__main__":
    main()
