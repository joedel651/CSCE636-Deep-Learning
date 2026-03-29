       """
generate_extra_data.py
======================
Simple fast data generator for the 3 hardest combos:
    (9,4,5), (9,5,4), (9,6,3)

Uses uniform(-100, 100) to match the original training data exactly.
(Verified: original P matrices have abs mean ~35-39, max=100)

Usage:
    python3 generate_extra_data.py
    python3 generate_extra_data.py --plot
"""

import numpy as np
import pickle
import time
import os
import argparse
from itertools import combinations

TARGET_COMBOS = [
    (9, 4, 5),
    (9, 5, 4),
    (9, 6, 3),
]

SAMPLES_PER_COMBO = 10000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_height_c11(n, k, P):
    r = n - k
    H = np.hstack([-P.T, np.eye(r)])
    col_indices = list(range(n))
    best = 0.0
    for S in combinations(col_indices, r):
        S = list(S)
        H_S = H[:, S]
        if abs(np.linalg.det(H_S)) < 1e-10:
            continue
        H_S_inv = np.linalg.inv(H_S)
        S_bar = [j for j in col_indices if j not in S]
        H_Sbar = H[:, S_bar]
        for idx in range(r):
            val = np.sum(np.abs(H_S_inv[idx, :] @ H_Sbar))
            if val > best:
                best = val
    return float(best)


def generate_combo(n, k, m, num_samples, rng):
    r = n - k
    samples = []
    heights = []
    t0 = time.time()

    for i in range(num_samples):
        P = rng.uniform(-100, 100, size=(k, r))
        h = compute_height_c11(n, k, P)

        if h <= 0 or np.isinf(h):
            continue
        # Cap to match original training data range
        if np.log2(h) > 23.0:
            continue

        samples.append([n, k, m, P])
        heights.append(float(h))

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (num_samples - i - 1)
            print(f"  ({n},{k},{m}): {i+1}/{num_samples}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    log2h = np.log2(np.array(heights))
    print(f"  ({n},{k},{m}): DONE — {len(samples)} samples in {elapsed:.0f}s  "
          f"log2h mean={log2h.mean():.2f}  std={log2h.std():.2f}  "
          f"min={log2h.min():.2f}  max={log2h.max():.2f}")

    return samples, heights


def plot_results(all_samples, all_heights):
    import matplotlib.pyplot as plt

    colors = ['#e74c3c', '#8e44ad', '#e67e22']
    orig_means = {(9,4,5): 13.39, (9,5,4): 13.70, (9,6,3): 12.74}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Extra Data — log₂(m-height) Distribution\n"
                 "Should match original (uniform -100 to 100)",
                 fontsize=12, fontweight='bold')

    for ax, (n, k, m), color in zip(axes, TARGET_COMBOS, colors):
        combo_heights = [h for h, s in zip(all_heights, all_samples)
                         if s[0]==n and s[1]==k and s[2]==m]
        if not combo_heights:
            ax.set_title(f"({n},{k},{m}) — no data")
            continue

        log2h = np.log2(np.array(combo_heights))
        ax.hist(log2h, bins=50, color=color, edgecolor='white',
                linewidth=0.4, alpha=0.85)
        ax.axvline(log2h.mean(), color='black', linestyle='--',
                   linewidth=1.5, label=f'new mean={log2h.mean():.1f}')
        ax.axvline(orig_means[(n,k,m)], color='#27ae60', linestyle=':',
                   linewidth=1.5, label=f'orig mean={orig_means[(n,k,m)]:.1f}')
        ax.set_title(
            f"({n},{k},{m})  n={len(combo_heights):,}\n"
            f"mean={log2h.mean():.2f}  std={log2h.std():.2f}  "
            f"min={log2h.min():.2f}",
            fontsize=9, color=color)
        ax.set_xlabel("log₂(height)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'extra_data_histogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true',
                        help='Plot histogram after generation')
    args = parser.parse_args()

    rng = np.random.default_rng(12345)

    print(f"\n{'='*55}")
    print(f"Generating {SAMPLES_PER_COMBO:,} samples each for:")
    for n, k, m in TARGET_COMBOS:
        print(f"  ({n},{k},{m})")
    print(f"Total: {SAMPLES_PER_COMBO * len(TARGET_COMBOS):,} samples")
    print(f"P scale: uniform(-100, 100) — matches original training data")
    print(f"{'='*55}\n")

    all_samples = []
    all_heights = []

    for n, k, m in TARGET_COMBOS:
        print(f"\nGenerating ({n},{k},{m})...")
        s, h = generate_combo(n, k, m, SAMPLES_PER_COMBO, rng)
        all_samples.extend(s)
        all_heights.extend(h)

    data_path   = os.path.join(OUTPUT_DIR, 'extra_data.pkl')
    height_path = os.path.join(OUTPUT_DIR, 'extra_heights.pkl')

    with open(data_path, 'wb') as f:
        pickle.dump(all_samples, f)
    with open(height_path, 'wb') as f:
        pickle.dump(all_heights, f)

    h_arr = np.array(all_heights)
    print(f"\n{'='*55}")
    print(f"Done! {len(all_samples):,} samples saved.")
    print(f"  log2(h) mean:  {np.log2(h_arr).mean():.2f}")
    print(f"  log2(h) std:   {np.log2(h_arr).std():.2f}")
    print(f"  log2(h) range: [{np.log2(h_arr.min()):.2f}, {np.log2(h_arr.max()):.2f}]")
    print(f"\nMerge in your notebook:")
    print(f"  with open('extra_data.pkl', 'rb') as f:")
    print(f"      extra_data = pickle.load(f)")
    print(f"  with open('extra_heights.pkl', 'rb') as f:")
    print(f"      extra_heights = pickle.load(f)")
    print(f"  train_data    = train_data    + extra_data")
    print(f"  train_heights = train_heights + extra_heights")
    print(f"{'='*55}\n")

    if args.plot:
        print("Generating histogram...")
        plot_results(all_samples, all_heights)


if __name__ == '__main__':
    main()
