"""
synthetic_data_gen_v5.py
========================
Targeted synthetic data generator for the 3 hardest (n,k,m) combos:
    (9,4,5) · (9,5,4) · (9,6,3)

V5 approach: scale-controlled inverse design via C11 binary search.
No LP required. Runs sequentially (no multiprocessing).
"""

import numpy as np
import pickle
import time
import os
import argparse
from itertools import combinations
from scipy.optimize import brentq


# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────

def checkpoint_path(output_dir, n, k, m):
    ckpt_dir = os.path.join(output_dir, 'checkpoints_v5')
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f'ckpt_n{n}_k{k}_m{m}.pkl')

def save_checkpoint(output_dir, n, k, m, samples, heights):
    with open(checkpoint_path(output_dir, n, k, m), 'wb') as f:
        pickle.dump({'samples': samples, 'heights': heights}, f)

def load_checkpoint(output_dir, n, k, m):
    path = checkpoint_path(output_dir, n, k, m)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['samples'], data['heights']
    return None, None

def checkpoint_exists(output_dir, n, k, m):
    return os.path.exists(checkpoint_path(output_dir, n, k, m))


# ── COROLLARY 11 HEIGHT COMPUTATION ──────────────────────────────────────────

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


# ── GEOMETRY-MATCHED P INITIALIZATION ────────────────────────────────────────

def generate_P_for_target(target_log2h, rng, k, r):
    if target_log2h < 8.0:
        eps = rng.uniform(0.3, 2.0)
    elif target_log2h < 12.0:
        eps = rng.uniform(1.0, 20.0)
    elif target_log2h < 17.0:
        eps = rng.uniform(10.0, 200.0)
    else:
        eps = rng.uniform(100.0, 10000.0)
    return rng.standard_normal((k, r)) * eps


# ── SCALE-TO-TARGET VIA BINARY SEARCH ────────────────────────────────────────

def find_min_scale_and_height(P_base, n, k, lo=-3.0, hi=5.0, n_grid=15):
    best_log10a, best_h = 0.0, np.inf
    for log10a in np.linspace(lo, hi, n_grid):
        h = compute_height_c11(n, k, P_base * (10 ** log10a))
        if h < best_h:
            best_h = h
            best_log10a = log10a
    return best_log10a, best_h


def scale_to_target(P_base, n, k, target_h, scale_lo=-3.0, scale_hi=5.0):
    opt_log10a, min_h = find_min_scale_and_height(
        P_base, n, k, lo=scale_lo, hi=scale_hi)

    if min_h > target_h:
        return None

    f = lambda a: compute_height_c11(n, k, P_base * (10 ** a)) - target_h

    # Left branch
    try:
        if f(scale_lo) * (min_h - target_h) < 0:
            log10a = brentq(f, scale_lo, opt_log10a, xtol=0.05, maxiter=25)
            return float(10 ** log10a)
    except Exception:
        pass

    # Right branch
    try:
        if f(scale_hi) * (min_h - target_h) < 0:
            log10a = brentq(f, opt_log10a, scale_hi, xtol=0.05, maxiter=25)
            return float(10 ** log10a)
    except Exception:
        pass

    return None


# ── BIN TRACKER ───────────────────────────────────────────────────────────────

def make_bin_tracker(min_log2h, max_log2h, n_bins=18):
    edges = np.linspace(min_log2h, max_log2h, n_bins + 1)
    return {'edges': edges, 'counts': np.zeros(n_bins, dtype=int)}

def assign_bin(log2h, tracker):
    idx = np.searchsorted(tracker['edges'], log2h, side='right') - 1
    return int(np.clip(idx, 0, len(tracker['counts']) - 1))

def sample_target_log2h(tracker, rng, min_log2h, max_log2h):
    counts  = tracker['counts']
    weights = 1.0 / (counts + 1.0)
    weights /= weights.sum()
    bin_idx = rng.choice(len(counts), p=weights)
    lo, hi  = tracker['edges'][bin_idx], tracker['edges'][bin_idx + 1]
    return float(rng.uniform(lo, hi))


# ── PER-COMBO GENERATOR ───────────────────────────────────────────────────────

def generate_combo(n, k, m, target, min_log2h, max_log2h,
                   seed, log_dir, output_dir):

    rng      = np.random.default_rng(seed)
    tag      = f"n{n}_k{k}_m{m}"
    log_path = os.path.join(log_dir, f"{tag}_v5.log")
    r        = n - k

    # Resume from checkpoint if available
    existing = load_checkpoint(output_dir, n, k, m)
    if existing[0] is not None:
        samples, heights = existing
        print(f"  [{tag}] Resuming from checkpoint — {len(samples)} samples already done")
    else:
        samples, heights = [], []

    if len(samples) >= target:
        print(f"  [{tag}] Already complete — skipping")
        return samples, heights

    tracker   = make_bin_tracker(min_log2h, max_log2h)
    # Restore bin counts from existing samples
    for h in heights:
        bin_idx = assign_bin(np.log2(h), tracker)
        tracker['counts'][bin_idx] += 1

    attempts  = 0
    rejected  = 0
    t0        = time.time()
    log_every = 50  # print every 50 samples so we can see progress quickly

    print(f"  [{tag}] Starting — need {target - len(samples)} more samples")

    with open(log_path, 'a') as log:
        log.write(
            f"[{tag}] v5 — target={target}, seed={seed}, "
            f"log2h=[{min_log2h},{max_log2h}], P=({k},{r})\n\n"
        )
        log.flush()

        while len(samples) < target:
            attempts += 1

            target_log2h = sample_target_log2h(tracker, rng, min_log2h, max_log2h)
            target_h     = 2.0 ** target_log2h
            P_base       = generate_P_for_target(target_log2h, rng, k, r)
            alpha        = scale_to_target(P_base, n, k, target_h)

            if alpha is None:
                rejected += 1
                continue

            P = P_base * alpha

            try:
                actual_h = compute_height_c11(n, k, P)
            except Exception:
                rejected += 1
                continue

            if actual_h <= 0 or np.isinf(actual_h) or np.isnan(actual_h):
                rejected += 1
                continue

            actual_log2h = np.log2(actual_h)

            if actual_log2h < min_log2h or actual_log2h > max_log2h:
                rejected += 1
                continue
            if abs(actual_log2h - target_log2h) > 1.0:
                rejected += 1
                continue

            bin_idx     = assign_bin(actual_log2h, tracker)
            max_count   = max(tracker['counts'].max(), 1)
            this_count  = tracker['counts'][bin_idx]
            accept_prob = 1.0 / (1.0 + this_count / max(max_count * 0.2, 1))
            if rng.random() > accept_prob:
                rejected += 1
                continue

            tracker['counts'][bin_idx] += 1
            samples.append([n, k, m, P])
            heights.append(float(actual_h))

            if len(samples) % log_every == 0:
                elapsed     = time.time() - t0
                accept_rate = len(samples) / max(attempts, 1)
                eta         = (elapsed / len(samples)) * (target - len(samples))
                bin_summary = ' '.join(str(c) for c in tracker['counts'])
                msg = (
                    f"[{tag}] {len(samples)}/{target}  "
                    f"accept={accept_rate:.1%}  rejected={rejected}  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s\n"
                    f"  bin counts: [{bin_summary}]\n"
                )
                print(f"  {msg.strip()}")
                log.write(msg)
                log.flush()
                save_checkpoint(output_dir, n, k, m, samples, heights)

        elapsed     = time.time() - t0
        accept_rate = len(samples) / max(attempts, 1)
        log2h_arr   = np.log2(np.array(heights))
        msg = (
            f"\n[{tag}] DONE — {len(samples)} samples  "
            f"accept={accept_rate:.1%}  time={elapsed:.1f}s\n"
            f"  log2(h): mean={log2h_arr.mean():.2f}  "
            f"std={log2h_arr.std():.2f}  "
            f"range=[{log2h_arr.min():.2f},{log2h_arr.max():.2f}]\n"
            f"  bins: {tracker['counts'].tolist()}\n"
        )
        print(msg)
        log.write(msg)

    save_checkpoint(output_dir, n, k, m, samples, heights)
    return samples, heights


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot_results(output_dir, min_log2h, max_log2h):
    import matplotlib.pyplot as plt

    TARGET_COMBOS = [(9, 4, 5), (9, 5, 4), (9, 6, 3)]
    colors = ['#e74c3c', '#e67e22', '#8e44ad']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("v5 Synthetic Data — log₂(m-height) Distribution\n"
                 "Green dashed = target 7.5 region  |  Red dotted = original mean (~13.4)",
                 fontsize=11, fontweight='bold')

    for ax, (n, k, m), color in zip(axes, TARGET_COMBOS, colors):
        result = load_checkpoint(output_dir, n, k, m)
        if result[0] is None:
            ax.set_title(f"({n},{k},{m}) — no data")
            continue

        _, heights = result
        log2h = np.log2(np.array(heights))

        ax.hist(log2h, bins=50, color=color, edgecolor='white',
                linewidth=0.4, alpha=0.85)
        ax.axvline(log2h.mean(), color='black', linestyle='--',
                   linewidth=1.5, label=f'mean={log2h.mean():.1f}')
        ax.axvline(7.5, color='#27ae60', linestyle='--',
                   linewidth=1.5, label='target 7.5')
        ax.axvline(13.4, color='#c0392b', linestyle=':',
                   linewidth=1.2, label='original mean')
        ax.set_title(
            f"(n={n}, k={k}, m={m})  n={len(heights):,}\n"
            f"mean={log2h.mean():.2f}  std={log2h.std():.2f}  min={log2h.min():.2f}",
            fontsize=9, color=color)
        ax.set_xlabel("log₂(height)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'v5_synthetic_histogram.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to: {out_path}")
    plt.show()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='v5: Scale-controlled inverse design for (9,4,5), (9,5,4), (9,6,3)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--target',    type=int,   default=15000)
    parser.add_argument('--min_log2h', type=float, default=4.0)
    parser.add_argument('--max_log2h', type=float, default=23.0)
    parser.add_argument('--seed',      type=int,   default=999)
    parser.add_argument('--output',    type=str,
                        default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--plot',      action='store_true',
                        help='Plot histogram after generation')
    args = parser.parse_args()

    TARGET_COMBOS = [(9, 4, 5), (9, 5, 4), (9, 6, 3)]

    log_dir = os.path.join(args.output, 'gen_logs_v5')
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"synthetic_data_gen_v5 — sequential, no multiprocessing")
    print(f"  Target/combo: {args.target:,}  |  Total: {args.target*3:,}")
    print(f"  log2(h) range: [{args.min_log2h}, {args.max_log2h}]")
    print(f"{'='*60}\n")

    all_samples, all_heights = [], []

    for i, (n, k, m) in enumerate(TARGET_COMBOS):
        print(f"[{i+1}/3] Generating ({n},{k},{m})...")
        s, h = generate_combo(
            n, k, m,
            args.target,
            args.min_log2h,
            args.max_log2h,
            args.seed + i * 1000,
            log_dir,
            args.output
        )
        all_samples.extend(s)
        all_heights.extend(h)

    # Save merged output
    new_data_path   = os.path.join(args.output, 'new_samples_v5.pkl')
    new_height_path = os.path.join(args.output, 'new_heights_v5.pkl')

    with open(new_data_path, 'wb') as f:
        pickle.dump(all_samples, f)
    with open(new_height_path, 'wb') as f:
        pickle.dump(all_heights, f)

    h_arr = np.array(all_heights)
    print(f"\n{'='*60}")
    print(f"Done! {len(all_samples):,} total samples saved.")
    print(f"  log2(h) mean:  {np.log2(h_arr).mean():.2f}  (original ~13.4)")
    print(f"  log2(h) std:   {np.log2(h_arr).std():.2f}   (original ~2.0)")
    print(f"  log2(h) range: [{np.log2(h_arr.min()):.2f}, {np.log2(h_arr.max()):.2f}]")
    print(f"\nMerge in your notebook:")
    print(f"  with open('new_samples_v5.pkl', 'rb') as f:")
    print(f"      new_data = pickle.load(f)")
    print(f"  with open('new_heights_v5.pkl', 'rb') as f:")
    print(f"      new_heights_v5 = pickle.load(f)")
    print(f"  train_data    = train_data    + new_data")
    print(f"  train_heights = train_heights + new_heights_v5")
    print(f"{'='*60}\n")

    if args.plot:
        print("Generating histogram...")
        plot_results(args.output, args.min_log2h, args.max_log2h)


if __name__ == '__main__':
    main()
