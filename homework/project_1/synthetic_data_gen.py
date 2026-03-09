



import numpy as np
import pickle
import time
import os
import argparse
import multiprocessing
from math import comb
from itertools import combinations
from scipy.optimize import linprog


# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────

def checkpoint_path(output_dir, n, k, m):
    """Path for a single combo's checkpoint file."""
    return os.path.join(output_dir, 'checkpoints', f'ckpt_n{n}_k{k}_m{m}.pkl')

def save_checkpoint(output_dir, n, k, m, samples, heights):
    """Save a completed combo's results to its own checkpoint file."""
    ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = checkpoint_path(output_dir, n, k, m)
    with open(path, 'wb') as f:
        pickle.dump({'samples': samples, 'heights': heights}, f)

def load_checkpoint(output_dir, n, k, m):
    """Load a checkpoint if it exists. Returns (samples, heights) or (None, None)."""
    path = checkpoint_path(output_dir, n, k, m)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['samples'], data['heights']
    return None, None

def checkpoint_exists(output_dir, n, k, m):
    return os.path.exists(checkpoint_path(output_dir, n, k, m))


# ── LP CORE ───────────────────────────────────────────────────────────────────

def compute_m_height_lp(n, k, m, P):
    """
    Compute m-height via LP (Roth et al. 2025, Fig. 1).
    Returns float h >= 1.0, or np.inf if unbounded.
    """
    G = np.hstack([np.eye(k), P])
    best_h = 1.0

    for S in combinations(range(n), m):
        Sbar  = [j for j in range(n) if j not in S]
        G_S   = G[:, Sbar]
        A_ub  = np.vstack([G_S.T, -G_S.T])
        b_ub  = np.ones(2 * (n - m))

        for i in S:
            g_i = G[:, i]
            res = linprog(-g_i, A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None)] * k, method='highs')
            if res.status == 0:
                best_h = max(best_h, -res.fun)
            elif res.status == 3:
                return np.inf

    return best_h


# ── PER-COMBO WORKER ──────────────────────────────────────────────────────────

def generate_combo(args):
    """
    Worker function: generates `target` valid samples for one (n, k, m) combo.
    - Writes progress to a per-combo log file in log_dir
    - Saves a checkpoint to output_dir/checkpoints/ when done
    - Returns (samples_list, heights_list)
    """
    n, k, m, target, p_range, max_log2h, seed, log_dir, output_dir = args

    np.random.seed(seed)
    tag = f"n{n}_k{k}_m{m}"

    log_path = os.path.join(log_dir, f"{tag}.log")
    os.makedirs(log_dir, exist_ok=True)

    samples, heights = [], []
    attempts  = 0
    rejected  = 0
    t0        = time.time()
    log_every = max(1, target // 20)   # log ~20 times per combo

    with open(log_path, 'w') as log:
        log.write(f"[{tag}] Starting — target={target}, seed={seed}\n")
        log.flush()

        while len(samples) < target:
            attempts += 1
            P = np.random.uniform(-p_range, p_range, size=(k, n - k))
            h = compute_m_height_lp(n, k, m, P)

            if np.isinf(h) or h <= 0 or np.log2(h) > max_log2h:
                rejected += 1
                continue

            # Format: list [n, k, m, P] — exactly what encode_sample expects
            samples.append([n, k, m, P])
            heights.append(float(h))

            if len(samples) % log_every == 0:
                elapsed     = time.time() - t0
                accept_rate = len(samples) / attempts
                eta         = (elapsed / len(samples)) * (target - len(samples))
                msg = (f"[{tag}] {len(samples)}/{target}  "
                       f"accept={accept_rate:.1%}  "
                       f"rejected={rejected}  "
                       f"elapsed={elapsed:.0f}s  eta={eta:.0f}s\n")
                log.write(msg)
                log.flush()

        elapsed     = time.time() - t0
        accept_rate = len(samples) / attempts
        msg = (f"[{tag}] DONE — {len(samples)} samples  "
               f"accept={accept_rate:.1%}  "
               f"total_attempts={attempts}  "
               f"time={elapsed:.1f}s\n")
        log.write(msg)
        log.flush()

    # Save checkpoint immediately — this is the crash-safety guarantee
    save_checkpoint(output_dir, n, k, m, samples, heights)
    log.close() if hasattr(log, 'close') else None

    return samples, heights


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Parallel m-height synthetic data generator')
    parser.add_argument('--target',    type=int,   default=5000,
                        help='Samples per (n,k,m) combo (default: 5000)')
    parser.add_argument('--max_log2h', type=float, default=23.0,
                        help='Outlier cap in log2 space (default: 23.0)')
    parser.add_argument('--p_range',   type=float, default=100.0,
                        help='P entries drawn from [-p_range, p_range] (default: 100)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Base random seed (default: 42)')
    parser.add_argument('--output',    type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='Output directory for pkl files (default: script directory)')
    parser.add_argument('--workers',   type=int,   default=None,
                        help='Number of parallel workers (default: min(9, cpu_count))')
    args = parser.parse_args()

    PARAM_COMBOS = [
        (9, 4, 2), (9, 4, 3), (9, 4, 4), (9, 4, 5),
        (9, 5, 2), (9, 5, 3), (9, 5, 4),
        (9, 6, 2), (9, 6, 3),
    ]

    n_workers = args.workers or min(len(PARAM_COMBOS), multiprocessing.cpu_count())
    log_dir   = os.path.join(args.output, 'gen_logs')
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(log_dir,     exist_ok=True)

    total_samples = args.target * len(PARAM_COMBOS)
    print(f"\n{'='*60}")
    print(f"Parallel m-height data generation (with checkpointing)")
    print(f"  Combos:       {len(PARAM_COMBOS)}")
    print(f"  Target/combo: {args.target:,}")
    print(f"  Total target: {total_samples:,}")
    print(f"  Workers:      {n_workers}")
    print(f"  Output dir:   {args.output}")
    print(f"  Log dir:      {log_dir}")
    print(f"{'='*60}\n")

    # ── CHECK WHICH COMBOS ARE ALREADY DONE ───────────────────────────────────
    pending, done = [], []
    for n, k, m in PARAM_COMBOS:
        if checkpoint_exists(args.output, n, k, m):
            done.append((n, k, m))
        else:
            pending.append((n, k, m))

    print(f"Checkpoints found:  {len(done)}/9 combos already complete")
    for n, k, m in done:
        print(f"  ✓  ({n},{k},{m}) — skipping")
    print(f"Combos to run now:  {len(pending)}")
    for n, k, m in pending:
        print(f"  →  ({n},{k},{m})")

    if not pending:
        print('\nAll combos already done — skipping to merge step.')
    else:
        print(f'\nMonitor progress with:')
        print(f'  !tail -f {log_dir}/*.log\n')

        # Build worker args only for pending combos
        # Seed is kept consistent with original run (indexed by PARAM_COMBOS position)
        # so re-runs of the same combo use a different seed (offset by run count)
        worker_args = [
            (n, k, m, args.target, args.p_range, args.max_log2h,
             args.seed + PARAM_COMBOS.index((n, k, m)) * 1000, log_dir, args.output)
            for (n, k, m) in pending
        ]

        t0 = time.time()
        with multiprocessing.Pool(processes=min(n_workers, len(pending))) as pool:
            pool.map(generate_combo, worker_args)

        elapsed = time.time() - t0
        print(f'\nBatch complete in {elapsed/60:.1f} minutes.')

    # ── MERGE ALL CHECKPOINTS ─────────────────────────────────────────────────
    print(f'\nMerging checkpoints...')
    all_samples, all_heights = [], []
    missing = []

    for n, k, m in PARAM_COMBOS:
        s, h = load_checkpoint(args.output, n, k, m)
        if s is None:
            missing.append((n, k, m))
            print(f'  ✗  ({n},{k},{m}) — checkpoint missing, rerun to generate')
        else:
            print(f'  ✓  ({n},{k},{m}) — {len(s):,} samples loaded')
            all_samples.extend(s)
            all_heights.extend(h)

    if missing:
        print(f'\n{len(missing)} combos still missing — rerun the script to continue.')
        print(f'Partial data NOT saved to avoid overwriting a previous complete set.')
        return

    # Shuffle before saving
    rng         = np.random.default_rng(args.seed)
    idx         = rng.permutation(len(all_samples))
    all_samples = [all_samples[i] for i in idx]
    all_heights = [all_heights[i] for i in idx]

    # Save final merged files
    data_path = os.path.join(args.output, 'synthetic_training_data.pkl')
    h_path    = os.path.join(args.output, 'synthetic_training_heights.pkl')

    with open(data_path, 'wb') as f:
        pickle.dump(all_samples, f)
    with open(h_path, 'wb') as f:
        pickle.dump(all_heights, f)

    print(f"\n{'='*60}")
    print(f"All done!")
    print(f"Total samples: {len(all_samples):,}")
    print(f"Saved to:")
    print(f"  {data_path}")
    print(f"  {h_path}")

    # Quick format sanity check
    def encode_sample(sample):
        n, k, m, P = sample
        p_flat   = P.flatten()
        p_padded = np.zeros(20)
        p_padded[:len(p_flat)] = p_flat
        p_padded /= 100.0
        return np.concatenate([[n/9., k/9., m/9.], p_padded]).astype(np.float32)

    feat = encode_sample(all_samples[0])
    assert feat.shape == (23,), f"encode_sample shape mismatch: {feat.shape}"
    h_arr = np.array(all_heights)
    assert (h_arr >= 1.0).all(), "Some heights < 1.0"
    assert not np.isinf(h_arr).any(), "Inf heights found"
    print(f"\nFormat check passed:")
    print(f"  encode_sample shape:  {feat.shape}")
    print(f"  log2(h) range:        [{np.log2(h_arr.min()):.2f}, {np.log2(h_arr.max()):.2f}]")
    print(f"  All heights >= 1.0:   True")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()