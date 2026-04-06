import pickle
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import time
import multiprocessing as mp

TRAIN_DATA_PATH = 'CSCE-636-Project-2-Train-n_k_m_P'
TRAIN_HEIGHT_PATH = 'CSCE-636-Project-2-Train-mHeights'
OUTPUT_PATH = 'lp_bounds_hard_combos.pkl'

TARGET_COMBOS = {(9,4,5), (9,5,4)}

def compute_lp_bound(n, k, m, P):
    G = np.hstack([np.eye(k), P])
    col_indices = list(range(n))
    best = 0.0

    for S in combinations(col_indices, m):
        S = list(S)
        S_bar = [j for j in col_indices if j not in S]
        G_Sbar = G[:, S_bar]
        A_ub = np.vstack([G_Sbar.T, -G_Sbar.T])
        b_ub = np.ones(2 * len(S_bar))

        for i in S:
            g_i = G[:, i]
            c = -g_i
            result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                           bounds=[(None, None)] * k,
                           method='highs')
            if result.success:
                val = -result.fun
                if val > best:
                    best = val

    return best

def worker(args):
    idx, sample = args
    n, k, m, P = sample
    bound = compute_lp_bound(int(n), int(k), int(m), P)
    return idx, bound

def main():
    with open(TRAIN_DATA_PATH, 'rb') as f:
        train_data = pickle.load(f)

    print(f'Total samples: {len(train_data):,}')

    target_idx = [(i, s) for i, s in enumerate(train_data)
                  if (int(s[0]), int(s[1]), int(s[2])) in TARGET_COMBOS]

    print(f'Computing LP bounds for {len(target_idx):,} samples using 4 cores...')
    start = time.time()

    lp_bounds = {}
    completed = 0

    with mp.Pool(processes=4) as pool:
        for idx, bound in pool.imap_unordered(worker, target_idx, chunksize=10):
            lp_bounds[idx] = bound
            completed += 1

            if completed % 100 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                remaining = (len(target_idx) - completed) / rate
                print(f'  {completed}/{len(target_idx)} | '
                      f'elapsed: {elapsed/60:.1f}m | '
                      f'remaining: {remaining/60:.1f}m | '
                      f'last bound: {bound:.4f}')

                # save checkpoint every 1000 samples
                if completed % 1000 == 0:
                    with open(OUTPUT_PATH, 'wb') as f:
                        pickle.dump(lp_bounds, f)
                    print(f'  checkpoint saved ({completed} samples)')

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(lp_bounds, f)

    print(f'\nDone! Saved {len(lp_bounds):,} LP bounds to {OUTPUT_PATH}')
    print(f'Total time: {(time.time()-start)/60:.1f} minutes')

if __name__ == '__main__':
    main()
