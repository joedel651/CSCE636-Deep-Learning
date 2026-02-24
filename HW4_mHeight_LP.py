
import numpy as np
from scipy.optimize import linprog
import pickle
from itertools import combinations
import time


def compute_m_height(n, k, m, P):
    """
    Compute the m-height of a linear code with systematic generator matrix G = [I_k | P].

    Parameters
    ----------
    n : int
        Code length.
    k : int
        Code dimension.
    m : int
        The m in "m-height" (2 <= m <= n-k).
    P : np.ndarray
        The k x (n-k) parity submatrix, so G = [I_k | P].

    Returns
    -------
    float
        The m-height h_m(C).
    """
    # Build the systematic generator matrix G = [I_k | P], shape: k x n
    #glue identity matrix k to P
    G = np.hstack([np.eye(k), P])

    #create a list of integers to n-1
    all_positions = list(range(n))
    best_val = 0.0

    #picks n-m numbers from the all_positions list and gives you every possible combination of them.
    for S_subset in combinations(all_positions, n - m):
        #the remaining larger numbers not selected 
        large_positions = [i for i in all_positions if i not in S_subset]

       #define an empty set of rows
        A_rows = []
        b_rows = []
        #acts as a filter to give us small values between -1 and 1
        for j in S_subset:
            g_j = G[:, j]
            A_rows.append(g_j)      
            b_rows.append(1.0)
            A_rows.append(-g_j)      
            b_rows.append(1.0)

        #convert to a numpy array 
        A_ub = np.array(A_rows)   
        b_ub = np.array(b_rows)

       
        for i in large_positions:
            g_i = G[:, i]

            #call the LP solver to find the maximum value of the large position coordinate subject to the small position constraints
            res = linprog(
                -g_i,                          
                A_ub=A_ub, b_ub=b_ub,
                bounds=[(None, None)] * k,    #unbounded
                method='highs' #look for the large value
            )
            if res.status == 0:
                val = -res.fun    #optimal value             
                if val > best_val:
                    best_val = val

          
            res = linprog(
                g_i,
                A_ub=A_ub, b_ub=b_ub,
                bounds=[(None, None)] * k,
                method='highs'  #look for the larg value 
            )
            if res.status == 0:
                val = -res.fun
                if val > best_val:
                    best_val = val

    return best_val


def main():
    input_file = 'HW-4-n_k_m_P'
    output_file = 'HW-4-mHeights'

    print(f"Loading data from '{input_file}'...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples.")

    m_heights = []
    start_time = time.time()

    for idx, item in enumerate(data):
        n, k, m, P = item
        h = compute_m_height(n, k, m, P)
        m_heights.append(float(h))

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            remaining = elapsed / (idx + 1) * (len(data) - idx - 1)
            print(f"  [{idx+1}/{len(data)}] elapsed: {elapsed:.1f}s, "
                  f"remaining: {remaining:.1f}s, last h_m = {h:.4f}")

    total_time = time.time() - start_time
    print(f"\nCompleted {len(data)} samples in {total_time:.1f}s.")
    print(f"First 5 m-heights: {m_heights[:5]}")

    print(f"\nSaving m-heights to '{output_file}'...")
    with open(output_file, 'wb') as f:
        pickle.dump(m_heights, f)
    print("Done.")


if __name__ == '__main__':
    main()
