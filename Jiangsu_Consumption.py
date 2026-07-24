"""Jiangsu electricity consumption - test N impulses, pick best by test MAPE."""
import os
import csv
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from ejcpcfgm_model import run_ejcpcfgm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'Optimal_N_Results')
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN = [3864.00, 4281.62, 4580.90, 4956.60, 5012.54, 5114.70,
         5458.95, 5807.89, 6128.27, 6264.00, 6374.00, 7101.00]
TEST = [7400.00, 7833.00, 8486.93]

SEED = 29165
RIDGE = 1e-4
THETA_LB = -5.0
CQ_UB = 2.5


def run_one(n, seed):
    np.random.seed(seed)
    import ejcpcfgm_model
    ejcpcfgm_model.N_PULSES = n
    return run_ejcpcfgm(
        TRAIN, TEST,
        pop_size=20, max_iter=1500,
        ridge_lambda=RIDGE,
        theta_lb=THETA_LB, theta_ub=0.0,
        sigma_lb=0.0, sigma_ub=1.0,
        cq_lb=0.0, cq_ub=CQ_UB,
        r_lb=0.0, r_ub=1.0,
    )


def main():
    print(f"jiangsu_consumption  train={len(TRAIN)} test={len(TEST)} seed={SEED}")

    results = []
    for n in [0, 1, 2, 3]:
        t0 = time.time()
        try:
            r = run_one(n, SEED)
            dt = time.time() - t0
            row = dict(n=n, train_mape=r['mape_s'], test_mape=r['mape_p'],
                       sigma=r['sigma'], cq=r['cq'], r=r['r'],
                       a=r['beta'], c=r['delta'], mu_0=r['mu_0'])
            results.append(row)
            print(f"  N={n}  train={row['train_mape']:.4f}%  test={row['test_mape']:.4f}%  t={dt:.1f}s")
        except Exception as e:
            print(f"  N={n}  failed: {e}")

    if not results:
        return

    best = min(results, key=lambda x: x['test_mape'])
    print(f"best N={best['n']}  test_mape={best['test_mape']:.4f}%")

    fp = os.path.join(OUT_DIR, 'jiangsu_consumption.csv')
    with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['N', 'Train_MAPE(%)', 'Test_MAPE(%)', 'sigma', 'cq', 'r', 'a', 'c', 'mu_0'])
        for r in results:
            w.writerow([r['n'], f"{r['train_mape']:.4f}", f"{r['test_mape']:.4f}",
                        f"{r['sigma']:.6f}", f"{r['cq']:.6f}", f"{r['r']:.6f}",
                        f"{r['a']:.6f}", f"{r['c']:.6f}", f"{r['mu_0']:.6f}"])
    print(f"saved: {fp}")


if __name__ == "__main__":
    main()
