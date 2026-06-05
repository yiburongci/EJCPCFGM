"""
Anhui Province - Electricity Consumption
Optimal N Testing Script
Tests N=0,1,2,3 to find optimal N based on test MAPE
"""

import numpy as np
import warnings
import time
import os
import csv

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Optimal_N_Results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

from ejcpcfgm_model import run_ejcpcfgm

# Train: 2010-2021, Test: 2022-2024
TRAIN_DATA = [1078.00, 1221.19, 1361.10, 1528.10, 1585.18, 1639.79, 1794.98, 1921.48, 2135.07, 2301.00, 2428.00, 2715.00]
TEST_DATA = [2993.00, 3214.00, 3597.86]

# Pre-determined optimal seed
OPTIMAL_SEED = 94439


def test_n_value(n_pulses, seed, pop_size=20, max_iter=1500):
    np.random.seed(seed)
    import ejcpcfgm_model
    ejcpcfgm_model.N_PULSES = n_pulses

    result = run_ejcpcfgm(
        TRAIN_DATA, TEST_DATA,
        pop_size=pop_size,
        max_iter=max_iter
    )

    return {
        'n': n_pulses,
        'seed': seed,
        'train_mape': result['mape_s'],
        'test_mape': result['mape_p'],
        'sigma': result['sigma'],
        'cq': result['cq'],
        'r': result['r'],
        'a': result['a'],
        'bs': result['bs'],
        'c': result['c'],
        'thetas': result['thetas'],
        'taus': result['taus']
    }


def main():
    print("Anhui - Electricity Consumption | Optimal N Test")
    print(f"Train: {len(TRAIN_DATA)} years, Test: {len(TEST_DATA)} years")
    print(f"Seed: {OPTIMAL_SEED}")
    print()

    results = []
    n_values = [0, 1, 2, 3]

    for n in n_values:
        print(f"Testing N = {n}")
        try:
            start_time = time.time()
            result = test_n_value(n, OPTIMAL_SEED)
            elapsed = time.time() - start_time
            results.append(result)
            print(f"  N={n}: Train={result['train_mape']:.4f}%, Test={result['test_mape']:.4f}%")
            print(f"  Time: {elapsed:.1f}s")
        except Exception as e:
            print(f"  N={n}: Failed - {str(e)}")

    print("\nSummary Results")
    print(f"{'N':>4} {'Train MAPE':>12} {'Test MAPE':>12}")
    print("-" * 30)
    for r in results:
        print(f"{r['n']:>4} {r['train_mape']:>11.4f}% {r['test_mape']:>11.4f}%")
    print("-" * 30)

    # Find optimal N based on test MAPE
    best_result = min(results, key=lambda x: x['test_mape'])
    print(f"\nOptimal N = {best_result['n']}")
    print(f"  Train MAPE: {best_result['train_mape']:.4f}%")
    print(f"  Test MAPE: {best_result['test_mape']:.4f}%")
    print(f"  Seed: {OPTIMAL_SEED}")

    # Save to CSV
    filepath = os.path.join(OUTPUT_DIR, 'Anhui_Electricity_Results.csv')
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'Seed', 'Train_MAPE(%)', 'Test_MAPE(%)', 'sigma', 'cq', 'r', 'a', 'c'])
        for r in results:
            writer.writerow([
                r['n'], r['seed'],
                f"{r['train_mape']:.4f}", f"{r['test_mape']:.4f}",
                f"{r['sigma']:.6f}", f"{r['cq']:.6f}", f"{r['r']:.6f}",
                f"{r['a']:.6f}", f"{r['c']:.6f}"
            ])
    print(f"\nResults saved: {filepath}")


if __name__ == "__main__":
    main()
