"""
EJCPCFGM Model - Extended CPCF Grey Model with Impulse Basis Functions
CPCFAGO: CPCF Accumulated Generating Operation
CPCFIAGO: CPCF Inverse Accumulated Generating Operation
"""

import numpy as np
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')


# CPCF core constants computation
def compute_core_constants(alpha, c=1.0):
    K1 = 1.0 - alpha
    K0 = alpha * (c ** alpha)
    if alpha > 0 and alpha < 1:
        lam = math.exp(-alpha / (1.0 - alpha))
    else:
        lam = 0.0
    if alpha > 0:
        W = (1.0 - lam) / alpha
    else:
        W = 1.0
    return K1, K0, lam, W


# Build n x n CPCF matrix L (lower triangular)
def compute_cpcf_matrix(n, alpha, c=1.0):
    K1, K0, lam, W = compute_core_constants(alpha, c)
    L = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(k + 1):
            if k == 0 and i == 0:
                L[k, i] = W * K1
            elif i == k and k > 0:
                L[k, i] = W * (K1 + K0)
            elif i == 0 and k > 0:
                power = k
                L[k, i] = W * ((lam ** power) * K1 - (lam ** (power - 1)) * K0)
            else:
                power = k - i
                L[k, i] = W * ((lam ** power) * (K1 + K0) - (lam ** (power - 1)) * K0)
    return L


# CPCFAGO: Accumulated Generating Operation - solve L_ago x X_r = X
def cpcfago_forward(X, r, c=1.0):
    X = np.array(X, dtype=np.float64)
    n = len(X)
    L_ago = compute_cpcf_matrix(n, r, c)
    X_r = np.linalg.solve(L_ago, X)
    return X_r


# CPCFIAGO: Inverse Accumulated Generating Operation - X = L_ago x X_r
def cpcfiago_inverse(X_r, r, c=1.0):
    X_r = np.array(X_r, dtype=np.float64)
    n = len(X_r)
    L_ago = compute_cpcf_matrix(n, r, c)
    X = L_ago @ X_r
    return X


# Compute discrete CPCF derivative: D = L_diff x X_r
def compute_cpcf_derivative(X_r, sigma, c=1.0):
    X_r = np.array(X_r, dtype=np.float64)
    n = len(X_r)
    L_diff = compute_cpcf_matrix(n, sigma, c)
    D = L_diff @ X_r
    return D


# Grey background value Z(k) = 0.5 * (x_r(k) + x_r(k-1))
def compute_Z_k(X_r, k):
    if k <= 1:
        return X_r[0]
    return 0.5 * (X_r[int(k) - 1] + X_r[int(k) - 2])


# Number of impulse basis functions
N_PULSES = 1


# Map normalized tau_array to [1, n] interval
def compute_tau_positions(tau_array, n):
    tau_array = np.array(tau_array, dtype=np.float64)
    N = len(tau_array)
    taus = np.zeros(N, dtype=np.float64)
    for i in range(N):
        tau_start = 1.0 + (n - 1) * float(i) / float(N)
        tau_end = 1.0 + (n - 1) * float(i + 1) / float(N)
        taus[i] = tau_start + tau_array[i] * (tau_end - tau_start)
    return taus


# Impulse basis function: E_i(k) = e^{theta_i (k-0.5-tau_i)} * U(k-0.5-tau_i)
def impulse_basis(k, theta_i, tau_i):
    t = float(k) - 0.5
    if t < tau_i:
        return 0.0
    return math.exp(theta_i * (t - tau_i))


# OLS solver for EJCPCFGM N-impulse model
def solve_ols_multi_pulse(D, X_r, thetas, taus):
    global N_PULSES
    N = N_PULSES
    n = len(X_r)
    k_start = 2
    n_eq = n - 1

    if n_eq <= 0:
        return np.zeros(N + 2)

    Y = np.zeros(n_eq, dtype=np.float64)
    for i in range(n_eq):
        k = k_start + i
        Y[i] = 0.5 * (D[k - 1] + D[k - 2])

    n_cols = N + 2
    B = np.zeros((n_eq, n_cols), dtype=np.float64)
    for i in range(n_eq):
        k = k_start + i
        Z_k = compute_Z_k(X_r, k)
        B[i, 0] = -Z_k
        for j in range(N):
            B[i, 1 + j] = impulse_basis(k, thetas[j], taus[j])
        B[i, N + 1] = 1.0

    try:
        BT = B.T
        BT_B = BT @ B
        for j in range(n_cols):
            BT_B[j, j] += 1e-8
        P = np.linalg.solve(BT_B, BT @ Y).flatten()
        return P
    except:
        return np.zeros(N + 2)


# Time response function for EJCPCFGM N-impulse model
def time_response(t, sigma, cq, a, P, thetas, taus, x1_r):
    if t < 1e-12:
        t = 1e-12

    global N_PULSES
    N = N_PULSES

    K1 = 1.0 - sigma
    K0 = sigma * (cq ** sigma)
    A = K0 + a * K1
    B = K1 + a * sigma
    Delta = sigma ** 2 * (cq ** sigma) - K1 ** 2

    eps = 1e-15
    if abs(A) < eps:
        A = eps if A >= 0 else -eps
    if abs(B) < eps:
        B = eps if B >= 0 else -eps
    if abs(Delta) < eps:
        Delta = eps if Delta >= 0 else -eps

    lam = B / A
    C_val = Delta / (A ** 2)
    ADB = A / B if abs(B) > eps else 0.0

    d = P[N + 1] if len(P) > N + 1 else P[-1]
    bs = P[1:1 + N] if N > 0 else np.array([])

    exp_m_t = math.exp(-lam * t)

    # Direct term
    f_t = d
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t >= tau_i:
            f_t += bi * math.exp(theta_i * (t - tau_i))
    direct = (K1 / A) * f_t

    # Convolution term
    I_d = d * ADB * (1.0 - exp_m_t)

    I_pulses = 0.0
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t < tau_i:
            continue
        denom = theta_i + lam
        exp_theta_tau = math.exp(theta_i * (t - tau_i))
        exp_m_t_tau = math.exp(-lam * (t - tau_i))
        I_E = (exp_theta_tau - exp_m_t_tau) / denom if abs(denom) > eps \
              else (t - tau_i) * exp_m_t_tau
        I_pulses += bi * I_E

    Phi_t = direct + C_val * (I_d + I_pulses)

    # t=1 anchoring
    t1 = 1.0
    exp_m_1 = math.exp(-lam)

    f_t1 = d
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t1 >= tau_i:
            f_t1 += bi * math.exp(theta_i * (t1 - tau_i))
    direct_1 = (K1 / A) * f_t1

    I_d_1 = d * ADB * (1.0 - exp_m_1)

    I_pulses_1 = 0.0
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t1 < tau_i:
            continue
        denom = theta_i + lam
        exp_theta_tau_1 = math.exp(theta_i * (t1 - tau_i))
        exp_m_1_tau = math.exp(-lam * (t1 - tau_i))
        I_E_1 = (exp_theta_tau_1 - exp_m_1_tau) / denom if abs(denom) > eps \
                else (t1 - tau_i) * exp_m_1_tau
        I_pulses_1 += bi * I_E_1

    Phi_1 = direct_1 + C_val * (I_d_1 + I_pulses_1)

    correction = (x1_r - Phi_1) * math.exp(-lam * (t - 1.0))

    return Phi_t + correction


# Fitness function for EJCPCFGM N-impulse model
def fitness_function(vars_tuple, X_train):
    global N_PULSES
    N = N_PULSES

    sigma = vars_tuple[0]
    cq = vars_tuple[1]
    r = vars_tuple[2]
    thetas = list(vars_tuple[3:3 + N])
    taus_norms = list(vars_tuple[3 + N:3 + 2 * N])

    if not (0.0 < sigma < 1.0 and 0.0 < cq < 5.0 and 0.0 < r < 1.0):
        return 1e10
    for theta_i in thetas:
        if not (-5.0 < theta_i <= 0.0):
            return 1e10

    n_train = len(X_train)
    taus = compute_tau_positions(taus_norms, n_train)

    try:
        mape, _ = evaluate_model(X_train, sigma, cq, r, thetas, taus)
        if np.isnan(mape) or np.isinf(mape):
            return 1e10
        return mape
    except Exception:
        return 1e10


# DAI DE helper functions
def calculate_dimension_diversity(population):
    NP, D = population.shape
    diversity = np.zeros(D)
    for j in range(D):
        mean_val = np.mean(population[:, j])
        diversity[j] = np.sum((population[:, j] - mean_val) ** 2) / NP
    return diversity


def calculate_diversity_ranking(diversity):
    D = len(diversity)
    sorted_indices = np.argsort(diversity)
    diversity_ranking = np.zeros(D, dtype=int)
    for rank, idx in enumerate(sorted_indices):
        diversity_ranking[idx] = rank + 1
    return diversity_ranking


def calculate_fitness_ranking(fitness):
    NP = len(fitness)
    sorted_indices = np.argsort(fitness)
    fitness_ranking = np.zeros(NP, dtype=int)
    for rank, idx in enumerate(sorted_indices):
        fitness_ranking[idx] = rank + 1
    return fitness_ranking


def calculate_dimension_threshold(rank_i, NP, D):
    N_g_i = np.ceil(D * (1 - rank_i / NP)).astype(int)
    return N_g_i


def generate_base_cr_values(X_i, X_best, fitness, best_idx, F_scale=0.5):
    small_cr = np.random.uniform(0.05, 0.15)
    large_cr = np.random.uniform(0.7, 0.95)
    return small_cr, large_cr


def dai_crossover(target_vec, mutant_vec, dai_cr, D):
    trial = np.copy(target_vec)
    j_rand = np.random.randint(0, D)
    for j in range(D):
        if np.random.rand() <= dai_cr[j] or j == j_rand:
            trial[j] = mutant_vec[j]
    return trial


# DAI Differential Evolution for EJCPCFGM
def optimize_dai(train_data, pop_size=30, max_iter=500):
    global N_PULSES
    N = N_PULSES
    D = 3 + 2 * N
    NP = pop_size

    # Parameter bounds: [sigma, cq, r, theta_1..N, tau_1..N]
    lb = np.zeros(D, dtype=float)
    ub = np.zeros(D, dtype=float)
    lb[0] = 0.00; ub[0] = 1.00
    lb[1] = 0.00; ub[1] = 2.00
    lb[2] = 0.00; ub[2] = 1.00
    for j in range(N):
        lb[3 + j] = -4.0; ub[3 + j] = 0.0
    for j in range(N):
        lb[3 + N + j] = 0.0; ub[3 + N + j] = 1.0

    Q = 0.7
    F = 0.5

    # Initialize population
    population = np.zeros((NP, D), dtype=float)
    for i in range(NP):
        population[i] = lb + np.random.rand(D) * (ub - lb)

    # Evaluate initial population
    fitness = np.zeros(NP)
    for i in range(NP):
        fitness[i] = fitness_function(tuple(population[i]), train_data)

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx].copy()

    # Main loop
    for g in range(max_iter):
        diversity = calculate_dimension_diversity(population)
        diversity_ranking = calculate_diversity_ranking(diversity)
        fitness_ranking = calculate_fitness_ranking(fitness)
        mean_cr_estimate = 0.5
        strategy = 1 if mean_cr_estimate < Q else 2

        dai_cr_matrix = np.zeros((NP, D))
        for i in range(NP):
            small_cr, large_cr = generate_base_cr_values(
                population[i], best_solution, fitness, best_idx, F
            )
            N_g_i = calculate_dimension_threshold(fitness_ranking[i], NP, D)
            for j in range(D):
                if strategy == 1:
                    dai_cr_matrix[i, j] = small_cr if diversity_ranking[j] < N_g_i else large_cr
                else:
                    dai_cr_matrix[i, j] = large_cr if diversity_ranking[j] < N_g_i else small_cr

        # Mutation
        mutant_population = np.zeros((NP, D), dtype=float)
        for i in range(NP):
            candidates = list(range(NP))
            candidates.remove(i)
            r_indices = np.random.choice(candidates, 3, replace=False)
            mutant_population[i] = population[r_indices[0]] + F * (population[r_indices[1]] - population[r_indices[2]])

        # Boundary handling
        for i in range(NP):
            for j in range(D):
                if mutant_population[i, j] < lb[j]:
                    mutant_population[i, j] = lb[j] + np.random.rand() * (population[i, j] - lb[j])
                elif mutant_population[i, j] > ub[j]:
                    mutant_population[i, j] = ub[j] - np.random.rand() * (ub[j] - population[i, j])

        # Crossover and selection
        new_population = np.zeros((NP, D), dtype=float)
        new_fitness = np.zeros(NP)

        for i in range(NP):
            trial = dai_crossover(population[i], mutant_population[i], dai_cr_matrix[i], D)
            trial = np.clip(trial, lb, ub)
            trial_fitness = fitness_function(tuple(trial), train_data)

            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]

        population = new_population
        fitness = new_fitness

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        if g % 50 == 0:
            print(f"DAI iter {g:3d}: MAPE = {best_fitness:.4f}%, strategy = {'I' if strategy == 1 else 'II'}")

    print(f"DAI iter {max_iter:3d}: MAPE = {best_fitness:.4f}%")

    return best_solution, best_fitness


# Evaluate EJCPCFGM on training data
def evaluate_model(X_train, sigma, cq, r, thetas, taus):
    global N_PULSES
    N = N_PULSES

    X_origin = np.array(X_train, dtype=np.float64)
    n = len(X_origin)

    x1 = X_origin[0]
    X_norm = X_origin / x1

    X_r = cpcfago_forward(X_norm, r, cq)
    D = compute_cpcf_derivative(X_r, sigma, cq)
    P = solve_ols_multi_pulse(D, X_r, thetas, taus)
    a = P[0]

    x1_r = X_r[0]
    predictions_r = np.zeros(n, dtype=np.float64)
    for k in range(1, n + 1):
        t = float(k)
        predictions_r[k - 1] = time_response(
            t, sigma, cq, a, P, thetas, taus, x1_r
        )

    predictions_norm = cpcfiago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)

    mape = np.mean(np.abs((X_origin - predictions) / X_origin)) * 100.0

    return mape, P


# Predict using EJCPCFGM
def predict_model(X_train, X_test, sigma, cq, r, thetas, taus):
    global N_PULSES
    N = N_PULSES

    X_origin_train = np.array(X_train, dtype=np.float64)
    n_train = len(X_origin_train)
    n_total = n_train + len(X_test)

    x1 = X_origin_train[0]
    X_norm_train = X_origin_train / x1

    X_r_train = cpcfago_forward(X_norm_train, r, cq)
    D = compute_cpcf_derivative(X_r_train, sigma, cq)
    P = solve_ols_multi_pulse(D, X_r_train, thetas, taus)
    a = P[0]

    x1_r = X_r_train[0]
    predictions_r = np.zeros(n_total, dtype=np.float64)
    for k in range(1, n_total + 1):
        t = float(k)
        predictions_r[k - 1] = time_response(
            t, sigma, cq, a, P, thetas, taus, x1_r
        )

    predictions_norm = cpcfiago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)

    return predictions, P


# Main function for EJCPCFGM N-impulse model
def run_ejcpcfgm(train_data, test_data, pop_size=20, max_iter=100):
    global N_PULSES
    N = N_PULSES

    print("=" * 60)
    print(f"EJCPCFGM {N}-Impulse Model")
    print("=" * 60)
    print(f"Train length: {len(train_data)}")
    print(f"Test length: {len(test_data)}")
    print(f"Impulses N: {N}")
    print(f"Optimizer: DAI (Dimension-Adaptive)")
    print(f"Params: [sigma, cq, r, theta_1, tau_1, ..., theta_N, tau_N] ({3+2*N}D)")
    print("CPCFAGO: Accumulation operator")
    print("CPCFIAGO: Inverse accumulation operator")
    print("=" * 80)

    print("\nStarting DAI optimization...")
    best_params, best_mape = optimize_dai(train_data, pop_size, max_iter)

    sigma = best_params[0]
    cq = best_params[1]
    r = best_params[2]
    thetas = list(best_params[3:3 + N])
    taus_norms = list(best_params[3 + N:3 + 2 * N])
    n_train = len(train_data)
    taus = list(compute_tau_positions(taus_norms, n_train))

    print("\n" + "=" * 60)
    print("Optimal Parameters:")
    print(f"  sigma = {sigma:.6f}")
    print(f"  cq    = {cq:.6f}")
    print(f"  r     = {r:.6f}")
    for i in range(N):
        print(f"  theta_{i+1} = {thetas[i]:.6f}")
        print(f"  tau_{i+1}   = {taus[i]:.4f}")
    print(f"  Train MAPE = {best_mape:.4f}%")
    print("=" * 60)

    print("\nPredicting...")
    predictions, P = predict_model(
        train_data, test_data, sigma, cq, r, thetas, taus
    )

    a = P[0]
    bs = P[1:1 + N] if N > 0 else np.array([])
    c = P[N + 1]

    train_predictions = predictions[:len(train_data)]
    test_predictions = predictions[len(train_data):]

    train_mape = np.mean(np.abs((np.array(train_data) - train_predictions) / np.array(train_data))) * 100.0
    test_mape = np.mean(np.abs((np.array(test_data) - test_predictions) / np.array(test_data))) * 100.0

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Train MAPE: {train_mape:.4f}%")
    print(f"  Test MAPE: {test_mape:.4f}%")
    print("=" * 60)

    print("\nOLS Parameters:")
    print(f"  a = {a:.6f}")
    for i in range(N):
        print(f"  b_{i+1} = {bs[i]:.6f}")
    print(f"  c = {c:.6f}")

    return {
        'sigma': sigma,
        'cq': cq,
        'r': r,
        'thetas': thetas,
        'taus': taus,
        'a': a,
        'bs': bs,
        'c': c,
        'mape_s': train_mape,
        'mape_p': test_mape,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions
    }


# Provincial data (2010-2024)
def get_provincial_data():
    electricity_data = {
        'Hebei': [2692.00, 2984.90, 3077.70, 3251.20, 3314.11, 3175.66, 3264.52, 3441.74, 3665.66, 3856.00, 3934.00, 4294.00, 4344.00, 4757.00, 4986.67],
        'Inner Mongolia': [1537.00, 1864.07, 2016.80, 2181.90, 2416.74, 2542.87, 2605.03, 2891.87, 3353.44, 3653.00, 3900.00, 3957.00, 4291.00, 4823.00, 5193.41],
        'Jiangsu': [3864.00, 4281.62, 4580.90, 4956.60, 5012.54, 5114.70, 5458.95, 5807.89, 6128.27, 6264.00, 6374.00, 7101.00, 7400.00, 7833.00, 8486.93],
        'Anhui': [1078.00, 1221.19, 1361.10, 1528.10, 1585.18, 1639.79, 1794.98, 1921.48, 2135.07, 2301.00, 2428.00, 2715.00, 2993.00, 3214.00, 3597.86],
        'Yunnan': [1004.00, 1204.07, 1315.90, 1459.80, 1529.38, 1438.61, 1410.52, 1538.10, 1679.08, 1812.00, 2025.00, 2139.00, 2390.00, 2512.46, 2791.74],
    }

    generation_data = {
        'Hebei': [1993.12, 2326.98, 2370.90, 2507.28, 2499.90, 2498.00, 2630.59, 2817.10, 3133.18, 3297.66, 3425.07, 3513.42, 3792.87, 4125.94, 4334.58],
        'Inner Mongolia': [2489.28, 2972.83, 3116.90, 3567.14, 3857.81, 3929.00, 3949.81, 4435.94, 5002.96, 5495.08, 5810.97, 6119.93, 6619.21, 7629.94, 8344.00],
        'Jiangsu': [3359.18, 3762.50, 3928.40, 4320.68, 4347.57, 4361.00, 4709.37, 4914.74, 5085.08, 5166.43, 5217.54, 5968.89, 6077.31, 6390.53, 6807.37],
        'Anhui': [1443.85, 1635.35, 1767.50, 1970.04, 2033.91, 2062.00, 2252.69, 2456.28, 2734.49, 2886.67, 2808.98, 3083.39, 3298.77, 3549.45, 3863.46],
        'Yunnan': [1365.03, 1555.08, 1533.90, 2180.51, 2550.01, 2553.00, 2692.54, 2955.06, 3240.99, 3465.63, 3674.44, 3770.23, 4016.65, 4151.01, 4646.34],
    }

    return electricity_data, generation_data


# Get datasets in unified format
def get_datasets():
    elec_data, gen_data = get_provincial_data()
    datasets = []
    for name in elec_data.keys():
        datasets.append((name, 'Electricity', '10^8 kWh', elec_data[name]))
        datasets.append((name, 'Generation', '10^8 kWh', gen_data[name]))
    return datasets
