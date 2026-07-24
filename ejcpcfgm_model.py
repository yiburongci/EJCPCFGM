"""
EJCPCFGM Model - Extended CPCF Grey Model with Impulse Basis Functions
Pure Discrete Formulation with Extended OLS (includes unknown initial value)

CPCFAGO:  CPCF Accumulated Generating Operation (forward)
CPCFIAGO: CPCF Inverse Accumulated Generating Operation

Normalization function (RATIONAL form, applied throughout):
    M(alpha) = 1 / (1 - alpha + alpha * c)

Both the accumulation (CPCFAGO / CPCFIAGO) and the fractional derivative
use this same rational normalization, so the time-scale factor c influences
all stages of the model.
"""

import numpy as np
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
import time
from scipy.linalg import solve_triangular

warnings.filterwarnings('ignore')


# ============================================================
# CPCFIAGO - Inverse accumulation weights L_{k,i}  (paper Eq.16)
#   x^{(sigma)}(k) = sum_{i=1}^k L_{k,i} x^{(0)}(i)
# ============================================================
def compute_cpcfiago_matrix(n, sigma, c=1.0):
    """
    Inverse accumulation matrix L_{k,i} (paper Eq.16), evaluated at sigma.

    Rational normalization function: M(r) = 1 / (1 - r + r*c).
    λ_r = exp(-r/(1-r))
    W(r) = (1 / [r * (1 - r + r*c)]) * (1 - λ_r)

    The kernel quantities K1 = 1-r and K0 = r*c^r come from the
    exponential kernel of the CPCF definition and are unchanged by the
    choice of normalization; only the overall scaling W changes.
    """
    if sigma <= 0.0 or sigma >= 1.0:
        return np.eye(n, dtype=np.float64)

    M_r = 1.0 / (1.0 - sigma + sigma * c)   # M(r) = 1 / (1 - r + r*c)  (RATIONAL form)
    lam = math.exp(-sigma / (1.0 - sigma))  # λ_r = exp(-r/(1-r))
    W = (M_r / sigma) * (1.0 - lam)

    K1 = 1.0 - sigma
    K0 = sigma * (c ** sigma)
    L = np.zeros((n, n), dtype=np.float64)
    for k_idx in range(n):
        k = k_idx + 1  # 1-based index
        for i_idx in range(k_idx + 1):
            i = i_idx + 1  # 1-based index
            if k == 1 and i == 1:
                L[k_idx, i_idx] = W * K1
            elif i == k and k > 1:
                L[k_idx, i_idx] = W * (K1 + K0)
            elif i == 1 and k > 1:
                L[k_idx, i_idx] = W * (
                    (lam ** (k - 1)) * K1 -
                    (lam ** (k - 2)) * K0
                )
            elif k > 2 and 1 < i < k:
                L[k_idx, i_idx] = W * (
                    (lam ** (k - i)) * (K1 + K0) -
                    (lam ** (k - i - 1)) * K0
                )
    return L


def cpcfiago_inverse(X, r, c=1.0):
    """
    Inverse accumulation: x^{(r)}(k) = sum_{i=1}^k L_{k,i} x^{(0)}(i).
    Uses the RATIONAL normalization M(r) = 1 / (1 - r + r*c).
    """
    X = np.array(X, dtype=np.float64)
    n = len(X)
    L_iago = compute_cpcfiago_matrix(n, r, c)
    return L_iago @ X


# ============================================================
# CPCFAGO - Forward accumulation
#   x^{(sigma)} = A_{ago} @ x^{(0)},  with  A_{ago} = L_{iago}^{-1}
# ============================================================
def cpcfago_forward(X, sigma, c=1.0):
    """
    Forward accumulation via triangular solve against L_{iago}.
    """
    X = np.array(X, dtype=np.float64)
    n = len(X)
    L_iago = compute_cpcfiago_matrix(n, sigma, c)
    return solve_triangular(L_iago, X, lower=True)


def compute_cpcf_derivative(X_r, sigma, c=1.0):
    """CPCFIAGO applied to an r-times accumulated sequence."""
    return cpcfiago_inverse(X_r, sigma, c)


def compute_Z_k(X_r, k):
    if k <= 1:
        return X_r[0]
    return 0.5 * (X_r[int(k) - 1] + X_r[int(k) - 2])


# Number of impulse basis functions
N_PULSES = 1


def compute_tau_positions(tau_array, n_train, n_pulses=None):
    """
    Map normalized tau_array in [0, 1] to segmented ranges.

    For N pulses, the training set [1, n_train] is evenly divided into N segments.
    tau_j (j=1..N) falls within segment j: [(j-1)*n_train/N + 1, j*n_train/N].

    Example: N=2, n_train=12
      tau_1 ∈ [1, 6],  tau_2 ∈ [7, 12]
    Example: N=3, n_train=12
      tau_1 ∈ [1, 4],  tau_2 ∈ [5, 8],  tau_3 ∈ [9, 12]
    Returns empty array when N=0.
    """
    tau_array = np.asarray(tau_array, dtype=np.float64).ravel()
    N = len(tau_array)
    if N == 0:
        return np.array([], dtype=np.float64)
    segment_len = n_train / N  # length of each segment
    taus = np.zeros(N, dtype=np.float64)
    for j in range(N):
        seg_start = j * segment_len + 1.0  # 1-based start of segment j
        taus[j] = seg_start + tau_array[j] * segment_len
    return taus


# ============================================================
# STEP 3: Unknown initial value stripping
#   I_k = I_tilde_k + x^{(r)}(0) * H(k)
# ============================================================
def _cpcf_constants(sigma, c):
    """Pre-compute the CPCF constants used by both I_tilde_k and H_k.

    Normalization function (rational): M(σ) = 1 / (1 - σ + σ*c).
    λ      = σ/(1-σ)
    K0_s   = σ · c^σ
    K1_s   = 1 - σ
    Λ      = 1 - σ - λ · K0_s  = 1 - σ - λ · σ · c^σ
    """
    lam = sigma / (1.0 - sigma) if sigma < 1.0 else 0.0
    M_s = 1.0 / (1.0 - sigma + sigma * c)  # M(σ) = 1 / (1 - σ + σ*c)  (RATIONAL form)
    K0_s = sigma * (c ** sigma)              # σ · c^σ
    K1_s = 1.0 - sigma                       # 1 - σ
    Lambda_val = K1_s - lam * K0_s            # 1 - σ - λ · σ · c^σ
    return lam, M_s, Lambda_val, K0_s, K1_s


def compute_I_tilde_k(X_r, k, sigma, c=1.0):
    """
    Pure known-data integral (paper definition):
      I_tilde_k = (M/sigma) * {
          (e^lam-1) * sum_{j=2}^{k-1} (Lambda/2)[x^{(r)}(j) e^{-lam(k-j)}
                                              + x^{(r)}(j-1) e^{-lam(k-j+1)}]
        + (e^lam-1) * (Lambda/2) * x^{(r)}(1) * e^{-lam(k-1)}
        + (1/2) * [ lam*sigma*c^sigma * x^{(r)}(k)
                  + (1-sigma - Lambda*e^{-lam}) * x^{(r)}(k-1) ]
      }
    The unknown x^{(r)}(0) has been algebraically stripped out into H(k).
    """
    lam, M_s, Lambda_val, K0_s, K1_s = _cpcf_constants(sigma, c)

    # Sum over j = 2, ..., k-1.  When k=2 the sum is empty.
    sum_val = 0.0
    for j in range(2, k):
        dj = X_r[j - 1]                  # x^{(r)}(j)
        dj_prev = X_r[j - 2]             # x^{(r)}(j-1)
        # (e^lam - 1) * e^{-lam*m} = e^{-lam*(m-1)} - e^{-lam*m}
        sum_dj = (
            (math.exp(-lam * (k - j - 1)) if lam > 0 else 1.0) -
            math.exp(-lam * (k - j))
        )
        sum_dj_prev = (
            math.exp(-lam * (k - j)) -
            math.exp(-lam * (k - j + 1))
        )
        sum_val += 0.5 * Lambda_val * (dj * sum_dj + dj_prev * sum_dj_prev)

    # Constant-in-j term: x^{(r)}(1) e^{-lam*(k-1)}
    if k >= 2:
        sum_val += 0.5 * Lambda_val * X_r[0] * (
            (math.exp(-lam * (k - 2)) if lam > 0 else 1.0) -
            math.exp(-lam * (k - 1))
        )

    dk = X_r[k - 1]                       # x^{(r)}(k)
    dk_prev = X_r[k - 2] if k >= 2 else 0.0  # x^{(r)}(k-1); only used for k>=2
    current_term = (
        lam * sigma * (c ** sigma) * dk +
        (1.0 - sigma - Lambda_val * math.exp(-lam)) * dk_prev
    )

    I_tilde = (M_s / sigma) * (sum_val + 0.5 * current_term)
    return I_tilde


def compute_H_k(k, sigma, c=1.0):
    """
    Time-varying coefficient of the unknown initial value x^{(r)}(0):
      H(k) = (M/sigma) * (e^lam - 1) * e^{-lam*k} * (Lambda/2 - sigma * c^sigma)

    The complete initial-value contribution comes from TWO independent paths:
      (a) the trapezoidal-rule start point in the summation (coefficient Lambda/2),
      (b) the residual boundary term -x(0) * e^{-lam*t} generated by integration
          by parts on the x'(tau) component of the CPCF kernel, which integrates
          over [k-1, k] to give coefficient -sigma * c^sigma.

    Both paths share the common factor (M/sigma) * (e^lam - 1) * e^{-lam*k}.
    Their SUM is the total coefficient of x^{(r)}(0) in I_k.
    """
    lam, M_s, Lambda_val, K0_s, K1_s = _cpcf_constants(sigma, c)
    e_lam_m1 = math.exp(lam) - 1.0
    e_lam_k = math.exp(-lam * k) if lam > 0 else 1.0
    return (M_s / sigma) * e_lam_m1 * e_lam_k * (0.5 * Lambda_val - K0_s)


# ============================================================
# STEP 3 (right-hand side): Exact integral of impulse over [k-1, k]
# ============================================================
def compute_Psi_i(k, theta_i, tau_i):
    """Exact integral of e^{theta*(t-tau)}*U(t-tau) over [k-1, k]."""
    t_end = float(k)
    t_start = float(k - 1)
    eps = 1e-15

    if t_end <= tau_i:
        return 0.0
    if abs(theta_i) < eps:
        if t_start <= tau_i < t_end:
            return t_end - tau_i
        else:
            return 1.0
    else:
        if t_start <= tau_i < t_end:
            return (math.exp(theta_i * (t_end - tau_i)) - 1.0) / theta_i
        else:
            return (math.exp(theta_i * (t_end - tau_i)) *
                    (1.0 - math.exp(-theta_i))) / theta_i


# ============================================================
# STEP 4: Extended OLS solver
#   P = [beta, gamma_1..gamma_N, delta, mu_0]^T
#   Y = I_tilde_k
#   B = [-Z(k), Psi_1(k)..Psi_N(k), 1, -H(k)]
# ============================================================
def solve_ols_multi_pulse(X_r, thetas, taus, sigma, cq,
                          ridge_lambda=1e-7,
                          ridge_scales=None,
                          use_ridge=True):
    """
    Extended OLS / Ridge solver with the unknown initial value stripped out.

    Parameters
    ----------
    X_r        : r-order accumulated sequence.
    thetas     : list of N impulse decay rates.
    taus       : list of N impulse onset positions.
    sigma      : fractional order of the CPCF derivative.
    cq         : time-scale factor c.
    ridge_lambda : non-negative L2 penalty.
    ridge_scales : optional length-(N+3) per-column weights.
    use_ridge  : master switch.

    Returns
    -------
    P : ndarray of length N+3, ordered as
        [beta, gamma_1, ..., gamma_N, delta, mu_0].
    """
    global N_PULSES
    N = N_PULSES
    n = len(X_r)
    k_start = 2
    n_eq = n - 1

    n_cols = N + 3   # beta, gammas, delta, mu_0
    if n_eq <= 0:
        return np.zeros(n_cols)

    Y = np.zeros(n_eq, dtype=np.float64)
    B = np.zeros((n_eq, n_cols), dtype=np.float64)
    for i in range(n_eq):
        k = k_start + i
        Y[i] = compute_I_tilde_k(X_r, k, sigma, cq)

        Z_k = compute_Z_k(X_r, k)
        B[i, 0] = -Z_k
        for j in range(N):
            B[i, 1 + j] = compute_Psi_i(k, thetas[j], taus[j])
        B[i, N + 1] = 1.0
        B[i, N + 2] = -compute_H_k(k, sigma, cq)

    try:
        BT = B.T
        BT_B = BT @ B
        if use_ridge and ridge_lambda > 0.0:
            if ridge_scales is None:
                ridge_scales = np.ones(n_cols, dtype=np.float64)
            else:
                ridge_scales = np.asarray(ridge_scales, dtype=np.float64)
                if ridge_scales.shape != (n_cols,):
                    ridge_scales = np.ones(n_cols, dtype=np.float64)
            diag_add = (ridge_scales ** 2) * ridge_lambda
            for j in range(n_cols):
                BT_B[j, j] += diag_add[j] if hasattr(diag_add, '__len__') else diag_add
        P = np.linalg.solve(BT_B, BT @ Y).flatten()
        return P
    except Exception:
        return np.zeros(n_cols)


# ============================================================
# STEP 5: Pure discrete recursive response
#   Gamma    = (M * lam * c^sigma) / 2     (coefficient of x^{(r)}(k) inside I_k)
#   Historical_k = I_k(known_history) - Gamma * x^{(r)}(k)    [using mu_0 for x^{(r)}(0)]
#   x^{(r)}(k) = (sum_i gamma_i * Psi_i(k) + delta
#                 - beta/2 * x^{(r)}(k-1) - Historical_k) / (Gamma + beta/2)
# ============================================================
def _gamma_coef(sigma, c):
    """Gamma = M(σ) · λ · c^σ / 2.

    For the RATIONAL normalization M(σ) = 1 / (1 - σ + σ*c) this is

        Gamma = (λ * c^σ) / (2 * (1 - σ + σ*c)).

    Note: under the exponential normalization M(σ) = c^(-σ) the
    product M(σ)*c^σ = 1, so Γ collapsed to λ/2 and c vanished from
    the response equation.  The rational form keeps c as an active
    degree of freedom, which is what the new specification requires.
    """
    lam, M_s, _, _, _ = _cpcf_constants(sigma, c)
    return 0.5 * M_s * lam * (c ** sigma)


def compute_historical_k(X_r, k, sigma, c=1.0, mu_0=None):
    """
    Pure historical-memory term built from x^{(r)}(0..k-1):
      Historical_k = I_k - Gamma * x^{(r)}(k)
    where x^{(r)}(0) is the caller-supplied ``mu_0`` (estimated by the
    extended OLS) and ``X_r`` carries x^{(r)}(1..k-1).

    The full integral I_k has the structure
        I_k = (M/sigma) * [sum + (1/2)*(Gamma_part * x^{(r)}(k)
                                       + current_partial * x^{(r)}(k-1))
                          + boundary(x^{(r)}(0))]
    and Gamma * x^{(r)}(k) is the part that depends on x^{(r)}(k). Subtracting
    it yields the historical-memory term.
    """
    if k < 2:
        return 0.0
    if mu_0 is None:
        mu_0 = X_r[0]  # fallback for backwards compatibility

    lam, M_s, Lambda_val, K0_s, K1_s = _cpcf_constants(sigma, c)

    # Sum over j = 1, ..., k-1.  x^{(r)}(j) and x^{(r)}(j-1) are known.
    # x^{(r)}(0) is mu_0; x^{(r)}(1..k-1) live in X_r[0..k-2].
    sum_val = 0.0
    for j in range(1, k):
        dj = X_r[j - 1]                  # x^{(r)}(j), j in 1..k-1
        dj_prev = mu_0 if j == 1 else X_r[j - 2]   # x^{(r)}(j-1)
        # (e^lam - 1) * e^{-lam*m} = e^{-lam*(m-1)} - e^{-lam*m}
        sum_dj = (
            (math.exp(-lam * (k - j - 1)) if lam > 0 else 1.0) -
            math.exp(-lam * (k - j))
        )
        sum_dj_prev = (
            math.exp(-lam * (k - j)) -
            math.exp(-lam * (k - j + 1))
        )
        sum_val += 0.5 * Lambda_val * (dj * sum_dj + dj_prev * sum_dj_prev)

    dk_prev = X_r[k - 2]               # x^{(r)}(k-1)
    current_partial = (1.0 - sigma - Lambda_val * math.exp(-lam)) * dk_prev

    # Boundary term (the integration-by-parts residual over [k-1, k]):
    #   - (M/sigma) * sigma * c^sigma * x^{(r)}(0) * ( e^{-lam*(k-1)} - e^{-lam*k} )
    #
    # Stripping only moves this coefficient into H(k) so that OLS can treat
    # x^{(r)}(0) as an unknown parameter. In the prediction phase, once mu_0
    # has been estimated, the FULL historical-memory term must include this
    # boundary contribution back -- otherwise the initial momentum leaks out.
    #
    # The same (e^lam - 1) * e^{-lam*k} factor identity used above lets us
    # write (e^{-lam*(k-1)} - e^{-lam*k}) = (e^lam - 1) * e^{-lam*k}.
    boundary_term = (M_s / sigma) * (-K0_s) * mu_0 * (
        (math.exp(-lam * (k - 1)) if lam > 0 else 1.0) -
        math.exp(-lam * k)
    )

    return (M_s / sigma) * (sum_val + 0.5 * current_partial) + boundary_term


def discrete_recursive_response(X_r, n_total, sigma, cq, P, thetas, taus):
    """
    Build the predicted r-order sequence {x^{(r)}(k)} for k = 0..n_total
    using the pure discrete recursive response function.

    Initial conditions:
      x^{(r)}(0) = mu_0 = P[-1]
      x^{(r)}(1) = X_r[0]  (observed anchor)

    Returns
    -------
    X_r_pred : ndarray of length n_total+1, indexed by k = 0..n_total.
    """
    global N_PULSES
    N = N_PULSES

    beta = P[0]
    gammas = P[1:1 + N] if N > 0 else np.array([])
    delta = P[N + 1]
    mu_0 = P[N + 2]

    Gamma = _gamma_coef(sigma, cq)

    X_r_pred = np.zeros(n_total + 1, dtype=np.float64)
    X_r_pred[0] = mu_0
    X_r_pred[1] = X_r[0]

    for k in range(2, n_total + 1):
        # X_r_pred[1..k-1] is the known history (X_r_pred[0] = mu_0 is handled
        # by the mu_0 argument inside compute_historical_k).
        hist = compute_historical_k(
            X_r_pred[1:k], k, sigma, cq, mu_0=mu_0
        )
        rhs = delta + sum(b * compute_Psi_i(k, theta, tau)
                          for b, theta, tau in zip(gammas, thetas, taus))
        lhs_extra = 0.5 * beta * X_r_pred[k - 1] + hist
        denom = Gamma + 0.5 * beta
        if abs(denom) < 1e-15:
            denom = 1e-15 if denom >= 0 else -1e-15
        X_r_pred[k] = (rhs - lhs_extra) / denom
    return X_r_pred


# ============================================================
# Fitness function for EJCPCFGM N-impulse model
# ============================================================
def fitness_function(vars_tuple, X_train, n_test=0,
                    ridge_lambda=1e-7, theta_lb=-5.0, theta_ub=0.0,
                    sigma_lb=0.0, sigma_ub=1.0,
                    cq_lb=0.0, cq_ub=2.0,
                    r_lb=0.0, r_ub=1.0):
    global N_PULSES
    N = N_PULSES

    sigma = vars_tuple[0]
    cq = vars_tuple[1]
    r = vars_tuple[2]
    thetas = list(vars_tuple[3:3 + N])
    taus_norms = list(vars_tuple[3 + N:3 + 2 * N])

    if not (sigma_lb < sigma < sigma_ub and cq_lb <= cq <= cq_ub and r_lb < r < r_ub):
        return 1e10
    for theta_i in thetas:
        if not (theta_lb <= theta_i <= theta_ub):
            return 1e10

    n_train = len(X_train)
    if n_train < 1:
        return 1e10
    # Each tau_j is confined to its own evenly-divided segment of the training set.
    # Example: N=2, n_train=12 -> tau_1∈[1,6], tau_2∈[7,12]
    if N_PULSES > 0:
        taus = compute_tau_positions(taus_norms, n_train)
    else:
        taus = []  # no impulses for N=0

    try:
        mape, _ = evaluate_model(X_train, sigma, cq, r, thetas, taus, ridge_lambda=ridge_lambda)
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
def optimize_dai(train_data, n_test=0, pop_size=30, max_iter=500,
                 ridge_lambda=1e-7,
                 theta_lb=-5.0, theta_ub=0.0,
                 sigma_lb=0.0, sigma_ub=1.0,
                 cq_lb=0.0, cq_ub=2.0,
                 r_lb=0.0, r_ub=1.0):
    global N_PULSES
    N = N_PULSES
    D = 3 + 2 * N
    NP = pop_size

    lb = np.zeros(D, dtype=float)
    ub = np.zeros(D, dtype=float)
    lb[0] = sigma_lb; ub[0] = sigma_ub
    lb[1] = cq_lb; ub[1] = cq_ub
    lb[2] = r_lb; ub[2] = r_ub
    for j in range(N):
        lb[3 + j] = theta_lb; ub[3 + j] = theta_ub
    for j in range(N):
        lb[3 + N + j] = 0.0; ub[3 + N + j] = 1.0

    Q = 0.7
    F = 0.5

    population = np.zeros((NP, D), dtype=float)
    for i in range(NP):
        population[i] = lb + np.random.rand(D) * (ub - lb)

    fitness = np.zeros(NP)
    for i in range(NP):
        fitness[i] = fitness_function(tuple(population[i]), train_data, n_test,
                                     ridge_lambda=ridge_lambda,
                                     theta_lb=theta_lb, theta_ub=theta_ub,
                                     sigma_lb=sigma_lb, sigma_ub=sigma_ub,
                                     cq_lb=cq_lb, cq_ub=cq_ub,
                                     r_lb=r_lb, r_ub=r_ub)

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx].copy()

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

        mutant_population = np.zeros((NP, D), dtype=float)
        for i in range(NP):
            candidates = list(range(NP))
            candidates.remove(i)
            r_indices = np.random.choice(candidates, 3, replace=False)
            mutant_population[i] = population[r_indices[0]] + F * (population[r_indices[1]] - population[r_indices[2]])

        for i in range(NP):
            for j in range(D):
                if mutant_population[i, j] < lb[j]:
                    mutant_population[i, j] = lb[j] + np.random.rand() * (population[i, j] - lb[j])
                elif mutant_population[i, j] > ub[j]:
                    mutant_population[i, j] = ub[j] - np.random.rand() * (ub[j] - population[i, j])

        new_population = np.zeros((NP, D), dtype=float)
        new_fitness = np.zeros(NP)

        for i in range(NP):
            trial = dai_crossover(population[i], mutant_population[i], dai_cr_matrix[i], D)
            trial = np.clip(trial, lb, ub)
            trial_fitness = fitness_function(tuple(trial), train_data, n_test,
                                           ridge_lambda=ridge_lambda,
                                           theta_lb=theta_lb, theta_ub=theta_ub,
                                           sigma_lb=sigma_lb, sigma_ub=sigma_ub,
                                           cq_lb=cq_lb, cq_ub=cq_ub,
                                           r_lb=r_lb, r_ub=r_ub)

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

    return best_solution, best_fitness


# ============================================================
# Evaluate EJCPCFGM on training data
# ============================================================
def evaluate_model(X_train, sigma, cq, r, thetas, taus, ridge_lambda=1e-7):
    global N_PULSES
    N = N_PULSES

    X_origin = np.array(X_train, dtype=np.float64)
    n = len(X_origin)

    x1 = X_origin[0]
    X_norm = X_origin / x1

    X_r = cpcfago_forward(X_norm, r, cq)
    P = solve_ols_multi_pulse(X_r, thetas, taus, sigma, cq, ridge_lambda=ridge_lambda)

    # Discrete recursive response: predict x^{(r)}(k) for k = 0..n
    X_r_pred_full = discrete_recursive_response(X_r, n, sigma, cq, P, thetas, taus)
    predictions_r = X_r_pred_full[1:]  # k = 1..n

    predictions_norm = cpcfiago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)

    mape = np.mean(np.abs((X_origin[1:] - predictions[1:]) / X_origin[1:])) * 100.0
    return mape, P


def predict_model(X_train, X_test, sigma, cq, r, thetas, taus, ridge_lambda=1e-7):
    global N_PULSES
    N = N_PULSES

    X_origin_train = np.array(X_train, dtype=np.float64)
    n_train = len(X_origin_train)
    n_total = n_train + len(X_test)

    x1 = X_origin_train[0]
    X_norm_train = X_origin_train / x1

    X_r_train = cpcfago_forward(X_norm_train, r, cq)
    P = solve_ols_multi_pulse(X_r_train, thetas, taus, sigma, cq, ridge_lambda=ridge_lambda)

    # Discrete recursive response: predict x^{(r)}(k) for k = 0..n_total
    X_r_pred_full = discrete_recursive_response(
        X_r_train, n_total, sigma, cq, P, thetas, taus
    )
    predictions_r = X_r_pred_full[1:]  # k = 1..n_total

    predictions_norm = cpcfiago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)
    return predictions, P


def run_ejcpcfgm_fixed(train_data, test_data, forecast_data=None,
                       sigma=None, cq=None, r=None, thetas=None, taus=None,
                       ridge_lambda=1e-7):
    """
    Run EJCPCFGM with pre-fixed parameters (from optimization).

    Parameters:
    -----------
    train_data : list
        Training data (2010-2021, 12 years)
    test_data : list
        Test data (2022-2024, 3 years)
    forecast_data : list, optional
        Future years data for comparison (e.g., 2025-2029)
    sigma, cq, r : float
        Pre-optimized parameters
    thetas, taus : list
        Pre-optimized impulse parameters
    ridge_lambda : float
        Ridge regularization parameter

    Returns:
    --------
    dict with predictions for train, test, and forecast periods
    """
    global N_PULSES
    N = N_PULSES

    X_origin_train = np.array(train_data, dtype=np.float64)
    n_train = len(X_origin_train)

    # Determine total prediction length
    if forecast_data is not None:
        n_forecast = len(forecast_data)
        n_total = n_train + len(test_data) + n_forecast
    elif test_data is not None:
        n_forecast = 0
        n_total = n_train + len(test_data)
    else:
        n_forecast = 0
        n_total = n_train

    x1 = X_origin_train[0]
    X_norm_train = X_origin_train / x1

    X_r_train = cpcfago_forward(X_norm_train, r, cq)
    P = solve_ols_multi_pulse(X_r_train, thetas, taus, sigma, cq, ridge_lambda=ridge_lambda)

    # Discrete recursive response
    X_r_pred_full = discrete_recursive_response(X_r_train, n_total, sigma, cq, P, thetas, taus)
    predictions_r = X_r_pred_full[1:]

    predictions_norm = cpcfiago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)

    # Split predictions
    train_predictions = predictions[:n_train]
    test_predictions = predictions[n_train:n_train + len(test_data)] if test_data else []
    forecast_predictions = predictions[n_train + len(test_data):] if forecast_data else []

    # Calculate MAPEs
    if len(test_data) > 0:
        test_mape = np.mean(np.abs((np.array(test_data) - test_predictions) / np.array(test_data))) * 100.0
    else:
        test_mape = None

    train_mape = np.mean(np.abs((np.array(train_data) - train_predictions) / np.array(train_data))) * 100.0

    return {
        'sigma': sigma,
        'cq': cq,
        'r': r,
        'thetas': thetas,
        'taus': taus,
        'beta': P[0],
        'bs': list(P[1:1 + N]) if N > 0 else [],
        'delta': P[N + 1],
        'mu_0': P[N + 2],
        'mape_s': train_mape,
        'mape_p': test_mape,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'forecast_predictions': forecast_predictions,
        'P': P,
    }


def run_ejcpcfgm(train_data, test_data, pop_size=20, max_iter=100,
                ridge_lambda=1e-7,
                theta_lb=-5.0, theta_ub=0.0,
                sigma_lb=0.0, sigma_ub=1.0,
                cq_lb=0.0, cq_ub=2.0,
                r_lb=0.0, r_ub=1.0):
    """
    Run EJCPCFGM with DAI optimization to find optimal parameters.

    Returns train and test predictions (no forecasting).
    """
    global N_PULSES
    N = N_PULSES

    best_params, best_mape = optimize_dai(
        train_data, len(test_data), pop_size, max_iter,
        ridge_lambda=ridge_lambda,
        theta_lb=theta_lb, theta_ub=theta_ub,
        sigma_lb=sigma_lb, sigma_ub=sigma_ub,
        cq_lb=cq_lb, cq_ub=cq_ub,
        r_lb=r_lb, r_ub=r_ub
    )

    sigma = best_params[0]
    cq = best_params[1]
    r = best_params[2]
    thetas = list(best_params[3:3 + N])
    taus_norms = list(best_params[3 + N:3 + 2 * N])
    n_train = len(train_data)
    taus = list(compute_tau_positions(taus_norms, n_train))

    predictions, P = predict_model(
        train_data, test_data, sigma, cq, r, thetas, taus,
        ridge_lambda=ridge_lambda
    )

    a = P[0]
    bs = P[1:1 + N] if N > 0 else np.array([])
    c = P[N + 1]
    mu_0 = P[N + 2]

    train_predictions = predictions[:len(train_data)]
    test_predictions = predictions[len(train_data):]

    train_mape = np.mean(np.abs((np.array(train_data) - train_predictions) / np.array(train_data))) * 100.0
    test_mape = np.mean(np.abs((np.array(test_data) - test_predictions) / np.array(test_data))) * 100.0

    print(f"N={N}  sigma={sigma:.4f} cq={cq:.4f} r={r:.4f}  "
          f"train={train_mape:.4f}%  test={test_mape:.4f}%")

    return {
        'sigma': sigma,
        'cq': cq,
        'r': r,
        'thetas': thetas,
        'taus': taus,
        'a': a,                  # legacy alias
        'beta': a,               # canonical name
        'bs': bs,                # legacy alias
        'gammas': list(bs),      # canonical name
        'c': c,                  # legacy alias
        'delta': c,              # canonical name
        'mu_0': mu_0,
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


def get_datasets():
    elec_data, gen_data = get_provincial_data()
    datasets = []
    for name in elec_data.keys():
        datasets.append((name, 'Consumption', '10^8 kWh', elec_data[name]))
        datasets.append((name, 'Generation', '10^8 kWh', gen_data[name]))
    return datasets
