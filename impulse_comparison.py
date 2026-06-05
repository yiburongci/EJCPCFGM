"""
EJCPCFGM(σ,1) Impulse Analysis - Different N Comparison
Only includes Jiangsu and Anhui provinces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
import warnings
import os

warnings.filterwarnings('ignore')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

TRAIN_YEARS = list(range(2010, 2022))
TEST_YEARS = list(range(2022, 2025))

OUTPUT_DIR = 'Impulse_Analysis_Results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROVINCE_MAP = {
    'Jiangsu': 'Jiangsu',
    'Anhui': 'Anhui'
}

JIANG_SU_GEN = [3359.18, 3762.50, 3928.40, 4320.68, 4347.57, 4361.00, 4709.37, 4914.74, 5085.08, 5166.43, 5217.54, 5968.89, 6077.31, 6390.53, 6807.37]
AN_HUI_GEN = [1443.85, 1635.35, 1767.50, 1970.04, 2033.91, 2062.00, 2252.69, 2456.28, 2734.49, 2886.67, 2808.98, 3083.39, 3298.77, 3549.45, 3863.46]

JIANG_SU_GEN_PARAMS = {
    'N': 2,
    'sigma': 0.851170,
    'cq': 0.821018,
    'r': 0.922361,
    'a': -0.261309,
    'c': -0.916400,
    'mape_s': 1.0603,
    'mape_p': 0.2023,
    'b': [0.400439, 0.107716],
    'theta': [-3.305917, -0.808059],
    'tau': [3.926936, 6.609563]
}

AN_HUI_GEN_PARAMS = {
    'N': 3,
    'sigma': 0.940099,
    'cq': 1.432646,
    'r': 0.844127,
    'a': -0.218089,
    'c': -0.706803,
    'mape_s': 0.7923,
    'mape_p': 0.5189,
    'b': [0.261740, 0.182852, 0.612698],
    'theta': [-1.104387, -0.564176, -2.485066],
    'tau': [3.500145, 6.612211, 8.777484]
}


def get_provincial_data():
    generation_data = {
        'Jiangsu': JIANG_SU_GEN,
        'Anhui': AN_HUI_GEN
    }
    return generation_data


def get_optimal_params():
    params = {
        ('Jiangsu', 'Gen'): JIANG_SU_GEN_PARAMS,
        ('Anhui', 'Gen'): AN_HUI_GEN_PARAMS
    }
    return params


def compute_core_constants(alpha, c=1.0):
    K1 = 1.0 - alpha
    K0 = alpha * (c ** alpha)
    if alpha > 0 and alpha < 1:
        lam = np.exp(-alpha / (1.0 - alpha))
    else:
        lam = 0.0
    if alpha > 0:
        W = (1.0 - lam) / alpha
    else:
        W = 1.0
    return K1, K0, lam, W


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


def compute_homologous_ago_forward(X, r, c=1.0):
    X = np.array(X, dtype=np.float64)
    n = len(X)
    L_ago = compute_cpcf_matrix(n, r, c)
    X_r = np.linalg.solve(L_ago, X)
    return X_r


def compute_homologous_ago_inverse(X_r, r, c=1.0):
    X_r = np.array(X_r, dtype=np.float64)
    n = len(X_r)
    L_ago = compute_cpcf_matrix(n, r, c)
    X = L_ago @ X_r
    return X


def compute_cpcf_derivative(X_r, sigma, c=1.0):
    X_r = np.array(X_r, dtype=np.float64)
    n = len(X_r)
    L_diff = compute_cpcf_matrix(n, sigma, c)
    D = L_diff @ X_r
    return D


def compute_Z_k(X_r, k):
    if k <= 1:
        return X_r[0]
    return 0.5 * (X_r[int(k) - 1] + X_r[int(k) - 2])


def compute_tau_positions(tau_array, n):
    tau_array = np.array(tau_array, dtype=np.float64)
    N = len(tau_array)
    taus = np.zeros(N, dtype=np.float64)
    for i in range(N):
        tau_start = 1.0 + (n - 1) * float(i) / float(N)
        tau_end = 1.0 + (n - 1) * float(i + 1) / float(N)
        taus[i] = tau_start + tau_array[i] * (tau_end - tau_start)
    return taus


def E_i(k, theta_i, tau_i):
    t = float(k) - 0.5
    if t < tau_i:
        return 0.0
    return np.exp(theta_i * (t - tau_i))


def solve_ols_multi_pulse(D, X_r, thetas, taus):
    n = len(X_r)
    k_start = 2
    n_eq = n - 1
    N = len(thetas)

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
            B[i, 1 + j] = E_i(k, thetas[j], taus[j])
        B[i, N + 1] = 1.0

    try:
        BT = B.T
        BT_B = BT @ B
        for j in range(n_cols):
            BT_B[j, j] += 1e-8
        P = np.linalg.solve(BT_B, BT @ Y).flatten()
        return P
    except:
        return None


def compute_Xt_multi_pulse_v2(t, sigma, cq, a, bs, c, thetas, taus, x1_r):
    if t < 1e-12:
        t = 1e-12

    N = len(thetas)
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

    f_t = c
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t >= tau_i:
            f_t += bi * np.exp(theta_i * (t - tau_i))

    exp_m_t = np.exp(-lam * t)
    direct = (K1 / A) * f_t
    I_d = c * ADB * (1.0 - exp_m_t)

    I_pulses = 0.0
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t < tau_i:
            continue
        denom = theta_i + lam
        exp_theta_tau = np.exp(theta_i * (t - tau_i))
        exp_m_t_tau = np.exp(-lam * (t - tau_i))
        I_E = (exp_theta_tau - exp_m_t_tau) / denom if abs(denom) > eps \
              else (t - tau_i) * exp_m_t_tau
        I_pulses += bi * I_E

    Phi_t = direct + C_val * (I_d + I_pulses)

    t1 = 1.0
    exp_m_1 = np.exp(-lam)

    f_t1 = c
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t1 >= tau_i:
            f_t1 += bi * np.exp(theta_i * (t1 - tau_i))
    direct_1 = (K1 / A) * f_t1

    I_d_1 = c * ADB * (1.0 - exp_m_1)

    I_pulses_1 = 0.0
    for bi, theta_i, tau_i in zip(bs, thetas, taus):
        if t1 < tau_i:
            continue
        denom = theta_i + lam
        exp_theta_tau_1 = np.exp(theta_i * (t1 - tau_i))
        exp_m_1_tau = np.exp(-lam * (t1 - tau_i))
        I_E_1 = (exp_theta_tau_1 - exp_m_1_tau) / denom if abs(denom) > eps \
                else (t1 - tau_i) * exp_m_1_tau
        I_pulses_1 += bi * I_E_1

    Phi_1 = direct_1 + C_val * (I_d_1 + I_pulses_1)
    correction = (x1_r - Phi_1) * np.exp(-lam * (t - 1.0))

    return Phi_t + correction


def predict_with_params(X_train, X_test, sigma, cq, r, thetas, taus, a, bs_active, c):
    X_origin_train = np.array(X_train, dtype=np.float64)
    n_train = len(X_origin_train)
    n_total = n_train + len(X_test)

    x1 = X_origin_train[0]
    X_norm_train = X_origin_train / x1

    X_r_train = compute_homologous_ago_forward(X_norm_train, r, cq)
    x1_r = X_r_train[0]

    predictions_r = np.zeros(n_total, dtype=np.float64)
    for k in range(1, n_total + 1):
        t = float(k)
        predictions_r[k - 1] = compute_Xt_multi_pulse_v2(
            t, sigma, cq, a, bs_active, c, thetas, taus, x1_r
        )

    predictions_norm = compute_homologous_ago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)

    train_predictions = predictions[:n_train]
    test_predictions = predictions[n_train:]

    mape_s = np.mean(np.abs((X_origin_train - train_predictions) / X_origin_train)) * 100.0
    mape_p = np.mean(np.abs((np.array(X_test) - test_predictions) / np.array(X_test))) * 100.0

    return train_predictions, test_predictions, {'mape_s': mape_s, 'mape_p': mape_p}


def add_radial_gradient(ax, cmap_name='Blues'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)

    y_min = ylim[0]
    y_max = ylim[1]
    Y_norm = (Y - y_min) / (y_max - y_min)

    intensity = Y_norm ** 0.7
    intensity = np.clip(intensity, 0.05, 0.8)

    cmap = plt.cm.get_cmap(cmap_name)
    Z = intensity * 0.6
    rgba = cmap(Z)
    rgba[..., 3] = intensity * 0.4

    ax.imshow(rgba, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
             origin='lower', aspect='auto', zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def predict_future(X_train, X_test, sigma, cq, r, thetas, taus, a, bs_active, c, n_years=4):
    X_origin_train = np.array(X_train, dtype=np.float64)
    n_train = len(X_origin_train)
    n_total = n_train + len(X_test) + n_years

    x1 = X_origin_train[0]
    X_norm_train = X_origin_train / x1

    X_r_train = compute_homologous_ago_forward(X_norm_train, r, cq)
    x1_r = X_r_train[0]

    predictions_r = np.zeros(n_total, dtype=np.float64)
    for k in range(1, n_total + 1):
        t = float(k)
        predictions_r[k - 1] = compute_Xt_multi_pulse_v2(
            t, sigma, cq, a, bs_active, c, thetas, taus, x1_r
        )

    predictions_norm = compute_homologous_ago_inverse(predictions_r, r, cq)
    predictions = predictions_norm * x1
    predictions = np.maximum(predictions, 0.0)

    future_start = n_train + len(X_test)
    return predictions[future_start:]


def plot_impulse_comparison(province, indicator, train_data, test_data, all_results, save_path, future_predictions=None):
    n_curves = len(all_results)

    fig = plt.figure(figsize=(14, 9), dpi=200)
    fig.patch.set_facecolor('white')

    ax = fig.add_subplot(1, 1, 1)

    province_en = PROVINCE_MAP.get(province, province)
    indicator_en = 'Generation'
    fig.suptitle(f'{province_en} - {indicator_en}', fontsize=22, fontweight='bold',
                 color='#333333', y=0.98)

    train_years = TRAIN_YEARS
    test_years = TEST_YEARS
    all_years = train_years + test_years
    all_data = list(train_data) + list(test_data)

    future_years = None
    if future_predictions is not None and len(future_predictions) > 0:
        future_years = list(range(2025, 2025 + len(future_predictions[0])))
        all_years_extended = all_years + future_years
        all_data_extended = all_data + list(future_predictions[0])
    else:
        all_years_extended = all_years
        all_data_extended = all_data

    y_max = max(all_data_extended) * 1.12
    y_min = min(all_data_extended) * 0.88

    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(2009.5, 2028.5)
    ax.set_ylim(y_min, y_max)

    add_radial_gradient(ax, 'Oranges')

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_curves))

    ax.plot(all_years, all_data, '-', color='#1a1a1a', linewidth=2.0,
            zorder=10, solid_capstyle='round', solid_joinstyle='round')

    for x, y in zip(all_years, all_data):
        ax.scatter(x+0.08, y-0.08*(y_max-y_min)/15, c='gray', s=60, zorder=9,
                  alpha=0.35, edgecolors='none')
        ax.scatter(x, y, c='white', s=60, zorder=10, edgecolors='#1a1a1a', linewidth=1.2)
        ax.scatter(x, y, c='#1a1a1a', s=40, zorder=11, edgecolors='none')

    opt_result = all_results[0]
    taus = opt_result.get('taus', [])
    thetas = opt_result.get('thetas', [])

    for idx, result in enumerate(all_results):
        n_active = result['n_active']
        train_pred = result['train_pred']
        test_pred = result['test_pred']
        mape_p = result['mape_p']
        mape_s = result['mape_s']

        if train_pred is None:
            continue

        color = colors[idx]

        train_pred_list = list(train_pred)
        test_pred_list = list(test_pred)
        future_pred_list = list(future_predictions[idx]) if future_predictions else []
        full_pred = train_pred_list + test_pred_list + future_pred_list
        full_years = all_years + future_years if future_years else all_years

        ax.plot(full_years, full_pred, '-', color=color, linewidth=4.0,
                alpha=0.95, zorder=5,
                path_effects=[pe.Stroke(linewidth=5.5, foreground='white'),
                              pe.Normal()])

        for x, y in zip(full_years, full_pred):
            ax.scatter(x+0.08, y-0.08*(y_max-y_min)/15, c='gray', s=80,
                      zorder=6, alpha=0.4, edgecolors='none')
            ax.scatter(x, y, c=color, s=70, marker='D', zorder=7,
                      edgecolors='white', linewidth=2, alpha=0.95)

    if len(taus) > 0:
        for i, (tau, theta) in enumerate(zip(taus, thetas)):
            tau_year = 2009 + tau
            closest_year = min(all_years, key=lambda x: abs(x - tau_year))
            idx = all_years.index(closest_year)
            y_val = all_data[idx]
            ax.scatter(closest_year, y_val, marker='*', s=700, c='red',
                      edgecolors='gold', linewidth=2.5, zorder=100, alpha=1.0)

    ax.axvline(x=2021.5, color='#666666', linestyle='--', alpha=0.7, linewidth=2, zorder=3)
    ax.axvline(x=2024.5, color='#333333', linestyle=':', alpha=0.7, linewidth=2, zorder=3)

    ax.text(2015.5, y_max * 0.96, 'Training', ha='center', fontsize=14,
            color='#444444', style='italic', fontweight='bold', zorder=15)
    ax.text(2023, y_max * 0.96, 'Testing', ha='center', fontsize=14,
            color='#444444', style='italic', fontweight='bold', zorder=15)
    if future_predictions is not None:
        ax.text(2026.5, y_max * 0.96, 'Forecast', ha='center', fontsize=14,
                color='#333333', style='italic', fontweight='bold', zorder=15)

    legend_elements = []
    for idx, result in enumerate(all_results):
        n_active = result['n_active']
        mape_p = result['mape_p']
        mape_s = result['mape_s']
        color = colors[idx]

        if n_active == 0:
            label = f'N={n_active} (MAPE_s={mape_s:.2f}%, MAPE_p={mape_p:.2f}%)'
        else:
            label = f'N={n_active} (MAPE_s={mape_s:.2f}%, MAPE_p={mape_p:.2f}%)'

        line = plt.Line2D([0], [0], color=color, linewidth=3.5, label=label)
        legend_elements.append(line)

    if len(taus) > 0:
        imp_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                               markersize=18, markeredgecolor='gold', markeredgewidth=2,
                               linestyle='None', label=f'Impulse Point (N={len(taus)})')
        legend_elements.append(imp_marker)

    legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=13,
                      framealpha=0.95, edgecolor='gray', fancybox=True)

    ax.set_xlabel('Year', fontsize=16, color='#333333', fontweight='bold')
    ax.set_ylabel('Electricity Generation (10$^8$ kWh)', fontsize=16, color='#333333', fontweight='bold')

    ax.tick_params(colors='#333333', labelsize=13)
    ax.set_xticks([2010, 2015, 2020, 2025])

    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(2)

    ax.grid(True, linestyle='-', alpha=0.15, color='#888888', zorder=1)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f'Saved: {save_path}')


def run_analysis_for_province(province, indicator, train_data, test_data, opt_params):
    N_max = opt_params['N']
    sigma = opt_params['sigma']
    cq = opt_params['cq']
    r = opt_params['r']
    thetas = opt_params['theta']
    taus = opt_params['tau']
    a = opt_params['a']
    c = opt_params['c']
    b_original = opt_params['b']

    results = []

    for n_active in range(N_max, -1, -1):
        if n_active == 0:
            bs_active = np.zeros(len(b_original))
        else:
            bs_active = np.zeros(len(b_original))
            for i in range(min(n_active, len(b_original))):
                bs_active[i] = b_original[i]

        train_pred, test_pred, mapes = predict_with_params(
            train_data, test_data, sigma, cq, r, thetas, taus, a, bs_active, c
        )

        if mapes is not None:
            result = {
                'n_active': n_active,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'mape_s': mapes['mape_s'],
                'mape_p': mapes['mape_p'],
                'taus': taus[:n_active] if n_active > 0 else [],
                'thetas': thetas[:n_active] if n_active > 0 else []
            }
            results.append(result)

    return results


def main():
    print("=" * 80)
    print("EJCPCFGM(sigma,1) Impulse Impact Analysis")
    print("Only includes Jiangsu and Anhui provinces - Generation only")
    print("=" * 80)

    print(f"\nOutput directory: {OUTPUT_DIR}/")

    generation_data = get_provincial_data()
    opt_params = get_optimal_params()

    all_results_data = {}

    for province in ['Jiangsu', 'Anhui']:
        key = (province, 'Gen')

        if key not in opt_params:
            print(f"Warning: No optimal params for {province}-Gen")
            continue

        full_data = generation_data[province]
        train_data = full_data[:12]
        test_data = full_data[12:]

        print(f"\nProcessing: {province} - Generation")
        print(f"  Optimal N = {opt_params[key]['N']}")

        results = run_analysis_for_province(
            province, 'Gen', train_data, test_data, opt_params[key]
        )

        opt = opt_params[key]
        sigma, cq, r = opt['sigma'], opt['cq'], opt['r']
        thetas, taus = opt['theta'], opt['tau']
        a, c = opt['a'], opt['c']
        b_original = opt['b']
        N_max = opt['N']

        future_preds_all = []
        for n_active in range(N_max, -1, -1):
            if n_active == 0:
                bs_active = np.zeros(len(b_original))
            else:
                bs_active = np.zeros(len(b_original))
                for i in range(min(n_active, len(b_original))):
                    bs_active[i] = b_original[i]

            future_pred = predict_future(train_data, test_data, sigma, cq, r, thetas, taus, a, bs_active, c, n_years=4)
            future_preds_all.append(list(future_pred))

        all_results_data[key] = {'impulse_results': results, 'future_preds': future_preds_all}

        import pandas as pd
        province_en = PROVINCE_MAP[province]
        filename = f'{province_en}_Generation_impulse_analysis.csv'
        filepath = os.path.join(OUTPUT_DIR, filename)

        rows = []
        for idx, r in enumerate(results):
            row = {
                'N_active': r['n_active'],
                'MAPE_s(%)': round(r['mape_s'], 4),
                'MAPE_p(%)': round(r['mape_p'], 4)
            }
            for i, v in enumerate(r['train_pred']):
                row[f'Train_{2010+i}'] = round(v, 2)
            for i, v in enumerate(r['test_pred']):
                row[f'Test_{2022+i}'] = round(v, 2)
            if idx < len(future_preds_all):
                for i, v in enumerate(future_preds_all[idx]):
                    row[f'Forecast_{2025+i}'] = round(v, 2)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"  Results saved to: {filepath}")

    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)

    for key, data_container in all_results_data.items():
        province, indicator = key
        province_en = PROVINCE_MAP[province]

        full_data = generation_data[province]
        train_data = full_data[:12]
        test_data = full_data[12:]

        results = data_container['impulse_results']
        future_preds_all = data_container['future_preds']

        save_path = os.path.join(OUTPUT_DIR, f'{province_en}_Generation_impulse_comparison.png')

        plot_impulse_comparison(province, indicator, train_data, test_data, results, save_path,
                              future_predictions=future_preds_all)

    print("\n" + "=" * 80)
    print("Summary: Impact of Impulse Terms on MAPE_p (%)")
    print("=" * 80)

    header = f"{'Province-Generation':<30}"
    for key in all_results_data:
        N_max = opt_params[key]['N']
        for n in range(N_max, -1, -1):
            header += f"{'N='+str(n):>10}"
        break
    print(header)
    print("-" * len(header))

    for key, data_container in all_results_data.items():
        results = data_container['impulse_results']
        province_en = PROVINCE_MAP[key[0]]
        label = f"{province_en}-Generation"
        row = f"{label:<30}"

        N_max = opt_params[key]['N']
        mape_dict = {r['n_active']: r['mape_p'] for r in results}

        for n in range(N_max, -1, -1):
            mape = mape_dict.get(n, None)
            if mape is not None:
                row += f"{mape:>9.2f}% "
            else:
                row += f"{'N/A':>10}"
        print(row)

    print("=" * 80)
    print(f"\nAnalysis complete! All results saved to {OUTPUT_DIR}/ folder")
    print("=" * 80)


if __name__ == "__main__":
    main()
