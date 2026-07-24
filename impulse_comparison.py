"""EJCPCFGM impulse analysis - drop impulses one by one, see effect on MAPE."""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

import ejcpcfgm_model

OUT = 'Impulse_Analysis_Results'
os.makedirs(OUT, exist_ok=True)

TRAIN_YEARS = list(range(2010, 2022))
TEST_YEARS = list(range(2022, 2025))
FORECAST_YEARS = list(range(2025, 2029))
ALL_YEARS = TRAIN_YEARS + TEST_YEARS + FORECAST_YEARS
FCST_N = 4

PARAMS = {
    'Jiangsu_Consumption': {
        'train': [3864.00, 4281.62, 4580.90, 4956.60, 5012.54, 5114.70,
                  5458.95, 5807.89, 6128.27, 6264.00, 6374.00, 7101.00],
        'test':  [7400.00, 7833.00, 8486.93],
        'unit': '10$^8$ kWh', 'province': 'Jiangsu', 'indicator': 'Consumption',
        'sigma': 0.569226, 'cq': 1.000695, 'r': 0.787209, 'beta': -0.802443,
        'gammas': [0.064682, 0.082418, -0.047629], 'delta': -0.130997,
        'thetas': [-3.039530, -4.450782, -2.243019], 'taus': [3.6973, 7.8770, 9.9782],
        'ridge_lambda': 1e-4,
    },
    'Jiangsu_Generation': {
        'train': [3359.18, 3762.50, 3928.40, 4320.68, 4347.57, 4361.00,
                  4709.37, 4914.74, 5085.08, 5166.43, 5217.54, 5968.89],
        'test':  [6077.31, 6390.53, 6807.37],
        'unit': '10$^8$ kWh', 'province': 'Jiangsu', 'indicator': 'Generation',
        'sigma': 0.584791, 'cq': 0.979412, 'r': 0.883509, 'beta': -0.767212,
        'gammas': [-0.410004, -0.049093, -0.293859], 'delta': -0.274805,
        'thetas': [-3.842468, -2.321793, -5.548332], 'taus': [1.7145, 5.4386, 9.9381],
        'ridge_lambda': 1e-9,
    },
    'Anhui_Consumption': {
        'train': [1078.00, 1221.19, 1361.10, 1528.10, 1585.18, 1639.79,
                  1794.98, 1921.48, 2135.07, 2301.00, 2428.00, 2715.00],
        'test':  [2993.00, 3214.00, 3597.86],
        'unit': '10$^8$ kWh', 'province': 'Anhui', 'indicator': 'Consumption',
        'sigma': 0.535735, 'cq': 0.504678, 'r': 0.868266, 'beta': -1.169766,
        'gammas': [-0.017114, -0.002261, 0.010280], 'delta': 0.024993,
        'thetas': [-2.770816, -0.222741, -2.669932], 'taus': [3.7366, 8.2288, 10.6323],
        'ridge_lambda': 1e-9,
    },
    'Anhui_Generation': {
        'train': [1443.85, 1635.35, 1767.50, 1970.04, 2033.91, 2062.00,
                  2252.69, 2456.28, 2734.49, 2886.67, 2808.98, 3083.39],
        'test':  [3298.77, 3549.45, 3863.46],
        'unit': '10$^8$ kWh', 'province': 'Anhui', 'indicator': 'Generation',
        'sigma': 0.582981, 'cq': 0.687953, 'r': 0.870417, 'beta': -0.924513,
        'gammas': [0.122975, 0.030213, -0.048822], 'delta': -0.193216,
        'thetas': [-3.789682, -0.280118, -0.000000], 'taus': [3.8033, 7.6843, 10.1200],
        'ridge_lambda': 1e-6,
    },
}


def run_active(p, n_active):
    all_thetas, all_taus, all_gammas = p['thetas'], p['taus'], p['gammas']
    order = np.argsort(all_taus)
    keep = sorted(order[:n_active])
    kept_t = [all_thetas[i] for i in keep]
    kept_tau = [all_taus[i] for i in keep]
    kept_g = [all_gammas[i] for i in keep]

    X0 = np.array(p['train'], dtype=np.float64)
    n_train, n_test = len(X0), len(p['test'])
    n_total = n_train + n_test + FCST_N
    x1 = X0[0]
    Xn = X0 / x1
    Xr = ejcpcfgm_model.cpcfago_forward(Xn, p['r'], p['cq'])

    ejcpcfgm_model.N_PULSES = len(all_taus)
    full = ejcpcfgm_model.run_ejcpcfgm_fixed(
        p['train'], p['test'], forecast_data=[0] * FCST_N,
        sigma=p['sigma'], cq=p['cq'], r=p['r'],
        thetas=list(all_thetas), taus=list(all_taus),
        ridge_lambda=p['ridge_lambda'],
    )
    P = full['P'].copy()
    for idx in order[n_active:]:
        pos = list(order).index(idx)
        P[1 + pos] = 0.0

    Xr_pred = ejcpcfgm_model.discrete_recursive_response(
        Xr, n_total, p['sigma'], p['cq'], P, list(all_thetas), list(all_taus)
    )
    pr = Xr_pred[1:]
    pred = np.maximum(ejcpcfgm_model.cpcfiago_inverse(pr, p['r'], p['cq']) * x1, 0.0)
    tr_pred, te_pred, fc_pred = pred[:n_train], pred[n_train:n_train + n_test], pred[n_train + n_test:]

    mape_s = float(np.mean(np.abs((X0[1:] - tr_pred[1:]) / X0[1:])) * 100)
    mape_p = float(np.mean(np.abs((np.array(p['test']) - te_pred) / np.array(p['test']))) * 100)
    return dict(mape_s=mape_s, mape_p=mape_p, train=tr_pred, test=te_pred, fc=fc_pred,
                taus=kept_tau, thetas=kept_t, gammas=kept_g)


def add_gradient(ax, cmap_name):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)
    intensity = np.clip(((Y - ylim[0]) / (ylim[1] - ylim[0])) ** 0.7, 0.05, 0.8)
    cmap = plt.cm.get_cmap(cmap_name)
    rgba = cmap(intensity * 0.6)
    rgba[..., 3] = intensity * 0.4
    ax.imshow(rgba, extent=[*xlim, *ylim], origin='lower', aspect='auto', zorder=0)
    ax.set_xlim(xlim); ax.set_ylim(ylim)


def cmap_for(indicator):
    return 'Oranges' if 'Consumption' in indicator else 'Blues' if 'Generation' in indicator else 'Greys'


def plot(p, results, fp):
    train, test = p['train'], p['test']
    all_actual = train + test
    n_total = len(p['taus'])
    n_list = list(range(n_total, -1, -1))

    all_pred = []
    for n in n_list:
        r = results[n]
        all_pred += list(r['train']) + list(r['test']) + list(r['fc'])
    y_max = max(max(all_actual), max(all_pred)) * 1.15
    y_min = min(min(all_actual), min(all_pred)) * 0.85

    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#FAFAFA')
    ax.set_xlim(2009.5, 2028.5)
    ax.set_ylim(y_min, y_max)
    add_gradient(ax, cmap_for(p['indicator']))

    cmap_base = plt.cm.viridis
    colors = [cmap_base(0.15 + 0.75 * i / max(len(n_list) - 1, 1)) for i in range(len(n_list))]

    ax.plot(TRAIN_YEARS + TEST_YEARS, all_actual, '-', color='#1a1a1a', linewidth=2.5, zorder=15,
            label='Actual')
    for x, y in zip(TRAIN_YEARS + TEST_YEARS, all_actual):
        ax.scatter(x, y, c='#1a1a1a', s=45, zorder=18)

    for idx, n in enumerate(n_list):
        r = results[n]
        full = list(r['train']) + list(r['test']) + list(r['fc'])
        ax.plot(ALL_YEARS, full, '-', color=colors[idx], linewidth=3.5, alpha=0.95, zorder=5,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='white'), pe.Normal()])
        for x, y in zip(TRAIN_YEARS, r['train']):
            ax.scatter(x, y, c=colors[idx], s=50, marker='D', zorder=8)
        for x, y in zip(TEST_YEARS, r['test']):
            ax.scatter(x, y, c=colors[idx], s=50, marker='D', zorder=8)
        for x, y in zip(FORECAST_YEARS, r['fc']):
            ax.scatter(x, y, c=colors[idx], s=50, marker='s', zorder=9, alpha=0.8)

    full_res = results[n_total]
    full_tr = list(full_res['train'])
    for tau in p['taus']:
        lo, hi = int(np.floor(tau)) - 1, int(np.ceil(tau)) - 1
        if 0 <= lo < len(full_tr) and 0 <= hi < len(full_tr):
            frac = tau - np.floor(tau)
            y_val = full_tr[lo] + frac * (full_tr[hi] - full_tr[lo])
            year = 2010 + (tau - 1)
            ax.scatter(year, y_val, marker='*', s=350, c='red', zorder=25, edgecolors='darkred', linewidths=1.2)
            ax.axvline(x=year, color='red', linestyle=':', alpha=0.35, linewidth=1.2, zorder=2)

    ax.axvline(x=2021.5, color='#666', linestyle='--', alpha=0.7, linewidth=2, zorder=3)
    ax.axvline(x=2024.5, color='#666', linestyle='--', alpha=0.7, linewidth=2, zorder=3)
    ax.text(2015.5, y_max * 0.97, 'Training', ha='center', fontsize=12, color='#444',
            style='italic', fontweight='bold')
    ax.text(2023, y_max * 0.97, 'Testing', ha='center', fontsize=12, color='#444',
            style='italic', fontweight='bold')
    ax.text(2027.5, y_max * 0.97, 'Forecasting', ha='center', fontsize=12, color='#444',
            style='italic', fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], color=colors[i], linewidth=3.5,
                   label=f'N={n}  train={results[n]["mape_s"]:.2f}%  test={results[n]["mape_p"]:.2f}%')
        for i, n in enumerate(n_list)
    ]
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                      markeredgecolor='darkred', markersize=18, label='Impulse'))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 0.85),
              fontsize=12, framealpha=0.95, edgecolor='gray')

    ax.set_xlabel('Year', fontsize=15, fontweight='bold')
    ax.set_ylabel(f'{p["indicator"]} ({p["unit"]})', fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=12)
    ax.set_xticks([2010, 2015, 2020, 2025, 2028])
    ax.set_xticks(list(range(2010, 2029)), minor=True)
    ax.grid(True, which='minor', axis='x', linestyle=':', alpha=0.15, color='#888', zorder=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color('#333'); s.set_linewidth(2)

    fig.suptitle(f'{p["indicator"]}', fontsize=22, fontweight='bold', color='#333', y=0.98)
    plt.tight_layout()
    plt.savefig(fp, dpi=250, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    print(f"output: {OUT}/")
    all_ds = {}
    for name, p in PARAMS.items():
        print(f"\n[{name}]  N_optimal={len(p['taus'])}  sigma={p['sigma']:.4f} cq={p['cq']:.4f} r={p['r']:.4f}")
        results = {}
        for n in range(len(p['taus']), -1, -1):
            r = run_active(p, n)
            results[n] = r
            print(f"  N={n}  train={r['mape_s']:.4f}%  test={r['mape_p']:.4f}%")

        plot(p, results, os.path.join(OUT, f'{name}.png'))

        with open(os.path.join(OUT, f'{name}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['N_Active', 'MAPE_s(%)', 'MAPE_p(%)', 'sigma', 'cq', 'r', 'beta', 'delta',
                        'thetas', 'taus', 'gammas'])
            for n in range(len(p['taus']), -1, -1):
                r = results[n]
                w.writerow([n, f"{r['mape_s']:.4f}", f"{r['mape_p']:.4f}",
                            f"{p['sigma']:.6f}", f"{p['cq']:.6f}", f"{p['r']:.6f}",
                            f"{p['beta']:.6f}", f"{p['delta']:.6f}",
                            str(r['thetas']), str(r['taus']), str(r['gammas'])])
        all_ds[name] = results

    print("\nMAPE_p by removed impulses (newest first):")
    header = f"{'dataset':<25}"
    max_n = max(len(p['taus']) for p in PARAMS.values())
    for i in range(max_n, -1, -1):
        header += f"  N={i:>2}"
    print(header)
    for name, p in PARAMS.items():
        row = f"{name:<25}"
        for n in range(len(p['taus']), -1, -1):
            row += f"  {all_ds[name][n]['mape_p']:>5.2f}%"
        print(row)
    print("\ndone.")


if __name__ == "__main__":
    main()
