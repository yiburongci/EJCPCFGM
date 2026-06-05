"""
8 Methods Comparison Analysis (with EJCPCFGM(σ,1))
Only includes Jiangsu and Anhui provinces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import warnings
import os

warnings.filterwarnings('ignore')
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False

TRAIN_YEARS = list(range(2010, 2022))
TEST_YEARS = list(range(2022, 2025))
ALL_YEARS = TRAIN_YEARS + TEST_YEARS

METHOD_NAMES = ['EJCPCFGM(σ,1)', 'ARIMA', 'GM(1,1)', 'FAGM(1,1)', 'JFGM(r,s,ad,1)', 'NGBM(1,1)', 'LSSVR', 'LSTM']

METHOD_TITLES = {
    'EJCPCFGM(σ,1)': r'EJCPCFGM($\sigma$, 1)',
    'ARIMA': 'ARIMA',
    'GM(1,1)': 'GM(1,1)',
    'FAGM(1,1)': 'FAGM(1,1)',
    'JFGM(r,s,ad,1)': r'JFGM($r$,$s$,$ad$,1)',
    'NGBM(1,1)': 'NGBM(1,1)',
    'LSSVR': 'LSSVR',
    'LSTM': 'LSTM'
}

METHOD_COLORS = [
    '#0066CC',
    '#CC3300',
    '#006633',
    '#660099',
    '#CC9900',
    '#0099CC',
    '#FF6666',
    '#FF1493',
]

GRADIENT_CMAPS = [
    'Blues',
    'OrRd',
    'Greens',
    'Purples',
    'YlOrBr',
    'BuPu',
    'Reds',
    'RdPu',
]

OUTPUT_DIR = 'Comparison_Results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


JIANG_SU_ELEC_TRAIN = [3864.00, 4281.62, 4580.90, 4956.60, 5012.54, 5114.70, 5458.95, 5807.89, 6128.27, 6264.00, 6374.00, 7101.00]
JIANG_SU_ELEC_TEST = [7400.00, 7833.00, 8486.93]

JIANG_SU_GEN_TRAIN = [3359.18, 3762.50, 3928.40, 4320.68, 4347.57, 4361.00, 4709.37, 4914.74, 5085.08, 5166.43, 5217.54, 5968.89]
JIANG_SU_GEN_TEST = [6077.31, 6390.53, 6807.37]

AN_HUI_ELEC_TRAIN = [1078.00, 1221.19, 1361.10, 1528.10, 1585.18, 1639.79, 1794.98, 1921.48, 2135.07, 2301.00, 2428.00, 2715.00]
AN_HUI_ELEC_TEST = [2993.00, 3214.00, 3597.86]

AN_HUI_GEN_TRAIN = [1443.85, 1635.35, 1767.50, 1970.04, 2033.91, 2062.00, 2252.69, 2456.28, 2734.49, 2886.67, 2808.98, 3083.39]
AN_HUI_GEN_TEST = [3298.77, 3549.45, 3863.46]

JIANG_SU_ELEC_7M = {
    'ARIMA': [7388.73432878311, 7676.466937730422, 7964.197826852214],
    'GM(1,1)': [7249.81467541575, 7593.13946784148, 7952.722870779544],
    'FAGM(1,1)': [7294.351452172894, 7655.1200027342275, 8034.547224081136],
    'JFGM(r,s,ad,1)': [7130.981436426216, 7350.425436006854, 7562.637236704915],
    'NGBM(1,1)': [7185.382893251095, 7498.934129206798, 7824.599608642617],
    'LSSVR': [7229.9958801401135, 7474.736594518214, 7703.118730420734],
    'LSTM': [7283.2353515625, 7607.91357421875, 7918.06103515625]
}

JIANG_SU_GEN_7M = {
    'ARIMA': [6194.845587900419, 6420.783968343592, 6646.705142639939],
    'GM(1,1)': [5945.387002621952, 6189.862531741383, 6444.390944602666],
    'FAGM(1,1)': [5745.4317632462335, 5917.187760995272, 6090.192606964934],
    'JFGM(r,s,ad,1)': [5704.142928238412, 5792.819848092031, 5871.81949228004],
    'NGBM(1,1)': [5900.24509367509, 6124.514856371046, 6356.260353373844],
    'LSSVR': [5881.211632573809, 6046.010721802026, 6258.02589446504],
    'LSTM': [6012.54736328125, 6277.89453125, 6550.9580078125]
}

AN_HUI_ELEC_7M = {
    'ARIMA': [2906.6051322057097, 3096.1363096090063, 3283.6159809223686],
    'GM(1,1)': [2873.327257343306, 3100.5147074938286, 3345.66535253419],
    'FAGM(1,1)': [2956.5239948669005, 3221.1956095085225, 3511.025856858476],
    'JFGM(r,s,ad,1)': [2947.49836103533, 3206.181052359905, 3489.107147136696],
    'NGBM(1,1)': [2852.48561407176, 3068.79416027337, 3300.9130003223763],
    'LSSVR': [2871.915715923475, 3038.591837633166, 3148.4198632178127],
    'LSTM': [2974.8818359375, 3251.69970703125, 3514.572021484375]
}

AN_HUI_GEN_7M = {
    'ARIMA': [3226.673090043039, 3369.95546024994, 3513.23711062432],
    'GM(1,1)': [3323.5409590790805, 3536.023040897984, 3762.0896205913014],
    'FAGM(1,1)': [3255.368070615772, 3434.516358342131, 3621.8390844004753],
    'JFGM(r,s,ad,1)': [3496.0172281190316, 3753.3874135366787, 4030.005823773914],
    'NGBM(1,1)': [3283.063741747119, 3475.3254026697286, 3677.762220973418],
    'LSSVR': [3207.615067510485, 3308.202157550096, 3340.0970632672443],
    'LSTM': [3251.416748046875, 3378.757080078125, 3476.470703125]
}

JIANG_SU_ELEC_EJCPCFGM_TRAIN_FIT = [3864.00, 4281.5636, 4581.0841, 4818.8895, 5019.2523, 5223.3969, 5444.8243, 5807.8745, 6088.1660, 6320.9783, 6374.0037, 6971.5078]
JIANG_SU_ELEC_EJCPCFGM_TEST_PRE = [7384.6889, 7843.0550, 8352.1323]
JIANG_SU_ELEC_EJCPCFGM_MAPE_S = 0.7241
JIANG_SU_ELEC_EJCPCFGM_MAPE_P = 0.6412

JIANG_SU_GEN_EJCPCFGM_TRAIN_FIT = [3359.18, 3809.9678, 3902.7677, 4320.68, 4347.57, 4361.00, 4709.37, 4914.7391, 5045.8657, 5240.6237, 5479.3026, 5754.8955]
JIANG_SU_GEN_EJCPCFGM_TEST_PRE = [6066.2584, 6415.0864, 6804.5923]
JIANG_SU_GEN_EJCPCFGM_MAPE_S = 1.0603
JIANG_SU_GEN_EJCPCFGM_MAPE_P = 0.2023

AN_HUI_ELEC_EJCPCFGM_TRAIN_FIT = [1078.00, 1221.19, 1361.10, 1501.0252, 1585.18, 1681.4298, 1793.0858, 1921.48, 2131.4184, 2294.8866, 2490.2313, 2715.00]
AN_HUI_ELEC_EJCPCFGM_TEST_PRE = [2972.0277, 3265.2654, 3599.4838]
AN_HUI_ELEC_EJCPCFGM_MAPE_S = 0.6180
AN_HUI_ELEC_EJCPCFGM_MAPE_P = 0.7803

AN_HUI_GEN_EJCPCFGM_TRAIN_FIT = [1443.85, 1674.3599, 1728.6254, 1970.0399, 2043.5217, 2061.0053, 2252.69, 2444.0822, 2734.4896, 2886.6702, 2913.6580, 3088.9147]
AN_HUI_GEN_EJCPCFGM_TEST_PRE = [3301.4917, 3542.7013, 3813.8510]
AN_HUI_GEN_EJCPCFGM_MAPE_S = 0.7923
AN_HUI_GEN_EJCPCFGM_MAPE_P = 0.5189


def build_all_results():
    all_results = {}

    key = ('Jiangsu', 'Elec')
    all_results[key] = {
        'province': 'Jiangsu',
        'indicator': 'Elec',
        'train_actual': dict(zip(TRAIN_YEARS, JIANG_SU_ELEC_TRAIN)),
        'test_actual': dict(zip(TEST_YEARS, JIANG_SU_ELEC_TEST)),
        'methods': {
            'EJCPCFGM(σ,1)': {
                'train_fitted': dict(zip(TRAIN_YEARS, JIANG_SU_ELEC_EJCPCFGM_TRAIN_FIT)),
                'test_predicted': dict(zip(TEST_YEARS, JIANG_SU_ELEC_EJCPCFGM_TEST_PRE)),
                'full_fitted': dict(zip(ALL_YEARS, JIANG_SU_ELEC_EJCPCFGM_TRAIN_FIT + JIANG_SU_ELEC_EJCPCFGM_TEST_PRE)),
                'mape_s': JIANG_SU_ELEC_EJCPCFGM_MAPE_S,
                'mape_p': JIANG_SU_ELEC_EJCPCFGM_MAPE_P
            }
        }
    }
    for method, preds in JIANG_SU_ELEC_7M.items():
        all_results[key]['methods'][method] = {
            'train_fitted': {},
            'test_predicted': dict(zip(TEST_YEARS, preds)),
            'full_fitted': {},
            'mape_s': None,
            'mape_p': None
        }

    key = ('Jiangsu', 'Gen')
    all_results[key] = {
        'province': 'Jiangsu',
        'indicator': 'Gen',
        'train_actual': dict(zip(TRAIN_YEARS, JIANG_SU_GEN_TRAIN)),
        'test_actual': dict(zip(TEST_YEARS, JIANG_SU_GEN_TEST)),
        'methods': {
            'EJCPCFGM(σ,1)': {
                'train_fitted': dict(zip(TRAIN_YEARS, JIANG_SU_GEN_EJCPCFGM_TRAIN_FIT)),
                'test_predicted': dict(zip(TEST_YEARS, JIANG_SU_GEN_EJCPCFGM_TEST_PRE)),
                'full_fitted': dict(zip(ALL_YEARS, JIANG_SU_GEN_EJCPCFGM_TRAIN_FIT + JIANG_SU_GEN_EJCPCFGM_TEST_PRE)),
                'mape_s': JIANG_SU_GEN_EJCPCFGM_MAPE_S,
                'mape_p': JIANG_SU_GEN_EJCPCFGM_MAPE_P
            }
        }
    }
    for method, preds in JIANG_SU_GEN_7M.items():
        all_results[key]['methods'][method] = {
            'train_fitted': {},
            'test_predicted': dict(zip(TEST_YEARS, preds)),
            'full_fitted': {},
            'mape_s': None,
            'mape_p': None
        }

    key = ('Anhui', 'Elec')
    all_results[key] = {
        'province': 'Anhui',
        'indicator': 'Elec',
        'train_actual': dict(zip(TRAIN_YEARS, AN_HUI_ELEC_TRAIN)),
        'test_actual': dict(zip(TEST_YEARS, AN_HUI_ELEC_TEST)),
        'methods': {
            'EJCPCFGM(σ,1)': {
                'train_fitted': dict(zip(TRAIN_YEARS, AN_HUI_ELEC_EJCPCFGM_TRAIN_FIT)),
                'test_predicted': dict(zip(TEST_YEARS, AN_HUI_ELEC_EJCPCFGM_TEST_PRE)),
                'full_fitted': dict(zip(ALL_YEARS, AN_HUI_ELEC_EJCPCFGM_TRAIN_FIT + AN_HUI_ELEC_EJCPCFGM_TEST_PRE)),
                'mape_s': AN_HUI_ELEC_EJCPCFGM_MAPE_S,
                'mape_p': AN_HUI_ELEC_EJCPCFGM_MAPE_P
            }
        }
    }
    for method, preds in AN_HUI_ELEC_7M.items():
        all_results[key]['methods'][method] = {
            'train_fitted': {},
            'test_predicted': dict(zip(TEST_YEARS, preds)),
            'full_fitted': {},
            'mape_s': None,
            'mape_p': None
        }

    key = ('Anhui', 'Gen')
    all_results[key] = {
        'province': 'Anhui',
        'indicator': 'Gen',
        'train_actual': dict(zip(TRAIN_YEARS, AN_HUI_GEN_TRAIN)),
        'test_actual': dict(zip(TEST_YEARS, AN_HUI_GEN_TEST)),
        'methods': {
            'EJCPCFGM(σ,1)': {
                'train_fitted': dict(zip(TRAIN_YEARS, AN_HUI_GEN_EJCPCFGM_TRAIN_FIT)),
                'test_predicted': dict(zip(TEST_YEARS, AN_HUI_GEN_EJCPCFGM_TEST_PRE)),
                'full_fitted': dict(zip(ALL_YEARS, AN_HUI_GEN_EJCPCFGM_TRAIN_FIT + AN_HUI_GEN_EJCPCFGM_TEST_PRE)),
                'mape_s': AN_HUI_GEN_EJCPCFGM_MAPE_S,
                'mape_p': AN_HUI_GEN_EJCPCFGM_MAPE_P
            }
        }
    }
    for method, preds in AN_HUI_GEN_7M.items():
        all_results[key]['methods'][method] = {
            'train_fitted': {},
            'test_predicted': dict(zip(TEST_YEARS, preds)),
            'full_fitted': {},
            'mape_s': None,
            'mape_p': None
        }

    return all_results


def calculate_mapes(method_data, train_actual, test_actual):
    train_fitted = method_data.get('train_fitted', {})
    test_predicted = method_data.get('test_predicted', {})

    ape_s_list = []
    for year, actual in train_actual.items():
        fitted = train_fitted.get(year)
        if fitted is not None and actual != 0:
            ape = abs((actual - fitted) / actual) * 100
            ape_s_list.append((year, actual, fitted, ape))

    mape_s = np.mean([x[3] for x in ape_s_list]) if ape_s_list else None

    ape_p_list = []
    for year, actual in test_actual.items():
        predicted = test_predicted.get(year)
        if predicted is not None and actual != 0:
            ape = abs((actual - predicted) / actual) * 100
            ape_p_list.append((year, actual, predicted, ape))

    mape_p = np.mean([x[3] for x in ape_p_list]) if ape_p_list else None

    all_apes = [x[3] for x in ape_s_list] + [x[3] for x in ape_p_list]
    mape = np.mean(all_apes) if all_apes else None

    return mape_s, mape_p, mape, ape_s_list, ape_p_list


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


def plot_province_indicator(data, province_en, indicator_en):
    n_methods = 8
    n_cols = 4
    n_rows = 2

    fig = plt.figure(figsize=(22, 11))
    fig.patch.set_facecolor('white')

    for idx, method in enumerate(METHOD_NAMES):
        row = idx // n_cols
        col = idx % n_cols

        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        color = METHOD_COLORS[idx]
        cmap_name = GRADIENT_CMAPS[idx]

        method_data = data['methods'].get(method, {})
        full_fitted = method_data.get('full_fitted', {})
        test_mape = method_data.get('mape_p')

        train_actual = data['train_actual']
        test_actual = data['test_actual']

        train_years = sorted(train_actual.keys())
        test_years = sorted(test_actual.keys())
        all_years = train_years + test_years

        train_vals = [train_actual[y] for y in train_years]
        test_vals = [test_actual[y] for y in test_years]
        all_vals = train_vals + test_vals

        fitted_years = sorted(full_fitted.keys())
        fitted_vals = [full_fitted[y] for y in fitted_years]

        y_max = max(all_vals) * 1.12
        y_min = min(all_vals) * 0.88

        ax.set_facecolor('#FAFAFA')
        ax.set_xlim(2009.5, 2024.5)
        ax.set_ylim(y_min, y_max)
        add_radial_gradient(ax, cmap_name)

        ax.plot(all_years, all_vals, '-', color='#1a1a1a', linewidth=2.0,
                zorder=6, solid_capstyle='round', solid_joinstyle='round')

        for x, y in zip(all_years, all_vals):
            ax.scatter(x+0.08, y-0.08*(y_max-y_min)/15, c='gray', s=60, zorder=7,
                      alpha=0.35, edgecolors='none')
            ax.scatter(x, y, c='white', s=60, zorder=8, edgecolors='#1a1a1a', linewidth=1.5)
            ax.scatter(x, y, c='#1a1a1a', s=40, zorder=9, edgecolors='none')

        if fitted_years and len(fitted_vals) > 0:
            actual_for_fitted = [train_actual.get(y, test_actual.get(y, np.nan)) for y in fitted_years]

            ax.plot(fitted_years, fitted_vals, '-', color=color, linewidth=3.5,
                    alpha=0.95, zorder=5, solid_capstyle='round', solid_joinstyle='round',
                    path_effects=[pe.Stroke(linewidth=4.5, foreground='white'),
                                  pe.Normal()])

            for x, y in zip(fitted_years, fitted_vals):
                ax.scatter(x+0.08, y-0.08*(y_max-y_min)/15, c='gray', s=70, zorder=6,
                          alpha=0.3, edgecolors='none')
                ax.scatter(x, y, c='white', s=70, zorder=7, edgecolors=color, linewidth=2)
                ax.scatter(x, y, c=color, s=55, marker='D', zorder=8, edgecolors='none')

        ax.axvline(x=2021.5, color='#666666', linestyle='--', alpha=0.7,
                   linewidth=2, zorder=3)

        ax.text(2015.5, y_max * 0.96, 'Training', ha='center', fontsize=10,
                color='#555555', style='italic', fontweight='bold', zorder=15)
        ax.text(2023, y_max * 0.96, 'Testing', ha='center', fontsize=10,
                color='#555555', style='italic', fontweight='bold', zorder=15)

        mape_p = method_data.get('mape_p')
        display_title = METHOD_TITLES.get(method, method)
        mape_p_str = f'MAPE$_p$: {mape_p:.2f}%' if mape_p else 'MAPE$_p$: N/A'
        ax.set_title(f'{display_title}\n{mape_p_str}', fontsize=12, fontweight='bold',
                     color=color, pad=10, zorder=20)

        ax.set_xlabel('Year', fontsize=10, color='#333333', fontweight='bold', labelpad=5)
        ax.set_ylabel('Value', fontsize=10, color='#333333', fontweight='bold', labelpad=5)

        ax.tick_params(colors='#333333', labelsize=9)
        ax.set_xticks([2010, 2015, 2020, 2024])

        for spine in ax.spines.values():
            spine.set_color('#333333')
            spine.set_linewidth(2)
            spine.set_zorder(10)

        ax.grid(True, linestyle='-', alpha=0.15, color='#888888', zorder=1,
                which='major', axis='both')
        ax.set_axisbelow(True)

    plt.tight_layout(pad=1.5, h_pad=2, w_pad=1.5)

    filename = f'{province_en}_{indicator_en}.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f'Saved: {filepath}')


def find_best_model(all_results):
    print("\n" + "=" * 80)
    print("Best Model Analysis")
    print("=" * 80)

    best_results = []

    for key, data in all_results.items():
        province = data['province']
        indicator = data['indicator']

        best_method = None
        best_mape_p = float('inf')

        for method in METHOD_NAMES:
            if method in data['methods']:
                method_data = data['methods'][method]
                mape_p = method_data.get('mape_p')

                if mape_p is None:
                    test_actual = data['test_actual']
                    test_predicted = method_data.get('test_predicted', {})
                    ape_p_list = [abs((a - test_predicted.get(y, 0)) / a) * 100
                                  for y, a in test_actual.items() if a != 0 and y in test_predicted]
                    mape_p = np.mean(ape_p_list) if ape_p_list else None

                if mape_p is not None and mape_p < best_mape_p:
                    best_mape_p = mape_p
                    best_method = method

        if best_method:
            method_data = data['methods'][best_method]
            mape_s = method_data.get('mape_s')

            if mape_s is None:
                train_actual = data['train_actual']
                train_fitted = method_data.get('train_fitted', {})
                ape_s_list = [abs((a - train_fitted.get(y, 0)) / a) * 100
                              for y, a in train_actual.items() if a != 0 and y in train_fitted]
                mape_s = np.mean(ape_s_list) if ape_s_list else None

            best_results.append({
                'Province': province,
                'Indicator': indicator,
                'Best Method': best_method,
                'MAPE_s(%)': round(mape_s, 4) if mape_s else '',
                'MAPE_p(%)': round(best_mape_p, 4)
            })
            print(f"{province}-{indicator}: Best method is {best_method}, MAPE_s = {mape_s:.4f}%  |  MAPE_p = {best_mape_p:.4f}%" if mape_s else f"{province}-{indicator}: Best method is {best_method}, MAPE_p = {best_mape_p:.4f}%")

    print("\n" + "-" * 70)
    print("Best model count for each method:")
    print("-" * 70)

    method_best_count = {m: 0 for m in METHOD_NAMES}
    for result in best_results:
        method_best_count[result['Best Method']] += 1

    sorted_methods = sorted(method_best_count.items(), key=lambda x: -x[1])
    for method, count in sorted_methods:
        bar = '*' * count
        print(f"  {method:<18}: {count:>2} times {bar}")

    print("\n" + "=" * 80)
    print("Overall Ranking by Average MAPE_p:")
    print("=" * 80)

    method_avg_mape_p = {m: [] for m in METHOD_NAMES}
    method_avg_mape_s = {m: [] for m in METHOD_NAMES}

    for key, data in all_results.items():
        for method in METHOD_NAMES:
            if method in data['methods']:
                method_data = data['methods'][method]

                mape_p = method_data.get('mape_p')
                if mape_p is None:
                    test_actual = data['test_actual']
                    test_predicted = method_data.get('test_predicted', {})
                    ape_p_list = [abs((a - test_predicted.get(y, 0)) / a) * 100
                                  for y, a in test_actual.items() if a != 0 and y in test_predicted]
                    mape_p = np.mean(ape_p_list) if ape_p_list else None

                if mape_p is not None:
                    method_avg_mape_p[method].append(mape_p)

                mape_s = method_data.get('mape_s')
                if mape_s is None:
                    train_actual = data['train_actual']
                    train_fitted = method_data.get('train_fitted', {})
                    ape_s_list = [abs((a - train_fitted.get(y, 0)) / a) * 100
                                  for y, a in train_actual.items() if a != 0 and y in train_fitted]
                    mape_s = np.mean(ape_s_list) if ape_s_list else None

                if mape_s is not None:
                    method_avg_mape_s[method].append(mape_s)

    avg_mapes_p = {}
    avg_mapes_s = {}
    for method in METHOD_NAMES:
        if method_avg_mape_p[method]:
            avg_mapes_p[method] = np.mean(method_avg_mape_p[method])
        if method_avg_mape_s[method]:
            avg_mapes_s[method] = np.mean(method_avg_mape_s[method])

    sorted_avg = sorted(avg_mapes_p.items(), key=lambda x: x[1])

    print(f"\n{'Rank':<6} {'Method':<18} {'Avg MAPE_s(%)':<15} {'Avg MAPE_p(%)':<15} {'Visual'}")
    print("-" * 70)

    for rank, (method, avg_mape_p) in enumerate(sorted_avg, 1):
        avg_mape_s = avg_mapes_s.get(method, 0)
        blocks = '*' * (len(sorted_avg) - rank + 1)
        print(f"{rank:<6} {method:<18} {avg_mape_s:>10.4f}%   {avg_mape_p:>10.4f}%   {blocks}")

    overall_best = sorted_avg[0]
    avg_mape_s_best = avg_mapes_s.get(overall_best[0], 0)
    print(f"\nBest: {overall_best[0]}, Average MAPE_s = {avg_mape_s_best:.4f}%  |  Average MAPE_p = {overall_best[1]:.4f}%")

    return best_results, avg_mapes_p, avg_mapes_s


def print_summary_table(all_results):
    print("\n" + "=" * 200)
    print(" " * 70 + "8 Methods MAPE Comparison Summary (MAPE_s | MAPE_p)")
    print("=" * 200)

    header = f"{'Province-Indicator':<28}"
    for method in METHOD_NAMES:
        header += f"{method:>22}"
    print(header)
    print("-" * 200)

    for key, data in all_results.items():
        label = f"{data['province']}-{data['indicator']}"
        row = f"{label:<28}"

        for method in METHOD_NAMES:
            if method in data['methods']:
                method_data = data['methods'][method]
                mape_s = method_data.get('mape_s')
                mape_p = method_data.get('mape_p')

                if mape_s is None:
                    train_actual = data['train_actual']
                    train_fitted = method_data.get('train_fitted', {})
                    ape_s_list = [abs((a - train_fitted.get(y, 0)) / a) * 100
                                  for y, a in train_actual.items() if a != 0 and y in train_fitted]
                    mape_s = np.mean(ape_s_list) if ape_s_list else None

                if mape_p is None:
                    mape_p = method_data.get('mape_p')

                if mape_s is not None and mape_p is not None:
                    row += f"{mape_s:>9.2f}% | {mape_p:>7.2f}%"
                elif mape_p is not None:
                    row += f"{'N/A':>9}  | {mape_p:>7.2f}%"
                else:
                    row += f"{'N/A':>9}  | {'N/A':>7}"
            else:
                row += f"{'N/A':>22}"

        print(row)

    print("=" * 200)


def save_combined_csv(all_results):
    import pandas as pd
    rows = []

    for key, data in all_results.items():
        province = data['province']
        indicator = data['indicator']

        for method in METHOD_NAMES:
            if method not in data['methods']:
                continue

            method_data = data['methods'][method]
            mape_s = method_data.get('mape_s')
            mape_p = method_data.get('mape_p')

            if mape_s is None:
                train_actual = data['train_actual']
                train_fitted = method_data.get('train_fitted', {})
                ape_s_list = []
                for year, actual in train_actual.items():
                    fitted = train_fitted.get(year)
                    if fitted is not None and actual != 0:
                        ape = abs((actual - fitted) / actual) * 100
                        ape_s_list.append(ape)
                mape_s = np.mean(ape_s_list) if ape_s_list else None

            if mape_p is None:
                test_actual = data['test_actual']
                test_predicted = method_data.get('test_predicted', {})
                ape_p_list = []
                for year, actual in test_actual.items():
                    predicted = test_predicted.get(year)
                    if predicted is not None and actual != 0:
                        ape = abs((actual - predicted) / actual) * 100
                        ape_p_list.append(ape)
                mape_p = np.mean(ape_p_list) if ape_p_list else None

            row = {
                'Province': province,
                'Indicator': indicator,
                'Method': method,
                'MAPE_s(%)': round(mape_s, 4) if mape_s is not None else '',
                'MAPE_p(%)': round(mape_p, 4) if mape_p is not None else ''
            }

            for year in TRAIN_YEARS:
                actual = data['train_actual'].get(year, '')
                fitted = method_data.get('train_fitted', {}).get(year, '')
                if actual != '' and fitted != '' and actual != 0:
                    ape = abs((actual - fitted) / actual) * 100
                    row[f'{year} APE(%)'] = round(ape, 4)
                else:
                    row[f'{year} APE(%)'] = ''
                row[f'{year} Actual'] = round(actual, 2) if actual != '' else ''
                row[f'{year} Fitted'] = round(fitted, 2) if fitted != '' else ''

            for year in TEST_YEARS:
                actual = data['test_actual'].get(year, '')
                predicted = method_data.get('test_predicted', {}).get(year, '')
                if actual != '' and predicted != '' and actual != 0:
                    ape = abs((actual - predicted) / actual) * 100
                    row[f'{year} APE(%)'] = round(ape, 4)
                else:
                    row[f'{year} APE(%)'] = ''
                row[f'{year} Actual'] = round(actual, 2) if actual != '' else ''
                row[f'{year} Predicted'] = round(predicted, 2) if predicted != '' else ''

            rows.append(row)

    df = pd.DataFrame(rows)

    cols = ['Province', 'Indicator', 'Method', 'MAPE_s(%)', 'MAPE_p(%)']
    for year in TRAIN_YEARS:
        cols.extend([f'{year} APE(%)', f'{year} Actual', f'{year} Fitted'])
    for year in TEST_YEARS:
        cols.extend([f'{year} APE(%)', f'{year} Actual', f'{year} Predicted'])

    df = df[[c for c in cols if c in df.columns]]
    filepath = os.path.join(OUTPUT_DIR, 'Complete_Results.csv')
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Complete results saved to: {filepath}")
    return df


def save_detailed_mape_table(all_results):
    import pandas as pd
    rows = []

    for key, data in all_results.items():
        province = data['province']
        indicator = data['indicator']

        for method in METHOD_NAMES:
            if method not in data['methods']:
                continue

            method_data = data['methods'][method]
            mape_s = method_data.get('mape_s')
            mape_p = method_data.get('mape_p')

            if mape_s is None:
                train_actual = data['train_actual']
                train_fitted = method_data.get('train_fitted', {})
                ape_s_list = []
                for year, actual in train_actual.items():
                    fitted = train_fitted.get(year)
                    if fitted is not None and actual != 0:
                        ape = abs((actual - fitted) / actual) * 100
                        ape_s_list.append(ape)
                mape_s = np.mean(ape_s_list) if ape_s_list else None

            if mape_p is None:
                test_actual = data['test_actual']
                test_predicted = method_data.get('test_predicted', {})
                ape_p_list = []
                for year, actual in test_actual.items():
                    predicted = test_predicted.get(year)
                    if predicted is not None and actual != 0:
                        ape = abs((actual - predicted) / actual) * 100
                        ape_p_list.append(ape)
                mape_p = np.mean(ape_p_list) if ape_p_list else None

            if mape_s is not None and mape_p is not None:
                mape = (mape_s + mape_p) / 2
            else:
                mape = None

            row = {
                'Province': province,
                'Indicator': indicator,
                'Method': method,
                'MAPE_s(%)': round(mape_s, 4) if mape_s is not None else '',
                'MAPE_p(%)': round(mape_p, 4) if mape_p is not None else '',
                'MAPE(%)': round(mape, 4) if mape is not None else ''
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    cols = ['Province', 'Indicator', 'Method', 'MAPE_s(%)', 'MAPE_p(%)', 'MAPE(%)']
    df = df[[c for c in cols if c in df.columns]]

    filepath = os.path.join(OUTPUT_DIR, 'Detailed_MAPE_Table.csv')
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"Detailed MAPE table saved to: {filepath}")
    return df


def main():
    print("=" * 80)
    print("8 Methods Comparison Analysis (with EJCPCFGM(sigma,1))")
    print("Only includes Jiangsu and Anhui provinces")
    print("=" * 80)

    print(f"\nOutput directory: {OUTPUT_DIR}/")

    print("\nBuilding results from embedded data...")
    all_results = build_all_results()

    print(f"\nTotal {len(all_results)} datasets:")
    for key in all_results:
        data = all_results[key]
        print(f"  {data['province']}-{data['indicator']}: {len(data['methods'])} methods")

    print("\nSaving results...")
    save_combined_csv(all_results)
    save_detailed_mape_table(all_results)

    print_summary_table(all_results)

    best_results, avg_mapes_p, avg_mapes_s = find_best_model(all_results)

    print("\nGenerating comparison plots...")

    province_map = {
        'Jiangsu': 'Jiangsu',
        'Anhui': 'Anhui'
    }

    indicator_map = {
        'Elec': 'Electricity_Consumption',
        'Gen': 'Electricity_Generation'
    }

    for key, data in all_results.items():
        province = data['province']
        indicator = data['indicator']

        province_en = province_map.get(province, province)
        indicator_en = indicator_map.get(indicator, indicator)

        print(f"Plotting {province_en} - {indicator_en}...")
        plot_province_indicator(data, province_en, indicator_en)

    import pandas as pd
    best_df = pd.DataFrame(best_results)
    best_filepath = os.path.join(OUTPUT_DIR, 'Best_Model_Summary.csv')
    best_df.to_csv(best_filepath, index=False, encoding='utf-8-sig')
    print(f"\nBest model summary saved to: {best_filepath}")

    print("\n" + "=" * 80)
    print(f"Analysis complete! All results saved to {OUTPUT_DIR}/ folder")
    print(f"Total 4 figures generated (2 provinces x 2 indicators)")
    print("=" * 80)


if __name__ == "__main__":
    main()
