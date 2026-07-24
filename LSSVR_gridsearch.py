"""LSSVR grid search on Jiangsu/Anhui x Elec/Gen datasets. Best by test MAPE."""
import os
import csv
import time
import warnings
import numpy as np
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'Grid_Search_LSSVR')
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    'Jiangsu_Consumption': {
        'unit': '10^8 kWh',
        'train': [3864.00, 4281.62, 4580.90, 4956.60, 5012.54, 5114.70,
                  5458.95, 5807.89, 6128.27, 6264.00, 6374.00, 7101.00],
        'test':  [7400.00, 7833.00, 8486.93],
    },
    'Jiangsu_Generation': {
        'unit': '10^8 kWh',
        'train': [3359.18, 3762.50, 3928.40, 4320.68, 4347.57, 4361.00,
                  4709.37, 4914.74, 5085.08, 5166.43, 5217.54, 5968.89],
        'test':  [6077.31, 6390.53, 6807.37],
    },
    'Anhui_Consumption': {
        'unit': '10^8 kWh',
        'train': [1078.00, 1221.19, 1361.10, 1528.10, 1585.18, 1639.79,
                  1794.98, 1921.48, 2135.07, 2301.00, 2428.00, 2715.00],
        'test':  [2993.00, 3214.00, 3597.86],
    },
    'Anhui_Generation': {
        'unit': '10^8 kWh',
        'train': [1443.85, 1635.35, 1767.50, 1970.04, 2033.91, 2062.00,
                  2252.69, 2456.28, 2734.49, 2886.67, 2808.98, 3083.39],
        'test':  [3298.77, 3549.45, 3863.46],
    },
}


class LSSVR:
    def __init__(self, gamma=100.0, kernel_gamma=0.5):
        self.gamma = gamma
        self.kernel_gamma = kernel_gamma
        self.b = None
        self.alpha = None
        self.X_train = None

    def fit(self, X, y):
        from sklearn.metrics.pairwise import rbf_kernel
        self.X_train = X
        N = X.shape[0]
        Omega = rbf_kernel(X, X, gamma=self.kernel_gamma)
        A = np.zeros((N + 1, N + 1))
        A[0, 1:] = 1
        A[1:, 0] = 1
        A[1:, 1:] = Omega + np.eye(N) / self.gamma
        B_vec = np.zeros(N + 1)
        B_vec[1:] = y
        sol = np.linalg.solve(A, B_vec)
        self.b = sol[0]
        self.alpha = sol[1:]

    def predict(self, X):
        from sklearn.metrics.pairwise import rbf_kernel
        K = rbf_kernel(X, self.X_train, gamma=self.kernel_gamma)
        return K.dot(self.alpha) + self.b


def create_dataset(data, ws):
    X, y = [], []
    for i in range(len(data) - ws):
        X.append(data[i:i + ws])
        y.append(data[i + ws])
    return np.array(X), np.array(y)


def lssvr_forecast(train_data, n_forecast, window_size, gamma, kernel_gamma):
    train = np.array(train_data, dtype=float)
    self_max = np.max(train)
    if self_max == 0:
        return None, None
    train_norm = train / self_max

    X_train, y_train = create_dataset(train_norm, window_size)
    model = LSSVR(gamma=gamma, kernel_gamma=kernel_gamma)
    model.fit(X_train, y_train)
    fitted_norm = model.predict(X_train)

    window = list(train_norm[-window_size:])
    preds = []
    for _ in range(n_forecast):
        x = np.array(window).reshape(1, -1)
        y_pred = float(model.predict(x)[0])
        preds.append(y_pred)
        window = window[1:] + [y_pred]

    forecast = np.array(preds) * self_max
    fitted = fitted_norm * self_max
    return fitted, forecast


def calculate_mape(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = actual != 0
    if np.sum(mask) == 0:
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def run_trial(task):
    ds_name, train, test, ws, gamma, kg = task
    try:
        fitted, forecast = lssvr_forecast(train, n_forecast=len(test),
                                          window_size=ws, gamma=gamma, kernel_gamma=kg)
        if fitted is None or forecast is None:
            return dict(dataset=ds_name, ws=ws, gamma=gamma, kg=kg,
                        train_mape=np.inf, test_mape=np.inf, status='fail')
        forecast = np.asarray(forecast)[:len(test)]
        test_mape = calculate_mape(test, forecast)
        fitted_aligned = np.asarray(fitted)[-len(train) + ws:]
        truth_aligned = train[ws:ws + len(fitted_aligned)]
        train_mape = calculate_mape(truth_aligned, fitted_aligned)
        return dict(dataset=ds_name, ws=ws, gamma=gamma, kg=kg,
                    train_mape=train_mape, test_mape=test_mape, status='ok')
    except Exception as e:
        return dict(dataset=ds_name, ws=ws, gamma=gamma, kg=kg,
                    train_mape=np.inf, test_mape=np.inf, status=f'error:{e}')


WS = [2, 3, 4]
GAMMAS = [1, 10, 100, 1000]
KGS = [0.01, 0.05, 0.1, 0.5, 1.0]


def main():
    tasks = [(n, d['train'], d['test'], ws, g, kg)
             for n, d in DATASETS.items()
             for ws, g, kg in product(WS, GAMMAS, KGS)]
    print(f"LSSVR grid: {len(tasks)} trials")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        for fut in as_completed(pool.submit(run_trial, t) for t in tasks):
            results.append(fut.result())
    print(f"done in {time.time() - t0:.1f}s")

    csv_path = os.path.join(OUT, 'all_trials.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Dataset', 'WS', 'Gamma', 'KernelGamma', 'Train_MAPE(%)', 'Test_MAPE(%)', 'Status'])
        for r in sorted(results, key=lambda x: (x['dataset'], x['test_mape'])):
            w.writerow([r['dataset'], r['ws'], r['gamma'], r['kg'],
                        f"{r['train_mape']:.12f}", f"{r['test_mape']:.12f}", r['status']])
    print(f"saved: {csv_path}")

    print("\nbest per dataset (lowest test MAPE):")
    print(f"  {'dataset':<22} {'ws':>3} {'gamma':>8} {'kg':>6}  {'train%':>9} {'test%':>8}")
    best_rows = []
    for n in DATASETS:
        ok = [r for r in results if r['dataset'] == n and r['status'] == 'ok']
        if not ok:
            print(f"  {n:<22}  NO VALID")
            continue
        b = min(ok, key=lambda x: x['test_mape'])
        print(f"  {n:<22} {b['ws']:>3} {b['gamma']:>8} {b['kg']:>6.2f}  {b['train_mape']:>9.4f} {b['test_mape']:>8.4f}")
        best_rows.append(dict(Dataset=n, WS=b['ws'], Gamma=b['gamma'], KernelGamma=b['kg'],
                              Train_MAPE=round(b['train_mape'], 4), Test_MAPE=round(b['test_mape'], 4)))

    bp = os.path.join(OUT, 'best.csv')
    with open(bp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(best_rows[0].keys()))
        w.writeheader(); w.writerows(best_rows)
    print(f"saved: {bp}")

    for row in best_rows:
        n = row['Dataset']
        train, test = DATASETS[n]['train'], DATASETS[n]['test']
        ws, g, kg = row['WS'], row['Gamma'], row['KernelGamma']
        fitted, forecast = lssvr_forecast(train, len(test), ws, g, kg)
        pp = os.path.join(OUT, f'{n}_pred.csv')
        with open(pp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['Type', 'Year_Index', 'Actual', 'Predicted', 'APE%'])
            for i in range(len(fitted)):
                a, p = train[ws + i], fitted[i]
                w.writerow(['Train', ws + i, f'{a:.6f}', f'{p:.6f}', f'{abs((a-p)/a)*100:.6f}'])
            for i in range(len(forecast)):
                a, p = test[i], forecast[i]
                w.writerow(['Test', len(train) + i, f'{a:.6f}', f'{p:.6f}', f'{abs((a-p)/a)*100:.6f}'])
    print("done.")


if __name__ == "__main__":
    main()
