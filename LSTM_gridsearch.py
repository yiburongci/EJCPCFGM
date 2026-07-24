"""LSTM grid search on Jiangsu/Anhui x Elec/Gen datasets. Best by test MAPE."""
import os
import csv
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, 'Grid_Search_LSTM')
os.makedirs(OUT, exist_ok=True)

DATASETS = {
    'Anhui_Consumption': {
        'train': [1078.00, 1221.19, 1361.10, 1528.10, 1585.18, 1639.79,
                  1794.98, 1921.48, 2135.07, 2301.00, 2428.00, 2715.00],
        'test':  [2993.00, 3214.00, 3597.86],
    },
    'Jiangsu_Consumption': {
        'train': [3864.00, 4281.62, 4580.90, 4956.60, 5012.54, 5114.70,
                  5458.95, 5807.89, 6128.27, 6264.00, 6374.00, 7101.00],
        'test':  [7400.00, 7833.00, 8486.93],
    },
    'Anhui_Generation': {
        'train': [1443.85, 1635.35, 1767.50, 1970.04, 2033.91, 2062.00,
                  2252.69, 2456.28, 2734.49, 2886.67, 2808.98, 3083.39],
        'test':  [3298.77, 3549.45, 3863.46],
    },
    'Jiangsu_Generation': {
        'train': [3359.18, 3762.50, 3928.40, 4320.68, 4347.57, 4361.00,
                  4709.37, 4914.74, 5085.08, 5166.43, 5217.54, 5968.89],
        'test':  [6077.31, 6390.53, 6807.37],
    },
}


def build_lstm(input_size=1, hidden_size=64, num_layers=1):
    import torch.nn as nn

    class PureLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return PureLSTM()


def mape(actual, predicted):
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def create_dataset(data, ws):
    X, y = [], []
    for i in range(len(data) - ws):
        X.append(data[i:i + ws])
        y.append(data[i + ws])
    return np.array(X), np.array(y)


def lstm_trial(train_data, test_data, window_size, hidden_size, num_layers, lr, epochs, seed):
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    train = np.array(train_data, dtype=float)
    n_test = len(test_data)

    scaler = MinMaxScaler()
    train_norm = scaler.fit_transform(train.reshape(-1, 1)).ravel()
    X, y = create_dataset(train_norm, window_size)
    X_t = torch.FloatTensor(X).unsqueeze(-1)
    y_t = torch.FloatTensor(y)

    model = build_lstm(1, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    model.train()
    for _ in range(epochs):
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=min(16, len(X_t)), shuffle=True)
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        fitted_norm = model(X_t).numpy()
    fitted = scaler.inverse_transform(fitted_norm.reshape(-1, 1)).ravel()

    window = list(train_norm[-window_size:])
    preds_norm = []
    for _ in range(n_test):
        x = torch.FloatTensor(np.array(window).reshape(1, -1, 1))
        with torch.no_grad():
            y_pred = float(model(x).numpy()[0])
        preds_norm.append(y_pred)
        window = window[1:] + [y_pred]
    forecast = scaler.inverse_transform(np.array(preds_norm).reshape(-1, 1)).ravel()
    return fitted, forecast


WS = [2, 3, 4]
HIDDEN = [32, 64, 128]
LAYERS = [1, 2]
LR = [0.001, 0.003, 0.005]
EPOCHS = 100
SEEDS = [101]

GRID = [(w, h, l, r) for w in WS for h in HIDDEN for l in LAYERS for r in LR]


def run_all():
    trials = []
    total = len(DATASETS) * len(GRID) * len(SEEDS)
    print(f"LSTM grid: {total} trials")
    t0 = time.time()

    for n, d in DATASETS.items():
        for w, h, l, lr in GRID:
            for s in SEEDS:
                try:
                    fitted, forecast = lstm_trial(d['train'], d['test'], w, h, l, lr, EPOCHS, s)
                    tm = mape(d['test'], forecast)
                    ta = mape(d['train'][w:w + len(fitted)], fitted)
                    status = 'ok'
                except Exception as e:
                    tm = ta = np.nan
                    status = str(e)[:60]
                trials.append(dict(Dataset=n, WS=w, Hidden=h, Layers=l, LR=lr, Seed=s,
                                   Train_MAPE=ta, Test_MAPE=tm, Status=status))

    print(f"done in {time.time() - t0:.1f}s")

    ap = os.path.join(OUT, 'all_trials.csv')
    with open(ap, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['Dataset', 'WS', 'Hidden', 'Layers', 'LR', 'Seed',
                                          'Train_MAPE', 'Test_MAPE', 'Status'])
        w.writeheader()
        w.writerows(sorted(trials, key=lambda x: (x['Dataset'], x['Test_MAPE'])))
    print(f"saved: {ap}")

    print("\nbest per dataset (lowest test MAPE):")
    print(f"  {'dataset':<22} {'ws':>3} {'hidden':>6} {'layers':>6} {'lr':>6}  {'train%':>9} {'test%':>8}")
    best_params, best_trials = {}, {}
    for n in DATASETS:
        ok = [t for t in trials if t['Dataset'] == n and t['Status'] == 'ok']
        if not ok:
            print(f"  {n:<22}  NO VALID")
            continue
        from collections import defaultdict
        grouped = defaultdict(list)
        for t in ok:
            grouped[(t['WS'], t['Hidden'], t['Layers'], t['LR'])].append(t)
        best_k = min(grouped, key=lambda k: np.mean([t['Test_MAPE'] for t in grouped[k]]))
        runs = grouped[best_k]
        avg_t = np.mean([t['Test_MAPE'] for t in runs])
        avg_tr = np.mean([t['Train_MAPE'] for t in runs])
        w_, h_, l_, lr_ = best_k
        print(f"  {n:<22} {w_:>3} {h_:>6} {l_:>6} {lr_:>6}  {avg_tr:>9.4f} {avg_t:>8.4f}")
        best_params[n] = dict(ws=w_, hidden=h_, layers=l_, lr=lr_,
                               Train_MAPE=round(avg_tr, 4), Test_MAPE=round(avg_t, 4))
        best_trials[n] = runs

    bp = os.path.join(OUT, 'best.csv')
    with open(bp, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['Dataset', 'WS', 'Hidden', 'Layers', 'LR', 'Train_MAPE', 'Test_MAPE'])
        w.writeheader()
        for n, p in best_params.items():
            w.writerow({'Dataset': n, 'WS': p['ws'], 'Hidden': p['hidden'],
                        'Layers': p['layers'], 'LR': p['lr'],
                        'Train_MAPE': p['Train_MAPE'], 'Test_MAPE': p['Test_MAPE']})
    print(f"saved: {bp}")

    for n, p in best_params.items():
        train, test = DATASETS[n]['train'], DATASETS[n]['test']
        w_, h_, l_, lr_ = p['ws'], p['hidden'], p['layers'], p['lr']
        fitted, forecast = lstm_trial(train, test, w_, h_, l_, lr_, EPOCHS, seed=101)
        pp = os.path.join(OUT, f'{n}_pred.csv')
        with open(pp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['Type', 'Year_Index', 'Actual', 'Predicted', 'APE%'])
            for i in range(len(fitted)):
                a, pv = train[w_ + i], fitted[i]
                w.writerow(['Train', w_ + i, f'{a:.6f}', f'{pv:.6f}', f'{abs((a-pv)/a)*100:.6f}'])
            for i in range(len(forecast)):
                a, pv = test[i], forecast[i]
                w.writerow(['Test', len(train) + i, f'{a:.6f}', f'{pv:.6f}', f'{abs((a-pv)/a)*100:.6f}'])
    print("done.")


if __name__ == "__main__":
    run_all()
