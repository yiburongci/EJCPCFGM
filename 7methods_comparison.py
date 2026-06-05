"""
Seven Prediction Methods Comparison
Methods: ARIMA, GM(1,1), FGM, JFGM, NGBM, LSSVR, LSTM
Data: 5 provinces (2010-2024)
Train: 2010-2021, Test: 2022-2024
"""

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from matplotlib import rcParams
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

warnings.filterwarnings('ignore')
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


def get_provincial_data():
    """Electricity consumption and generation data (2010-2024)"""
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
    """Get all datasets"""
    elec_data, gen_data = get_provincial_data()
    datasets = []
    for name in elec_data.keys():
        datasets.append((name, 'Electricity', '10^8 kWh', elec_data[name]))
        datasets.append((name, 'Generation', '10^8 kWh', gen_data[name]))
    return datasets


def arima_predict(train_data, n_forecast=3, order=(1, 1, 1)):
    """ARIMA multi-step prediction"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()
        fitted = model_fit.fittedvalues
        predictions = model_fit.forecast(steps=n_forecast)
        fitted_values = list(fitted)
        if len(fitted_values) < len(train_data):
            fitted_values = [train_data[0]] + fitted_values
        return {
            'fitted': np.array(fitted_values[:len(train_data)]),
            'forecast': np.array(predictions),
            'order': order,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


def auto_arima(train_data):
    """Auto select best ARIMA order"""
    from statsmodels.tsa.arima.model import ARIMA
    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in range(0, 4):
        for d in range(0, 2):
            for q in range(0, 4):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)
                except:
                    continue
    return best_order


class GM11:
    """GM(1,1) Grey Prediction Model"""

    def __init__(self):
        self.a = None
        self.b = None
        self.x0_1 = None

    def fit(self, x0):
        """Train GM(1,1) model"""
        self.x0_1 = x0[0]
        n = len(x0)
        x1 = np.cumsum(x0)
        z1 = 0.5 * x1[1:] + 0.5 * x1[:-1]
        B = np.vstack((-z1, np.ones(n - 1))).T
        Y = x0[1:].reshape((n - 1, 1))
        u = np.linalg.inv(B.T @ B) @ B.T @ Y
        self.a = u[0][0]
        self.b = u[1][0]

    def predict(self, steps):
        """Multi-step prediction"""
        if self.a is None:
            raise ValueError("Call fit() first")
        total_steps = steps
        k = np.arange(total_steps)
        x1_hat = (self.x0_1 - self.b / self.a) * np.exp(-self.a * k) + self.b / self.a
        x0_hat = np.zeros(total_steps)
        x0_hat[0] = x1_hat[0]
        for i in range(1, total_steps):
            x0_hat[i] = x1_hat[i] - x1_hat[i - 1]
        return x0_hat


def gm11_predict(train_data, n_forecast=3):
    """GM(1,1) multi-step prediction"""
    try:
        gm = GM11()
        gm.fit(train_data)
        total_steps = len(train_data) + n_forecast
        predictions = gm.predict(total_steps)
        return {
            'fitted': predictions[:len(train_data)],
            'forecast': predictions[len(train_data):],
            'a': gm.a,
            'b': gm.b,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


def get_combinations(r, n):
    """Compute fractional order combination coefficients"""
    c = np.zeros(n)
    c[0] = 1.0
    for m in range(1, n):
        c[m] = c[m - 1] * (r + m - 1) / m
    return c


def fractional_AGO(x, r):
    """Fractional accumulation operation"""
    n = len(x)
    c = get_combinations(r, n)
    xr = np.zeros(n)
    for k in range(n):
        xr[k] = np.sum(c[k::-1] * x[:k + 1])
    return xr


def fgm_predict(train_data, n_forecast=3, r=None):
    """FGM Fractional Grey Model"""
    try:
        from scipy.optimize import minimize_scalar

        def objective(r_opt):
            if r_opt <= 0.01 or r_opt > 2.0:
                return 1e9
            n = len(train_data)
            xr = fractional_AGO(train_data, r_opt)
            z = 0.5 * xr[1:] + 0.5 * xr[:-1]
            Y1 = xr[1:] - xr[:-1]
            B1 = np.vstack((-z, np.ones(n - 1))).T
            try:
                u1 = np.linalg.inv(B1.T @ B1) @ B1.T @ Y1
                a, b = u1[0], u1[1]
            except:
                return 1e9
            k_arr = np.arange(1, n)
            exp_ak = np.exp(-a * k_arr)
            B2 = np.vstack((exp_ak, np.ones(n - 1))).T
            Y2 = xr[1:]
            try:
                u2 = np.linalg.inv(B2.T @ B2) @ B2.T @ Y2
                c_param, d_param = u2[0], u2[1]
            except:
                return 1e9
            xr_hat = np.zeros(n)
            xr_hat[0] = train_data[0]
            xr_hat[1:] = c_param * np.exp(-a * k_arr) + d_param
            x0_hat = fractional_AGO(xr_hat, -r_opt)
            mape = np.mean(np.abs((train_data - x0_hat) / train_data))
            return mape

        if r is None:
            opt_result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
            r = opt_result.x

        n = len(train_data)
        xr = fractional_AGO(train_data, r)
        z = 0.5 * xr[1:] + 0.5 * xr[:-1]
        Y1 = xr[1:] - xr[:-1]
        B1 = np.vstack((-z, np.ones(n - 1))).T
        u1 = np.linalg.inv(B1.T @ B1) @ B1.T @ Y1
        a, b = u1[0], u1[1]
        k_arr = np.arange(1, n)
        exp_ak = np.exp(-a * k_arr)
        B2 = np.vstack((exp_ak, np.ones(n - 1))).T
        Y2 = xr[1:]
        u2 = np.linalg.inv(B2.T @ B2) @ B2.T @ Y2
        c_param, d_param = u2[0], u2[1]
        total_len = n + n_forecast
        xr_hat = np.zeros(total_len)
        xr_hat[0] = train_data[0]
        k_pred = np.arange(1, total_len)
        xr_hat[1:] = c_param * np.exp(-a * k_pred) + d_param
        x0_hat = fractional_AGO(xr_hat, -r)
        return {
            'fitted': x0_hat[:n],
            'forecast': x0_hat[n:],
            'r': r,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


def jfgm_predict(train_data, n_forecast=3, r=None, S=None, ad=1.0, t0_idx=None):
    """JFGM Jump Fractional Grey Model"""
    try:
        from scipy.optimize import minimize

        n = len(train_data)
        if t0_idx is None:
            t0_idx = n - 2

        def objective(params):
            r_opt, S_opt = params
            if r_opt <= 0.01 or r_opt > 2.0 or S_opt <= 0.1 or S_opt > 10:
                return 1e9
            xr = fractional_AGO(train_data, r_opt)
            z = 0.5 * xr[1:] + 0.5 * xr[:-1]
            Y = xr[1:] - xr[:-1]
            B = np.zeros((n - 1, 3))
            B[:, 0] = -z
            for i in range(n - 1):
                k = i + 1
                if k >= t0_idx:
                    B[i, 1] = S_opt
                else:
                    B[i, 1] = 0.0
            B[:, 2] = 1.0
            try:
                u = np.linalg.inv(B.T @ B) @ B.T @ Y
                a, c, b = u[0], u[1], u[2]
            except:
                return 1e9
            xr_hat = np.zeros(n)
            xr_hat[0] = train_data[0]
            for i in range(1, n):
                k_paper = i + 1
                t0_paper = t0_idx + 1
                if k_paper < t0_paper:
                    xr_hat[i] = (train_data[0] - b / a) * np.exp(-a * (k_paper - 1)) + b / a
                else:
                    xr_hat[i] = (train_data[0] - b / a - c / a) * np.exp(-a * (k_paper - 1)) + \
                                (c * (ad ** (k_paper - t0_paper))) / a + b / a
            x0_hat = fractional_AGO(xr_hat, -r_opt)
            mape = np.mean(np.abs((train_data - x0_hat) / train_data))
            return mape

        if r is None or S is None:
            opt_result = minimize(objective, [0.5, 3.0], method='Nelder-Mead',
                                  options={'xatol': 1e-8, 'fatol': 1e-6})
            r = opt_result.x[0]
            S = opt_result.x[1]

        xr = fractional_AGO(train_data, r)
        z = 0.5 * xr[1:] + 0.5 * xr[:-1]
        Y = xr[1:] - xr[:-1]
        B = np.zeros((n - 1, 3))
        B[:, 0] = -z
        for i in range(n - 1):
            k = i + 1
            if k >= t0_idx:
                B[i, 1] = S
            else:
                B[i, 1] = 0.0
        B[:, 2] = 1.0
        u = np.linalg.inv(B.T @ B) @ B.T @ Y
        a, c, b = u[0], u[1], u[2]
        total_len = n + n_forecast
        xr_hat = np.zeros(total_len)
        xr_hat[0] = train_data[0]
        for i in range(1, total_len):
            k_paper = i + 1
            t0_paper = t0_idx + 1
            if k_paper < t0_paper:
                xr_hat[i] = (train_data[0] - b / a) * np.exp(-a * (k_paper - 1)) + b / a
            else:
                xr_hat[i] = (train_data[0] - b / a - c / a) * np.exp(-a * (k_paper - 1)) + \
                            (c * (ad ** (k_paper - t0_paper))) / a + b / a
        x0_hat = fractional_AGO(xr_hat, -r)
        return {
            'fitted': x0_hat[:n],
            'forecast': x0_hat[n:],
            'r': r,
            'S': S,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


def ngbm_predict(train_data, n_forecast=3, n_power=None):
    """NGBM Nonlinear Grey Bernoulli Model"""
    try:
        from scipy.optimize import minimize_scalar

        def objective(n_opt):
            if np.isclose(n_opt, 1.0) or n_opt < -2 or n_opt > 2:
                return 1e9
            m = len(train_data)
            x1 = np.cumsum(train_data)
            z1 = 0.5 * x1[1:] + 0.5 * x1[:-1]
            B = np.vstack((-z1, z1 ** n_opt)).T
            Y_N = train_data[1:]
            try:
                params = np.linalg.inv(B.T @ B) @ B.T @ Y_N
                a, b = params[0], params[1]
            except:
                return 1e9
            x1_hat = np.zeros(m)
            x1_hat[0] = train_data[0]
            for k in range(1, m):
                base = (train_data[0] ** (1 - n_opt) - b / a) * np.exp(-a * (1 - n_opt) * k) + b / a
                if base < 0:
                    return 1e9
                x1_hat[k] = base ** (1 / (1 - n_opt))
            x0_hat = np.zeros(m)
            x0_hat[0] = x1_hat[0]
            for k in range(1, m):
                x0_hat[k] = x1_hat[k] - x1_hat[k - 1]
            arpe = np.mean(np.abs((train_data[1:] - x0_hat[1:]) / train_data[1:])) * 100
            return arpe

        if n_power is None:
            opt_result = minimize_scalar(objective, bounds=(-2.0, 2.0), method='bounded')
            n_power = opt_result.x

        m = len(train_data)
        x1 = np.cumsum(train_data)
        z1 = 0.5 * x1[1:] + 0.5 * x1[:-1]
        B = np.vstack((-z1, z1 ** n_power)).T
        Y_N = train_data[1:]
        params = np.linalg.inv(B.T @ B) @ B.T @ Y_N
        a, b = params[0], params[1]
        total_len = m + n_forecast
        x1_hat = np.zeros(total_len)
        x1_hat[0] = train_data[0]
        for k in range(1, total_len):
            base = (train_data[0] ** (1 - n_power) - b / a) * np.exp(-a * (1 - n_power) * k) + b / a
            if base < 0:
                return {'fitted': None, 'forecast': None, 'success': False, 'error': 'overflow'}
            x1_hat[k] = base ** (1 / (1 - n_power))
        x0_hat = np.zeros(total_len)
        x0_hat[0] = x1_hat[0]
        for k in range(1, total_len):
            x0_hat[k] = x1_hat[k] - x1_hat[k - 1]
        return {
            'fitted': x0_hat[:m],
            'forecast': x0_hat[m:],
            'n': n_power,
            'a': a,
            'b': b,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


class LSSVR:
    """Least Squares Support Vector Regression"""

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
        solution = np.linalg.solve(A, B_vec)
        self.b = solution[0]
        self.alpha = solution[1:]

    def predict(self, X):
        from sklearn.metrics.pairwise import rbf_kernel
        K = rbf_kernel(X, self.X_train, gamma=self.kernel_gamma)
        return K.dot(self.alpha) + self.b


def lssvr_predict(train_data, n_forecast=3, window_size=5):
    """LSSVR multi-step recursive prediction"""
    try:
        def create_dataset(data, ws):
            X, y = [], []
            for i in range(len(data) - ws):
                X.append(data[i:i + ws])
                y.append(data[i + ws])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data, window_size)
        self_max = np.max(train_data)
        train_norm = train_data / self_max
        X_train_norm, y_train_norm = create_dataset(train_norm, window_size)
        model = LSSVR(gamma=100.0, kernel_gamma=0.5)
        model.fit(X_train_norm, y_train_norm)

        current_window = list(train_norm[-window_size:])
        predictions = []
        for _ in range(n_forecast):
            X_input = np.array(current_window).reshape(1, -window_size)
            y_pred = model.predict(X_input)[0]
            predictions.append(y_pred)
            current_window = current_window[1:] + [y_pred]

        predictions = np.array(predictions) * self_max
        fitted_norm = model.predict(X_train_norm) * self_max

        return {
            'fitted': fitted_norm,
            'forecast': predictions,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


def lstm_predict(train_data, n_forecast=3, window_size=5, hidden_size=32, num_layers=1, epochs=100):
    """LSTM multi-step autoregressive prediction"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.preprocessing import MinMaxScaler

        torch.manual_seed(42)
        np.random.seed(42)

        def create_dataset(data, ws):
            X, y = [], []
            for i in range(len(data) - ws):
                X.append(data[i:i + ws])
                y.append(data[i + ws])
            return np.array(X), np.array(y)

        scaler = MinMaxScaler()
        train_norm = scaler.fit_transform(train_data.reshape(-1, 1)).ravel()
        X_train, y_train = create_dataset(train_norm, window_size)
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train_tensor = torch.FloatTensor(y_train)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=min(16, len(X_train)), shuffle=True)

        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=1):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return out.squeeze(-1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_train_tensor = X_train_tensor.to(device)
            fitted_norm = model(X_train_tensor).cpu().numpy()

        fitted = scaler.inverse_transform(fitted_norm.reshape(-1, 1)).ravel()

        predictions = []
        current_window = list(train_norm[-window_size:])
        for _ in range(n_forecast):
            model.eval()
            with torch.no_grad():
                X_input = torch.FloatTensor(np.array(current_window).reshape(1, -1, 1)).to(device)
                y_pred = model(X_input).cpu().numpy()[0]
            predictions.append(y_pred)
            current_window = current_window[1:] + [y_pred]

        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()

        return {
            'fitted': fitted,
            'forecast': predictions,
            'success': True
        }
    except Exception as e:
        return {'fitted': None, 'forecast': None, 'success': False, 'error': str(e)}


def calculate_mape(actual, predicted):
    """Calculate MAPE"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    mask = actual != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def run_all_methods(name, indicator, unit, data, train_len=12, n_forecast=3):
    """Run all prediction methods for a single dataset"""

    train_data = np.array(data[:train_len])
    test_data = np.array(data[train_len:])

    results = {
        'name': name,
        'indicator': indicator,
        'unit': unit,
        'train_data': train_data,
        'test_data': test_data,
        'years': list(range(2010, 2025)),
        'train_years': list(range(2010, 2022)),
        'test_years': list(range(2022, 2025)),
    }

    methods = {}

    print(f"  Running ARIMA...")
    arima_result = arima_predict(train_data, n_forecast=n_forecast, order=(1, 1, 1))
    if arima_result['success']:
        mape_s = calculate_mape(train_data[1:], arima_result['fitted'][1:])
        mape_p = calculate_mape(test_data, arima_result['forecast'])
        methods['ARIMA'] = {
            'fitted': arima_result['fitted'],
            'forecast': arima_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p
        }

    print(f"  Running GM(1,1)...")
    gm_result = gm11_predict(train_data, n_forecast=n_forecast)
    if gm_result['success']:
        mape_s = calculate_mape(train_data, gm_result['fitted'])
        mape_p = calculate_mape(test_data, gm_result['forecast'])
        methods['GM(1,1)'] = {
            'fitted': gm_result['fitted'],
            'forecast': gm_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p
        }

    print(f"  Running FGM...")
    fgm_result = fgm_predict(train_data, n_forecast=n_forecast)
    if fgm_result['success']:
        mape_s = calculate_mape(train_data, fgm_result['fitted'])
        mape_p = calculate_mape(test_data, fgm_result['forecast'])
        methods['FGM'] = {
            'fitted': fgm_result['fitted'],
            'forecast': fgm_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p,
            'r': fgm_result['r']
        }

    print(f"  Running JFGM...")
    jfgm_result = jfgm_predict(train_data, n_forecast=n_forecast)
    if jfgm_result['success']:
        mape_s = calculate_mape(train_data, jfgm_result['fitted'])
        mape_p = calculate_mape(test_data, jfgm_result['forecast'])
        methods['JFGM'] = {
            'fitted': jfgm_result['fitted'],
            'forecast': jfgm_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p,
            'r': jfgm_result.get('r'),
            'S': jfgm_result.get('S')
        }

    print(f"  Running NGBM...")
    ngbm_result = ngbm_predict(train_data, n_forecast=n_forecast)
    if ngbm_result['success']:
        mape_s = calculate_mape(train_data, ngbm_result['fitted'])
        mape_p = calculate_mape(test_data, ngbm_result['forecast'])
        methods['NGBM'] = {
            'fitted': ngbm_result['fitted'],
            'forecast': ngbm_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p,
            'n': ngbm_result['n']
        }

    print(f"  Running LSSVR...")
    lssvr_result = lssvr_predict(train_data, n_forecast=n_forecast, window_size=5)
    if lssvr_result['success']:
        fitted_len = len(lssvr_result['fitted'])
        mape_s = calculate_mape(train_data[-fitted_len:], lssvr_result['fitted'])
        mape_p = calculate_mape(test_data, lssvr_result['forecast'])
        methods['LSSVR'] = {
            'fitted': lssvr_result['fitted'],
            'forecast': lssvr_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p
        }

    print(f"  Running LSTM...")
    lstm_result = lstm_predict(train_data, n_forecast=n_forecast, window_size=5)
    if lstm_result['success']:
        fitted_len = len(lstm_result['fitted'])
        mape_s = calculate_mape(train_data[-fitted_len:], lstm_result['fitted'])
        mape_p = calculate_mape(test_data, lstm_result['forecast'])
        methods['LSTM'] = {
            'fitted': lstm_result['fitted'],
            'forecast': lstm_result['forecast'],
            'mape_s': mape_s,
            'mape_p': mape_p
        }

    results['methods'] = methods
    return results


def plot_comparison(results_list):
    """Plot comparison charts"""
    n_datasets = len(results_list)
    n_cols = 2
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan']

    for idx, results in enumerate(results_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        name = results['name']
        indicator = results['indicator']
        train_data = results['train_data']
        test_data = results['test_data']
        methods = results['methods']
        train_years = results['train_years']
        test_years = results['test_years']

        ax.plot(train_years, train_data, 'ko-', label='Train Actual', markersize=6, linewidth=2)
        ax.plot(test_years, test_data, 'ks--', label='Test Actual', markersize=8, linewidth=2)

        for i, (method_name, method_data) in enumerate(methods.items()):
            if len(method_data['forecast']) == len(test_years):
                ax.plot(test_years, method_data['forecast'], marker='o',
                        color=colors[i % len(colors)], linestyle='--',
                        label=f"{method_name} (MAPE_p={method_data['mape_p']:.2f}%)", alpha=0.7)

        ax.axvline(x=2021.5, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(f"{name} - {indicator}", fontsize=12, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(results['unit'])
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('7methods_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    print("Comparison saved: 7methods_comparison.png")


def save_results_to_csv(results_list):
    """Save detailed results to CSV"""
    all_data = []

    for results in results_list:
        name = results['name']
        indicator = results['indicator']
        train_data = results['train_data']
        test_data = results['test_data']
        train_years = results['train_years']
        test_years = results['test_years']
        methods = results['methods']

        for i, (year, value) in enumerate(zip(train_years, train_data)):
            all_data.append({'Province': name, 'Indicator': indicator, 'Year': year, 'Type': 'Train Actual', 'Value': value})

        for year, value in zip(test_years, test_data):
            all_data.append({'Province': name, 'Indicator': indicator, 'Year': year, 'Type': 'Test Actual', 'Value': value})

        for method_name, method_data in methods.items():
            forecast = method_data['forecast']
            for i, (year, value) in enumerate(zip(test_years, forecast)):
                all_data.append({'Province': name, 'Indicator': indicator, 'Year': year, 'Type': f'{method_name} Forecast', 'Value': value})

    df = pd.DataFrame(all_data)
    df.to_csv('prediction_results_7methods.csv', index=False, encoding='utf-8-sig')
    print("Results saved: prediction_results_7methods.csv")


def print_summary_table(results_list):
    """Print summary table"""
    print("\n" + "=" * 120)
    print("7 Methods MAPE Comparison Summary")
    print("=" * 120)
    print(f"{'Province-Indicator':<25} {'ARIMA':>12} {'GM(1,1)':>12} {'FGM':>12} {'JFGM':>12} {'NGBM':>12} {'LSSVR':>12} {'LSTM':>12}")
    print("-" * 120)

    method_names = ['ARIMA', 'GM(1,1)', 'FGM', 'JFGM', 'NGBM', 'LSSVR', 'LSTM']

    for results in results_list:
        label = f"{results['name']}-{results['indicator']}"
        row = f"{label:<25}"

        for method in method_names:
            if method in results['methods']:
                mape = results['methods'][method]['mape_p']
                row += f"{mape:>11.2f}% "
            else:
                row += f"{'N/A':>12} "

        print(row)

    print("=" * 120)

    print("\nBest MAPE_p count per method:")
    method_best_count = {m: 0 for m in method_names}

    for results in results_list:
        best_method = None
        best_mape = float('inf')

        for method, data in results['methods'].items():
            if data['mape_p'] < best_mape:
                best_mape = data['mape_p']
                best_method = method

        if best_method:
            method_best_count[best_method] += 1

    for method, count in sorted(method_best_count.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count}")


def print_detailed_predictions(results_list):
    """Print detailed predictions"""
    for results in results_list:
        name = results['name']
        indicator = results['indicator']
        train_data = results['train_data']
        test_data = results['test_data']
        train_years = results['train_years']
        test_years = results['test_years']
        methods = results['methods']

        print(f"\n{'=' * 100}")
        print(f"{name} - {indicator}")
        print(f"{'=' * 100}")

        print(f"\nTraining Data (2010-2021):")
        print(f"{'Year':<8}", end="")
        for y in train_years:
            print(f"{y:<12}", end="")
        print()
        print(f"{'Actual':<8}", end="")
        for v in train_data:
            print(f"{v:<12.2f}", end="")
        print()

        for method_name, method_data in methods.items():
            fitted = method_data['fitted']
            if method_name in ['LSSVR', 'LSTM']:
                padding = '         '
                print(f"{method_name:<8}", end="")
                print(f"{padding:<12}", end="")
                for v in fitted:
                    print(f"{v:<12.2f}", end="")
                print()
            else:
                print(f"{method_name:<8}", end="")
                for v in fitted:
                    print(f"{v:<12.2f}", end="")
                print()

        print(f"\nTest Data (2022-2024):")
        print(f"{'Year':<8}", end="")
        for y in test_years:
            print(f"{y:<12}", end="")
        print()
        print(f"{'Actual':<8}", end="")
        for v in test_data:
            print(f"{v:<12.2f}", end="")
        print()

        for method_name, method_data in methods.items():
            forecast = method_data['forecast']
            mape_p = method_data['mape_p']
            print(f"{method_name:<8}", end="")
            for v in forecast:
                print(f"{v:<12.2f}", end="")
            print(f" [MAPE={mape_p:.2f}%]")

        print()


def print_all_results_summary(results_list):
    """Print complete results summary"""
    method_names = ['ARIMA', 'GM(1,1)', 'FGM', 'JFGM', 'NGBM', 'LSSVR', 'LSTM']

    print("\n" + "=" * 150)
    print("Complete Prediction Results Summary")
    print("=" * 150)

    for results in results_list:
        name = results['name']
        indicator = results['indicator']
        train_data = results['train_data']
        test_data = results['test_data']
        train_years = results['train_years']
        test_years = results['test_years']
        methods = results['methods']

        print(f"\n[{name} - {indicator}]")
        print("-" * 150)

        print(f"{'Method':<12}", end="")
        for y in train_years:
            print(f"{'Train'+str(y):<10}", end="")
        print("  TrainMAPE", end="")
        for y in test_years:
            print(f"{'Pred'+str(y):<10}", end="")
        print("  TestMAPE")
        print("-" * 150)

        print(f"{'Actual':<12}", end="")
        for v in train_data:
            print(f"{v:<10.2f}", end="")
        print(" " * 12, end="")
        for v in test_data:
            print(f"{v:<10.2f}", end="")
        print()

        for method_name in method_names:
            if method_name in methods:
                method_data = methods[method_name]
                fitted = method_data['fitted']
                forecast = method_data['forecast']
                mape_s = method_data.get('mape_s', 0)
                mape_p = method_data['mape_p']

                print(f"{method_name:<12}", end="")

                if method_name in ['LSSVR', 'LSTM']:
                    print(f"{'':>10}", end="")
                    for v in fitted:
                        print(f"{v:<10.2f}", end="")
                else:
                    for v in fitted:
                        print(f"{v:<10.2f}", end="")

                print(f"{mape_s:>8.2f}% ", end="")

                for v in forecast:
                    print(f"{v:<10.2f}", end="")

                print(f"{mape_p:>8.2f}%")
            else:
                print(f"{method_name:<12}", end="")
                print(" " * 10 * len(train_years), end="")
                print(" " * 12, end="")
                print(" " * 10 * len(test_years), end="")
                print()

    print("\n" + "=" * 150)
    print("Note: MAPE_s = training MAPE; MAPE_p = prediction MAPE")
    print("=" * 150)


def save_complete_results_to_csv(results_list):
    """Save complete results to CSV"""
    method_names = ['ARIMA', 'GM(1,1)', 'FGM', 'JFGM', 'NGBM', 'LSSVR', 'LSTM']
    all_rows = []

    for results in results_list:
        name = results['name']
        indicator = results['indicator']
        train_data = results['train_data']
        test_data = results['test_data']
        train_years = results['train_years']
        test_years = results['test_years']
        methods = results['methods']

        for method_name in method_names:
            if method_name in methods:
                method_data = methods[method_name]
                fitted = method_data['fitted']
                forecast = method_data['forecast']
                mape_s = method_data.get('mape_s', 0)
                mape_p = method_data['mape_p']

                row = {
                    'Province': name,
                    'Indicator': indicator,
                    'Method': method_name,
                    'MAPE_s(%)': round(mape_s, 4),
                    'MAPE_p(%)': round(mape_p, 4)
                }

                for i, (year, actual, fit) in enumerate(zip(train_years, train_data, fitted)):
                    row[f'Train{year}Actual'] = round(actual, 2)
                    if method_name in ['LSSVR', 'LSTM']:
                        row[f'Train{year}Fitted'] = round(fit, 2) if i >= 5 else ''
                    else:
                        row[f'Train{year}Fitted'] = round(fit, 2)

                for year, actual, pred in zip(test_years, test_data, forecast):
                    row[f'Test{year}Actual'] = round(actual, 2)
                    row[f'Test{year}Pred'] = round(pred, 2)

                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    cols = ['Province', 'Indicator', 'Method', 'MAPE_s(%)', 'MAPE_p(%)']
    for year in train_years:
        cols.extend([f'Train{year}Actual', f'Train{year}Fitted'])
    for year in test_years:
        cols.extend([f'Test{year}Actual', f'Test{year}Pred'])

    for col in cols:
        if col not in df.columns:
            df[col] = ''
    df = df[cols]

    df.to_csv('7methods_complete_results.csv', index=False, encoding='utf-8-sig')
    print("Complete results saved: 7methods_complete_results.csv")


def save_summary_mape_csv(results_list):
    """Save MAPE summary to CSV"""
    method_names = ['ARIMA', 'GM(1,1)', 'FGM', 'JFGM', 'NGBM', 'LSSVR', 'LSTM']

    rows = []
    for results in results_list:
        name = results['name']
        indicator = results['indicator']
        label = f"{name}-{indicator}"

        for method_name in method_names:
            if method_name in results['methods']:
                mape_s = results['methods'][method_name].get('mape_s', 0)
                mape_p = results['methods'][method_name]['mape_p']
                rows.append({
                    'Province-Indicator': label,
                    'Method': method_name,
                    'MAPE_s(%)': round(mape_s, 4),
                    'MAPE_p(%)': round(mape_p, 4)
                })

    df = pd.DataFrame(rows)
    df.to_csv('7methods_MAPE_summary.csv', index=False, encoding='utf-8-sig')
    print("MAPE summary saved: 7methods_MAPE_summary.csv")


if __name__ == "__main__":
    print("=" * 80)
    print("Seven Prediction Methods Comparison")
    print("=" * 80)
    print("Train: 2010-2021 (12 years)")
    print("Test: 2022-2024 (3 years)")
    print("=" * 80)

    datasets = get_datasets()
    train_len = 12
    n_forecast = 3

    all_results = []

    for idx, (name, indicator, unit, data) in enumerate(datasets):
        print(f"\n[{idx + 1}/{len(datasets)}] {name} - {indicator}")
        results = run_all_methods(name, indicator, unit, data, train_len, n_forecast)
        all_results.append(results)

    print_summary_table(all_results)
    print_detailed_predictions(all_results)
    print_all_results_summary(all_results)

    save_results_to_csv(all_results)
    save_complete_results_to_csv(all_results)
    save_summary_mape_csv(all_results)

    print("\nGenerating comparison charts...")
    plot_comparison(all_results)

    print("\nDone!")
