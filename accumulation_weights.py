"""
CPCFAGO and CPCFIAGO Operator Weight Analysis
Visualize weights for different sigma values
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Plot settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})


def get_L_weights(k_total, sigma, c=1.0):
    # CPCFIAGO inverse accumulation weights
    lam = np.exp(-sigma / (1 - sigma))
    W = (1 - lam) / sigma if sigma != 0 else 1.0

    weights = np.zeros(k_total)
    for idx in range(k_total):
        i = idx + 1
        if k_total == 1 and i == 1:
            weights[idx] = W * (1 - sigma)
        elif k_total > 1 and i == k_total:
            weights[idx] = W * ((1 - sigma) + sigma * (c**sigma))
        elif k_total > 1 and i == 1:
            weights[idx] = W * (lam**(k_total - 1) * (1 - sigma) - lam**(k_total - 2) * (sigma * c**sigma))
        elif k_total > 2 and 1 < i < k_total:
            weights[idx] = W * (lam**(k_total - i) * ((1 - sigma) + sigma * c**sigma) - lam**(k_total - i - 1) * (sigma * c**sigma))
    return weights


def get_A_weights(k_total, r, c=1.0):
    # CPCFAGO accumulation weights
    lam_r = np.exp(-r / (1 - r))
    W = (1 - lam_r) / r if r != 0 else 1.0
    gamma = (r * c**r) / (1 - r + r * c**r)

    weights = np.zeros(k_total)
    for idx in range(k_total):
        i = idx + 1
        if k_total == 1 and i == 1:
            weights[idx] = 1 / (W * (1 - r))
        elif k_total > 1 and i == k_total:
            weights[idx] = 1 / (W * (1 - r + r * c**r))
        elif k_total > 1 and i == 1:
            weights[idx] = 1 / (W * (1 - r + r * c**r)) * (gamma**(k_total - 2) * (r * c**r / (1 - r)) - lam_r)
        elif k_total > 2 and 1 < i < k_total:
            weights[idx] = 1 / (W * (1 - r + r * c**r)) * gamma**(k_total - i - 1) * (gamma - lam_r)
    return weights


# Parameters
K = 20
c = 1.0
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
i_vals = np.arange(1, K + 1)

# Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# Generate synthetic data
np.random.seed(42)
x_0 = np.linspace(10, 50, K) + np.random.randn(K) * 2

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# Left: CPCFAGO
ax1 = axes[0]
ax1.set_facecolor('white')
for idx, sig in enumerate(sigmas):
    w = get_A_weights(K, sig, c)
    ax1.plot(i_vals, np.abs(w), marker='s', markersize=5, linewidth=1.8,
             color=colors[idx], label=f'$\\sigma={sig}$')

ax1.set_xlabel('Time Step $i$', fontweight='normal')
ax1.set_ylabel('Absolute Weight $|A_{k,i}|$', fontweight='normal')
ax1.set_title('CPCFAGO Weights ($k=20$)', fontweight='bold', pad=10)
ax1.set_xticks(i_vals)
ax1.set_xlim(0.5, K + 0.5)
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fancybox=False)

# Right: CPCFIAGO
ax2 = axes[1]
ax2.set_facecolor('white')
for idx, sig in enumerate(sigmas):
    w = get_L_weights(K, sig, c)
    ax2.plot(i_vals, np.abs(w), marker='o', markersize=5, linewidth=1.8,
             color=colors[idx], label=f'$\\sigma={sig}$')

ax2.set_xlabel('Time Step $i$', fontweight='normal')
ax2.set_ylabel('Absolute Weight $|L_{k,i}|$', fontweight='normal')
ax2.set_title('CPCFIAGO Weights ($k=20$)', fontweight='bold', pad=10)
ax2.set_xticks(i_vals)
ax2.set_xlim(0.5, K + 0.5)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fancybox=False)

plt.tight_layout()
plt.savefig('operator_weights.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('operator_weights_detail.png', dpi=300, bbox_inches='tight', facecolor='white')

# Check new information priority for sigma = 0.05
sig_low = 0.05
w_L_low = get_L_weights(K, sig_low, c)
w_A_low = get_A_weights(K, sig_low, c)

df_low = pd.DataFrame({
    'i': i_vals,
    'L_weight_abs': np.abs(w_L_low),
    'A_weight_abs': np.abs(w_A_low),
    'L_weight_raw': w_L_low,
    'A_weight_raw': w_A_low
})
df_low.to_csv('low_sigma_weights.csv', index=False)

print("Check New Information Priority for sigma = 0.05 (Weights from i=1 to 20):")
print(df_low[['i', 'L_weight_abs', 'A_weight_abs']].to_string(index=False))

# Verify new information priority
priority_L = all(np.abs(w_L_low)[i] <= np.abs(w_L_low)[i+1] for i in range(1, K-1))
priority_A = all(np.abs(w_A_low)[i] <= np.abs(w_A_low)[i+1] for i in range(1, K-1))
print(f"\nCPCFIAGO satisfies new information priority (i>=2)? {priority_L}")
print(f"CPCFAGO satisfies new information priority (i>=2)? {priority_A}")
