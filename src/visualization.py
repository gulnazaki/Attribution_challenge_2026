# src/visualization.py
"""
Evaluation plots for extreme event attribution methods.

Public API
----------
    plot_time_evolution : Rolling Type I error and power over time.
    plot_qq_analysis    : Log-log QQ plot and power curves.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_time_evolution(df, algo_groups, window=1, save_path='figures/evolution.png'):
    """
    Rolling Type I error and statistical power over time.

    Two sub-panels per algorithm group:
        Left  — yearly Type I error rate (counterfactual scenario)
        Right — yearly statistical power  (factual scenario)

    Solid lines = alpha 5 %,  dashed lines = alpha 1 %.

    Parameters
    ----------
    df          : pd.DataFrame
        Results with columns 'scenario', 'time' (datetime), and one PN
        column per method.
    algo_groups : dict
        { group_label: [(display_label, column_name, color), ...] }
    window      : int   Rolling window in years (default 1 = no smoothing).
    save_path   : str   Output path (parent directory must exist).
    """
    def _rate(sub_df, col, alpha, win):
        rejected    = (sub_df[col] > (1 - alpha)).astype(float)
        yearly_rate = rejected.groupby(sub_df['time'].dt.year).mean()
        return yearly_rate.rolling(window=win, center=True).mean()

    df_null = df[df['scenario'] == 'counterfactual']
    df_fact = df[df['scenario'] == 'factual']

    n_groups = len(algo_groups)
    fig, axes = plt.subplots(n_groups, 2,
                             figsize=(14, 3 * n_groups),
                             sharex=True, squeeze=False)

    for i, (algo_name, configs) in enumerate(algo_groups.items()):
        ax_err, ax_pow = axes[i, 0], axes[i, 1]

        for label, col, color in configs:
            if col not in df.columns:
                continue
            ax_err.plot(_rate(df_null, col, 0.05, window).index,
                        _rate(df_null, col, 0.05, window),
                        color=color, lw=1.8, label=label)
            ax_err.plot(_rate(df_null, col, 0.01, window).index,
                        _rate(df_null, col, 0.01, window),
                        color=color, lw=1.2, ls='--', alpha=0.7)
            ax_pow.plot(_rate(df_fact, col, 0.05, window).index,
                        _rate(df_fact, col, 0.05, window),
                        color=color, lw=1.8)
            ax_pow.plot(_rate(df_fact, col, 0.01, window).index,
                        _rate(df_fact, col, 0.01, window),
                        color=color, lw=1.2, ls='--', alpha=0.7)

        ax_err.axhline(0.05, color='black', lw=0.8, ls=':', alpha=0.5)
        ax_err.set_title(f'{algo_name} — Type I Error (5 % & 1 %)',
                         loc='left', fontsize=10, fontweight='bold')
        ax_pow.set_title(f'{algo_name} — Statistical Power',
                         loc='left', fontsize=10, fontweight='bold')
        for ax in (ax_err, ax_pow):
            ax.grid(True, alpha=0.2)
        ax_err.legend(loc='upper left', fontsize=7, frameon=True)

    axes[-1, 0].set_xlabel('Year')
    axes[-1, 1].set_xlabel('Year')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')


def plot_qq_analysis(df, algo_groups, save_path='figures/qq_analysis.png'):
    """
    Log-log QQ plot of Type I error control and power curves.

    Panel A (left)  — Observed false-positive rate vs. nominal alpha (log-log).
                      Perfect calibration lies on the diagonal.
    Panel B (right) — Statistical power vs. significance level (log-x).

    Line style encodes the PN statistical estimator:
        solid  -> Empirical  label contains '(Emp)'
        dashed -> GEV        label contains '(GEV)'
        dotted -> Gaussian   label contains '(Norm)'

    Parameters
    ----------
    df          : pd.DataFrame
    algo_groups : dict   Same structure as plot_time_evolution.
    save_path   : str
    """
    df_null = df[df['scenario'] == 'counterfactual']
    df_fact = df[df['scenario'] == 'factual']
    alphas  = np.geomspace(1e-4, 1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    style_map = {'(Emp)': '-', '(GEV)': '--', '(Norm)': ':'}

    for algo_name, configs in algo_groups.items():
        for label, col, color in configs:
            if col not in df.columns:
                continue
            ls       = next((s for k, s in style_map.items() if k in label), '-')
            observed = [np.mean(df_null[col] > (1 - a)) for a in alphas]
            power    = [np.mean(df_fact[col] > (1 - a)) for a in alphas]
            ax1.plot(alphas, observed, label=label, color=color, ls=ls, lw=2)
            ax2.plot(alphas, power,    label=label, color=color, ls=ls, lw=2)

    # Perfect-calibration diagonal
    ax1.plot([1e-4, 1], [1e-4, 1], color='black', lw=1, ls='-', alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('A — Type I Error Control (QQ)', loc='left', fontweight='bold')
    ax1.set_xlabel(r'Nominal $\alpha$')
    ax1.set_ylabel('Observed Rate')
    ax1.grid(True, which='both', ls=':', alpha=0.3)

    ax2.set_xscale('log')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('B — Statistical Power', loc='left', fontweight='bold')
    ax2.set_xlabel(r'Significance Level ($\alpha$)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, which='both', ls=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')
