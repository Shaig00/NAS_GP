"""
Aggregate the per-channel tail indices and coverage already stored by
gp_runner.py.  Nothing is refitted.

The nu reported here is the nu the intervals were actually built with, so the
results table and the deployed model agree.  A separate bootstrap refit on the
pooled residuals would give tighter confidence intervals but would describe a
quantity no interval was constructed from; if both are reported, label which is
which.

    python gp_aggregate.py

Outputs to OUT_DIR:
    nu_by_channel.csv       nu per market, final split and stability across splits
    coverage_by_channel.csv empirical coverage per market at each nominal level
    nu_table.tex            booktabs table for the results chapter
    summary printed to stdout
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

parent_dir = Path.cwd().parent
RESULTS = f'{parent_dir}/Results_Evaluation'
OUT_DIR = Path('gp_aggregates')

LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)

# order must match the channel order of the price frame
COUNTRIES = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Czechia', 'Estonia',
             'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Latvia',
             'Lithuania', 'Netherlands', 'Poland', 'Portugal', 'Romania',
             'Slovakia', 'Spain']


def countries_from_frame(seq_len=96):
    """Read the channel order from the price frame rather than trusting a
    hardcoded list.  Falls back to COUNTRIES if the frame is not present."""
    path = (f'{parent_dir}/Energy_Production_Data/EU_Time_Series/'
            f'EU_Energy_Prices_Time_Series_{seq_len}_Train.pkl')
    try:
        with open(path, 'rb') as f:
            frame = pickle.load(f)
        cols = [c for c in frame.columns if c not in ('Subset', 'date')]
        return [c.split(' ')[0].replace('_', ' ') for c in cols]
    except Exception as exc:
        print(f'could not read channel order from the frame ({exc}); '
              f'using the hardcoded list')
        return COUNTRIES


def load_gp(tag, name, seq_len, run):
    path = (f'{RESULTS}/GP_{tag}_{name}_'
            f'EU_Energy_Prices_Time_Series_{seq_len}_{run}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)['GP_Results']


def _ordered(results):
    return sorted(results, key=lambda s: int(re.search(r'\d+', s).group()))


# ---------------------------------------------------------------------------
def nu_table(results, channels):
    """
    -> DataFrame indexed by country.

    nu_final  : the last split's value, fitted on the largest bank.  This is
                the headline figure.
    nu_mean   : mean across splits
    nu_sd     : spread across splits.  A small value supports the claim that
                the tail index is a structural property of a market rather than
                a block-level one.
    """
    splits = _ordered(results)
    nu = pd.DataFrame({s: np.atleast_1d(results[s]['calibration']['nu'])
                       for s in splits}, index=channels)

    out = pd.DataFrame({'nu_final': nu[splits[-1]],
                        'nu_mean': nu.mean(axis=1),
                        'nu_sd': nu.std(axis=1),
                        'nu_min': nu.min(axis=1),
                        'nu_max': nu.max(axis=1)})
    return out.sort_values('nu_final'), nu


def coverage_table(results, channels):
    """-> {level: DataFrame [C x splits]}, plus the mean across splits."""
    splits = _ordered(results)
    per_level, means = {}, {}
    for lvl in LEVELS:
        df = pd.DataFrame(
            {s: np.asarray(results[s]['metrics']['picp_t_per_channel'][lvl])
             for s in splits}, index=channels)
        per_level[lvl] = df
        means[lvl] = df.mean(axis=1)
    return per_level, pd.DataFrame(means)


def pooled_metrics(results):
    """Split-averaged scalars, for the results table."""
    splits = _ordered(results)
    rows = []
    for s in splits:
        m = results[s]['metrics']
        rows.append({'split': s,
                     'rmse': m['rmse_point'],
                     'nll_t': m['nll_t'],
                     'pinball_mean': float(np.mean(list(m['pinball'].values()))),
                     'mpiw90': m['sharpness_mpiw_90'],
                     'rho_var_err': m['spearman_var_err'],
                     **{f'picp{int(c*100)}': m['picp_t'][c] for c in LEVELS},
                     **{f'spread{int(c*100)}': m['picp_spread'][c] for c in LEVELS}})
    return pd.DataFrame(rows).set_index('split')


# ---------------------------------------------------------------------------
def write_nu_tex(nu_df, cov_mean, path):
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Fitted Student-$t$ tail index by market, from the final '
             r'walk-forward split, with the spread across splits and the '
             r'empirical coverage of the nominal 90 per cent interval. Markets '
             r'are ordered by tail index; a lower value is a heavier tail.}',
             r'\label{tab:nu-by-channel}',
             r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Market & $\nu$ & s.d.\ across splits & PICP$_{90}$ \\',
             r'\midrule']
    for c, r in nu_df.iterrows():
        lines.append(f'{c} & {r["nu_final"]:.2f} & {r["nu_sd"]:.2f} & '
                     f'{cov_mean.loc[c, 0.9]:.3f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
def main(tag='per_channel', name='ensemble', seq_len=96, run=1):
    OUT_DIR.mkdir(exist_ok=True)
    channels = countries_from_frame(seq_len)
    results = load_gp(tag, name, seq_len, run)
    print(f'{name}, seq_len {seq_len}, run {run}: '
          f'{len(results)} evaluated splits, {len(channels)} channels\n')

    nu_df, nu_raw = nu_table(results, channels)
    cov_per_level, cov_mean = coverage_table(results, channels)
    scalars = pooled_metrics(results)

    stem = f'{tag}_{name}_{seq_len}_{run}'
    nu_df.round(3).to_csv(OUT_DIR / f'nu_by_channel_{stem}.csv')
    cov_mean.round(4).to_csv(OUT_DIR / f'coverage_by_channel_{stem}.csv')
    scalars.round(4).to_csv(OUT_DIR / f'metrics_by_split_{stem}.csv')
    write_nu_tex(nu_df, cov_mean, OUT_DIR / f'nu_table_{stem}.tex')

    print('--- tail index by market, final split ---')
    print(nu_df.round(2).to_string(), '\n')

    print('--- coverage, mean across splits ---')
    print(cov_mean.round(3).to_string(), '\n')

    print('--- pooled, mean across splits ---')
    print(scalars.mean().round(4).to_string(), '\n')

    print('--- coverage spread across channels ---')
    for lvl in LEVELS:
        s = cov_per_level[lvl].mean(axis=1)
        print(f'  {int(lvl*100)}%: spread {s.max()-s.min():.3f}, '
              f'mean |error| {np.abs(s - lvl).mean():.3f}')

    heavy = (nu_df['nu_final'] < 2).sum()
    print(f'\nmarkets with nu < 2 (no finite variance): {heavy} of {len(nu_df)}')
    print(f'nu range: {nu_df["nu_final"].min():.2f} '
          f'({nu_df.index[0]}) to {nu_df["nu_final"].max():.2f} '
          f'({nu_df.index[-1]})')
    print(f'max s.d. across splits: {nu_df["nu_sd"].max():.2f} '
          f'({nu_df["nu_sd"].idxmax()})')
    print(f'\nsaved -> {OUT_DIR}/')


if __name__ == '__main__':
    main(tag='shared', name='ensemble', seq_len=48, run=1)
    main(tag='shared', name='ensemble', seq_len=96, run=1)
    main(tag='shared', name='ensemble', seq_len=192, run=1)