"""
Figures and descriptive statistics for Section 4.1.

Outputs, written to OUT_DIR as vector PDFs:

  fig_prices_weekly.pdf         daily mean price, nineteen countries
  fig_correlation.pdf          19 x 19 correlation heatmap
  fig_single_country.pdf       four weeks of hourly prices, weekends shaded
  fig_price_density_panel.pdf  per-country density, small multiples
  fig_gmm_fit.pdf              mixture components against the empirical density
  fig_regime_summary.pdf       near-zero weight, regime mean, regime volatility
  table_summary_stats.tex      booktabs table, one row per country
  tail_statistics.txt          kurtosis, regime kurtosis, Jarque-Bera

"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.mixture import GaussianMixture

parent_dir = Path.cwd().parent

# ---------------------------------------------------------------------------
# The 192-hour frame spans the whole modelling window and contains the 48- and 96-hour windows
SEQ_LEN    = 192
FRAME_PATH = (f'{parent_dir}/Energy_Production_Data/EU_Time_Series/'
              f'EU_Energy_Prices_Time_Series_{SEQ_LEN}_Train.pkl')
OUT_DIR    = 'figures_section41'

# notebook settings, kept so results stay comparable
GMM_THRESHOLD    = 20.0
GMM_RANDOM_STATE = 123
GMM_MAX_ITER     = 500
KDE_BW_ADJUST    = 0.5          
DENSITY_XLIM     = (-50, 300)

# The notebook's close-up asked for 2024-01-01 to 2024-01-28, but the frame
# starts on 2024-01-10, so it silently showed only 18 days under a title claiming
# four weeks.  This window lies wholly inside the sample.
CLOSEUP_COUNTRY = 'Germany'
CLOSEUP_START   = '2024-03-04'  # a Monday
CLOSEUP_WEEKS   = 4

CLUSTER_HEATMAP = True          # group coupled markets; False = alphabetical

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'pdf.fonttype': 42,
        'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})


# ---------------------------------------------------------------------------
def load_prices():
    with open(FRAME_PATH, 'rb') as f:
        frame = pickle.load(f)
    cols = [c for c in frame.columns if c not in ('Subset', 'date')]
    prices = frame[cols].astype(float).copy()
    prices.columns = [c.split(' ')[0].replace('_', ' ') for c in cols]
    prices.index = pd.to_datetime(prices.index)
    return prices


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'saved -> {path}')


def kde(x, grid, bw_adjust=KDE_BW_ADJUST):
    """Gaussian KDE with a seaborn-style bandwidth multiplier."""
    k = stats.gaussian_kde(x)
    k.set_bandwidth(k.factor * bw_adjust)
    return k(grid)


def grid_axes(n, ncols=5, height=2.0, sharex=True, sharey=False):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.3 * ncols / 5, height * nrows),
                             sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


# ---------------------------------------------------------------------------
# Gaussian mixture
# ---------------------------------------------------------------------------
def fit_gmm(x):
    spike, heavy = x[x < GMM_THRESHOLD], x[x >= GMM_THRESHOLD]
    # guard: a country with no near-zero observations cannot seed the low mode
    if len(spike) < 2 or len(heavy) < 2:
        return None
    gm = GaussianMixture(
        n_components=2, covariance_type='full',
        means_init=np.array([[0], [heavy.mean()]]),
        precisions_init=np.array([[[1 / 10 ** 2]],
                                  [[1 / heavy.std() ** 2]]]),
        max_iter=GMM_MAX_ITER, n_init=1, random_state=GMM_RANDOM_STATE,
    )
    return gm.fit(x.reshape(-1, 1))


def regime_indices(gm):
    """Operating regime = higher fitted mean, whatever its component label."""
    means = gm.means_.ravel()
    return int(np.argmin(means)), int(np.argmax(means))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_prices_weekly(prices):
    import matplotlib.dates as mdates
    
    daily = prices.resample('W').mean()
    med = daily.median(axis=1)
    q25, q75 = daily.quantile(0.25, axis=1), daily.quantile(0.75, axis=1)
    lo, hi = daily.min(axis=1), daily.max(axis=1)

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.fill_between(daily.index, lo, hi, color='0.85', lw=0, label='range across countries')
    ax.fill_between(daily.index, q25, q75, color='0.65', lw=0, label='interquartile range')
    ax.plot(daily.index, med, color='black', lw=0.9, label='median')
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    highlight = {'Germany': 'tab:blue', 'Finland': 'tab:red',
                 'Spain': 'tab:orange', 'Hungary': 'tab:purple'}
    for c, col in highlight.items():
        ax.plot(daily.index, daily[c], lw=0.7, color=col, label=c)

    ax.axhline(0, color='black', lw=0.4)
    ax.set_ylabel('EUR/MWh (weekly mean)')
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.legend(ncol=4, loc='upper left', frameon=False, fontsize=7)
    save(fig, 'fig_prices_weekly.pdf')


def fig_correlation(prices, lag=None, order=None):
    """Correlation heatmap. With lag=None the correlation is of price levels;
    with lag=24 it is of day-over-day changes, which removes both the common
    price level and the shared intraday shape. Pass order to reuse the market
    ordering of an earlier figure so the two are comparable."""
    x = prices if lag is None else prices.diff(lag).dropna()
    corr = x.corr()

    if order is not None:
        corr = corr.loc[order, order]
    elif CLUSTER_HEATMAP:
        dist = squareform(1.0 - corr.values, checks=False)
        corr = corr.iloc[(o := leaves_list(linkage(dist, method='average'))), o]

    # all-positive matrices waste half a diverging map, so use a sequential one
    neg = corr.values.min() < 0
    cmap, vmin, vmax = ('RdBu_r', -1, 1) if neg else ('Reds', 0, 1)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    cut = vmin + 0.65 * (vmax - vmin)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=4.2,
                color='white' if abs(v) > abs(cut) else 'black')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Pearson correlation' if lag is None
                   else f'Pearson correlation, $\\Delta_{{{lag}}}$ prices')

    save(fig, 'fig_correlation.pdf' if lag is None
              else f'fig_correlation_diff{lag}.pdf')
    return corr

def fig_single_country(prices):
    start = pd.Timestamp(CLOSEUP_START)
    end = start + pd.Timedelta(weeks=CLOSEUP_WEEKS) - pd.Timedelta(hours=1)
    s = prices.loc[start:end, CLOSEUP_COUNTRY]
    if s.empty:
        raise ValueError(f'no data for {CLOSEUP_COUNTRY} in {start}..{end}')
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    ax.plot(s.index, s.values, lw=0.8, color='tab:blue')
    for day in pd.date_range(start, end, freq='D'):
        if day.weekday() >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1), color='grey',
                       alpha=0.12, lw=0)
    ax.set_ylabel('EUR/MWh')
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, 'fig_single_country.pdf')
    return start, end


def fig_price_density_panel(prices):
    grid = np.linspace(*DENSITY_XLIM, 800)
    fig, axes = grid_axes(prices.shape[1], sharex=True, sharey=True)
    for ax, c in zip(axes, prices.columns):
        ax.plot(grid, kde(prices[c].dropna().values, grid), lw=1.0, color='tab:blue')
        ax.axvline(0, color='black', lw=0.4, ls=':')
        ax.set_title(c, pad=2)
        ax.set_xlim(*DENSITY_XLIM)
        ax.grid(alpha=0.2, lw=0.4)
    fig.supxlabel('Day-ahead price (EUR/MWh)', fontsize=9)
    fig.supylabel('Density', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig_price_density_panel.pdf')


def fig_gmm_fit(prices, gmms):
    grid = np.linspace(*DENSITY_XLIM, 1200)
    fig, axes = grid_axes(prices.shape[1], sharex=True, sharey=False)
    for ax, c in zip(axes, prices.columns):
        x = prices[c].dropna().values
        ax.plot(grid, kde(x, grid, bw_adjust=0.4), lw=1.3, color='steelblue',
                label='empirical')
        gm = gmms.get(c)
        if gm is not None:
            lo, hi = regime_indices(gm)
            for idx, colour, name in ((lo, 'tomato', 'near-zero'),
                                      (hi, 'seagreen', 'operating')):
                mu = gm.means_.ravel()[idx]
                sd = np.sqrt(gm.covariances_.ravel()[idx])
                w = gm.weights_[idx]
                ax.plot(grid, w * stats.norm.pdf(grid, mu, sd), ls='--', lw=1.0,
                        color=colour, label=name)
        ax.set_title(c, pad=2)
        ax.set_xlim(*DENSITY_XLIM)
        ax.grid(alpha=0.2, lw=0.4)
        ax.tick_params(labelleft=False)
    axes[0].legend(fontsize=5.5, loc='upper right', frameon=False)
    fig.supxlabel('Day-ahead price (EUR/MWh)', fontsize=9)
    fig.supylabel('Density', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig_gmm_fit.pdf')


def fig_regime_summary(tail):
    t = tail.dropna(subset=['low_weight']).sort_values('low_weight', ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.4), sharey=True)
    axes[0].barh(t.index, 100 * t['low_weight'], color='steelblue')
    axes[0].set_xlabel('Near-zero regime weight (%)')
    axes[1].barh(t.index, t['high_mean'], color='tomato')
    axes[1].set_xlabel('Operating regime mean\n(EUR/MWh)')
    axes[2].barh(t.index, t['high_sd'], color='seagreen')
    axes[2].set_xlabel('Operating regime s.d.\n(EUR/MWh)')
    for ax in axes:
        ax.grid(alpha=0.2, lw=0.4, axis='x')
    fig.tight_layout()
    save(fig, 'fig_regime_summary.pdf')


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def tail_statistics(prices):
    rows, gmms = [], {}
    for c in prices.columns:
        x = prices[c].dropna().values
        row = {'country': c, 'excess_kurtosis': stats.kurtosis(x, fisher=True)}

        gm = fit_gmm(x)
        if gm is None:
            row.update(regime_excess_kurtosis=np.nan, low_mean=np.nan,
                       low_weight=np.nan, high_mean=np.nan, high_sd=np.nan,
                       jb_stat=np.nan, jb_p=np.nan, label_swapped=np.nan)
        else:
            gmms[c] = gm
            lo, hi = regime_indices(gm)
            regime = x[gm.predict(x.reshape(-1, 1)) == hi]
            jb = stats.jarque_bera(regime)
            row.update(
                regime_excess_kurtosis=stats.kurtosis(regime, fisher=True),
                low_mean=gm.means_.ravel()[lo], low_weight=gm.weights_[lo],
                high_mean=gm.means_.ravel()[hi],
                high_sd=float(np.sqrt(gm.covariances_.ravel()[hi])),
                jb_stat=jb.statistic, jb_p=jb.pvalue,
                # component 1 = operating regime
                label_swapped=(hi != 1),
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index('country'), gmms


def write_tail_report(tail):
    path = os.path.join(OUT_DIR, 'tail_statistics.txt')
    ok = tail.dropna(subset=['regime_excess_kurtosis'])
    lines = [
        'Tail statistics, modelling window, nineteen countries',
        '=' * 72, '',
        tail.round(3).to_string(), '',
        'All kurtosis figures are EXCESS kurtosis (Gaussian = 0), as in the',
        'earlier notebook, so they compare directly with its values.', '',
        f'mean excess kurtosis, full distribution : {tail["excess_kurtosis"].mean():.2f}',
        f'mean excess kurtosis, operating regime  : {ok["regime_excess_kurtosis"].mean():.2f}',
        f'Jarque-Bera rejects normality at 0.001  : {int((ok["jb_p"] < 1e-3).sum())} of {len(ok)}',
    ]
    swapped = ok.index[ok['label_swapped'] == True].tolist()
    lines.append('')
    lines.append('countries where component 1 was NOT the operating regime: '
                 + (', '.join(swapped) if swapped else 'none'))
    if swapped:
        lines.append('  -> the notebook\'s regime==1 selection computed kurtosis on the')
        lines.append('     near-zero regime for these; the earlier 30 needs this correction.')
    skipped = tail.index[tail['regime_excess_kurtosis'].isna()].tolist()
    if skipped:
        lines.append(f'no near-zero observations, mixture not fitted: {", ".join(skipped)}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'saved -> {path}\n')
    print('\n'.join(lines))


def write_summary_table(prices, tail):
    first = prices.index.min().strftime('%-d %B %Y')
    last = prices.index.max().strftime('%-d %B %Y')
    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\small',
        r'\caption{Day-ahead electricity prices by country, ' + first + r' to '
        + last + r', ' + f'{len(prices):,}' + r' hourly observations. Prices in '
        r'EUR/MWh. Excess kurtosis is zero for a Gaussian distribution.}',
        r'\label{tab:summary-stats}',
        r'\begin{tabular}{@{}lrrrrrr@{}}',
        r'\toprule',
        r'Country & Mean & Std.\ dev. & Min & Max & Excess kurtosis & Negative (\%) \\',
        r'\midrule',
    ]
    for c in prices.columns:
        x = prices[c].dropna()
        lines.append(f'{c} & {x.mean():,.2f} & {x.std():,.2f} & {x.min():,.2f} & '
                     f'{x.max():,.2f} & {tail.loc[c, "excess_kurtosis"]:,.2f} & '
                     f'{100 * (x < 0).mean():.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path = os.path.join(OUT_DIR, 'table_summary_stats.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'saved -> {path}')




# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prices = load_prices()
    print(f'{prices.shape[1]} countries, {len(prices):,} hours, '
          f'{prices.index.min()} -> {prices.index.max()}\n')


    # visualize prices progression
    fig_prices_weekly(prices)
    fig_single_country(prices)
    
    # Price density distribution
    fig_price_density_panel(prices)

    # GMM fit
    tail, gmms = tail_statistics(prices)
    fig_gmm_fit(prices, gmms)
    # GMM summary
    fig_regime_summary(tail)

    # tail statistics
    write_tail_report(tail)
    write_summary_table(prices, tail)

    # correlation heatmaps (including differenced)
    corr = fig_correlation(prices)
    corr_24 = fig_correlation(prices, lag=24, order=corr.columns)

    pass

if __name__ == '__main__':
    main()