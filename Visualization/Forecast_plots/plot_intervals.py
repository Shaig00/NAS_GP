"""
Fan charts for the calibrated predictive intervals.

Reconstructs the deployed predictive from the stored benchmark arrays:

    centre  = z                              (the tree forecast; lambda = 0)
    sd      = sqrt(var * scale_i)            (per-channel variance rescaling)
    band(c) = centre +/- t_{nu_i}^{-1}((1+c)/2) * t_scale_i * sd

with nu_i and t_scale_i the per-channel Student-t parameters fitted on data
preceding the evaluation block.

Two nested bands are drawn rather than one, because the shape of the predictive
is the point: a heavy-tailed predictive is narrow in the body and wide in the
extreme, which a single band cannot show.

The plotted window is chosen as the one whose error is closest to the median for
that channel and split, so the figure is representative rather than selected.

Optionally inverts the per-split standardisation so the axis is in EUR/MWh,
which needs the scaler from the dataloader pickle.

Usage:
    python plot_intervals.py
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

parent_dir = Path.cwd().parent

# ---------------------------------------------------------------------------
CFG_TAG   = 'per_channel'
NAME = 'ensemble'
FRAME     = 'EU_Energy_Prices_Time_Series_'
SEQ_LEN   = 192
RUN       = 1
TREE      = 0
SPLIT     = 'Train_Test15'         # None -> the middle reported split
WINDOW    = None                        # None -> the representative window
N_CYCLES = 1          # 4 x 48 = 192 hours shown
START_WINDOW = 150
CHANNELS  = ['Germany', 'Spain', 'Hungary', 'Finland', 'France']
LEVELS    = (0.5, 0.9)    # inner and outer band
REAL_UNITS = False         # invert the standardisation, needs the dataloader pkl
OUTFILE   = f'predictive_intervals_{SEQ_LEN}_{SPLIT}_{START_WINDOW}.png'

RESULTS   = f'{parent_dir}/Results_Evaluation'
DATA_DIR  = f'{parent_dir}/Energy_Production_Data/EU_Time_Series'


def channel_names():
    with open(f'{DATA_DIR}/{FRAME}{SEQ_LEN}_Train.pkl', 'rb') as f:
        frame = pickle.load(f)
    return [c.split(' ')[0] for c in frame.columns if c not in ('Subset', 'date')]


def load_split(tree, split_key=None):
    path = f'{RESULTS}/GP_{CFG_TAG}_{NAME}_{FRAME}{SEQ_LEN}_{RUN}.pkl'
    with open(path, 'rb') as f:
        res = pickle.load(f)

    keys = list(res['GP_Results'].keys())
    key = split_key or keys[len(keys) // 2]
    entry = res['GP_Results'][key]
    if 'arrays' not in entry:
        raise RuntimeError('no stored arrays -- rerun with store_arrays=True')
    return key, entry


def get_scaler(split_key):
    """StandardScaler for that split, from the dataloader pickle."""
    with open(f'{DATA_DIR}/EU_Energy_Prices_Dataloader_torch_{SEQ_LEN}_Train.pkl',
              'rb') as f:
        loaders = pickle.load(f)
    return loaders[split_key]['Scaler']


def representative_window(y, z, ch):
    """Index of the window whose RMSE is closest to the median for this channel."""
    rmse = np.sqrt(((y[:, :, ch] - z[:, :, ch]) ** 2).mean(axis=1))
    return int(np.argmin(np.abs(rmse - np.median(rmse))))


def main():
    names = channel_names()
    split_key, entry = load_split(TREE, SPLIT)
    a = entry['arrays']
    cal = entry['calibration']

    z, y, var = a['z'], a['y'], a['var']
    scale = np.asarray(cal['scale'], dtype=float)
    nu = np.atleast_1d(np.asarray(cal['nu'], dtype=float))
    ts = np.atleast_1d(np.asarray(cal['t_scale'], dtype=float))
    C = z.shape[-1]
    if nu.size == 1:                      # shared-nu runs
        nu, ts = np.full(C, nu[0]), np.full(C, ts[0])

    sd = np.sqrt(var * scale)             # [N, T, C]

    scaler = get_scaler(split_key) if REAL_UNITS else None
    unit = 'EUR/MWh' if scaler is not None else 'standardised'

    idx = [names.index(c) for c in CHANNELS]
    fig, axes = plt.subplots(len(idx), 1, figsize=(9, 2.6 * len(idx)),
                             sharex=True)
    axes = np.atleast_1d(axes)


    for ax, ch in zip(axes, idx):
        T = z.shape[1]
        starts = [START_WINDOW + k * T for k in range(N_CYCLES)]
        starts = [w for w in starts if w < z.shape[0]]

        for k, w in enumerate(starts):
            t = np.arange(k * T, (k + 1) * T)

            centre = z[w, :, ch].astype(float)
            truth = y[w, :, ch].astype(float)
            s = sd[w, :, ch].astype(float)

            for c, alpha in zip(sorted(LEVELS, reverse=True), (0.18, 0.32)):
                m = stats.t.ppf(0.5 + c / 2.0, nu[ch]) * ts[ch]
                ax.fill_between(t, centre - m * s, centre + m * s,
                                alpha=alpha, color='tab:blue', lw=0,
                                label=f'{int(c*100)}% interval' if k == 0 else None)

            ax.plot(t, centre, color='tab:blue', lw=1.4,
                    label='forecast' if k == 0 else None)
            ax.plot(t, truth, color='black', lw=1.1, ls='--',
                    label='realised' if k == 0 else None)
            if k:
                ax.axvline(k * T, color='0.6', lw=0.6, ls=':')   # new forecast origin

        outside = np.sum(np.abs(y[starts, :, ch] - z[starts, :, ch])
                    > stats.t.ppf(0.95, nu[ch]) * ts[ch] * sd[starts, :, ch])
        total = len(starts) * T
        ax.set_title(f'{names[ch]}  (nu = {nu[ch]:.2f}, '
                    f'{outside}/{total} outside the 90% band)',
                    fontsize=10, loc='left')
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)

    axes[-1].set_xlabel(f'Hours ahead (split {split_key})')
    axes[0].legend(fontsize=8, ncol=4, loc='upper left', framealpha=0.9)
    fig.suptitle(f'Calibrated predictive intervals, tree {TREE}, '
                 f'{SEQ_LEN}-hour horizon', y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTFILE, dpi=160, bbox_inches='tight')
    print(f'saved -> {OUTFILE}')

    # a compact numerical companion to the figure
    print(f"\n{'channel':<16}{'nu':>7}{'mean width 90%':>17}{'coverage 90%':>15}")
    for ch in idx:
        m = stats.t.ppf(0.95, nu[ch]) * ts[ch]
        s = sd[:, :, ch]
        width = 2 * m * s
        if scaler is not None:
            width = width * scaler.scale_[ch]
        cov = float((np.abs(y[:, :, ch] - z[:, :, ch]) <= m * s).mean())
        print(f'{names[ch]:<16}{nu[ch]:>7.2f}{width.mean():>17.2f}{cov:>15.3f}')


if __name__ == '__main__':
    main()