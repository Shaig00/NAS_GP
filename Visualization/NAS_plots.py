"""
    Figures for Section 5.1. Each function takes the candidate table and returns a fig.

    from nas_figures import block_frequency, score_distribution, penalised_share

    fig = block_frequency(df, position='root_module')
    fig.savefig('fig_block_frequency.pdf')
"""

import numpy as np
import matplotlib.pyplot as plt

SPACE_COLOURS = {'tree': 'tab:blue', 'chain': 'tab:orange'}
EARLY_LATE_ALPHA = {'early': 0.45, 'late': 1.0}
WINDOW = 50       # episodes counted as "early" and as "late"
SMOOTH = 25       # moving-average window for per-episode curves


def _episode_window(df, which, n=WINDOW):
    eps = sorted(df.episode.unique())
    keep = eps[:n] if which == 'early' else eps[-n:]
    return df[df.episode.isin(keep)]


def _moving_average(x, w=SMOOTH):
    return np.convolve(np.asarray(x, float), np.ones(w) / w, mode='valid')


def _spaces(df):
    return [s for s in ('tree', 'chain') if s in set(df.space)]


def block_frequency(df, position='root_module', n=WINDOW, figsize=(7.0, 3.4)):
    """Grouped bars: block frequency early vs late, one panel per space."""
    spaces = _spaces(df)
    blocks = sorted(df[position].dropna().unique())
    x = np.arange(len(blocks))

    fig, axes = plt.subplots(1, len(spaces), figsize=figsize,
                             sharey=True, squeeze=False)

    for ax, space in zip(axes[0], spaces):
        d = df[df.space == space]
        for offset, which in ((-0.2, 'early'), (0.2, 'late')):
            window = _episode_window(d, which, n)
            share = window[position].value_counts(normalize=True)
            # print(share)
            ax.bar(x + offset, [share.get(b, 0.0) for b in blocks], width=0.4,
                   color=SPACE_COLOURS[space], alpha=EARLY_LATE_ALPHA[which],
                   label=f'{which} {n} episodes')
        ax.set_xticks(x)
        ax.set_xticklabels(blocks, rotation=35, ha='right', fontsize=7)
        ax.set_title(space, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, lw=0.4, axis='y')

    axes[0][0].set_ylabel('share of sampled architectures')
    fig.tight_layout()
    return fig


MODULE_COL = 'ts_module'
def node_long(df, max_nodes=7, col=MODULE_COL):
    """One row per (architecture, node). Adds node_idx and depth."""
    node_cols = [f'node{i}_{col}' for i in range(max_nodes)
                 if f'node{i}_{col}' in df.columns]
    if not node_cols:
        raise KeyError(f'no node{{i}}_{col} columns in frame')

    keep = [c for c in ('space', 'episode', 'timestep', 'nr_nodes',
                        'score', 'penalised') if c in df.columns]

    long = (df[keep + node_cols]
            .melt(id_vars=keep, value_vars=node_cols,
                  var_name='node', value_name='module')
            .dropna(subset=['module']))

    long['node_idx'] = long['node'].str.extract(r'node(\d+)_').astype(int)
    long['depth'] = np.floor(np.log2(long['node_idx'] + 1)).astype(int)
    return long.sort_values(['episode', 'timestep', 'node_idx'])


def depth_frequency(long, depth, nr_nodes=3, n=50):
    """Early-vs-late block share at one depth, as a frame. One row per block."""
    import pandas as pd
    sel = long[(long.nr_nodes == nr_nodes) & (long.depth == depth)]
    out = {}
    for space, d in sel.groupby('space'):
        eps = sorted(d.episode.unique())
        early = d[d.episode.isin(eps[:n])].module.value_counts(normalize=True)
        late = d[d.episode.isin(eps[-n:])].module.value_counts(normalize=True)
        out[(space, 'early')] = early
        out[(space, 'late')] = late
    return pd.DataFrame(out).fillna(0.0).round(3)

def score_distribution(df, score='score', bins=60, figsize=(7.0, 3.4)):
    """Overlaid score histograms with each space's maximum marked."""
    fig, ax = plt.subplots(figsize=figsize)

    for space in _spaces(df):
        s = df.loc[df.space == space, score].dropna()
        colour = SPACE_COLOURS[space]
        ax.hist(s, bins=bins, density=True, histtype='stepfilled',
                alpha=0.35, color=colour)

        ax.hist(s, bins=bins, density=True, histtype='step',
                lw=1.4, color=colour,
                label=f'{space}  mean {s.mean():.3f}  max {s.max():.3f}')
        ax.axvline(s.max(), color=colour, ls='--', lw=1.0)


    ax.set_xlabel(r'selection score r1+r2')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    return fig


def penalised_share(df, w=SMOOTH, figsize=(7.0, 3.4)):
    """Share of architectures receiving a walk-forward penalty, per episode."""
    fig, ax = plt.subplots(figsize=figsize)

    for space in _spaces(df):
        d = df[df.space == space]
        share = d.groupby('episode').penalised.mean().sort_index()
        colour = SPACE_COLOURS[space]
        ax.plot(share.index, share.values, lw=0.4, color=colour, alpha=0.30)
        ax.plot(share.index[w - 1:], _moving_average(share.values, w),
                lw=1.8, color=colour, label=space)

    ax.set_xlabel('Episode')
    ax.set_ylabel('share of architectures penalised')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    return fig


def search_progress(df, score='score_r1wv', verbose=False, w=SMOOTH, figsize=(7.0, 3.4)):
    """Per-episode max, median and min, plus best-so-far. One space at a time."""
    d = df.sort_values('episode')
    g = d.groupby('episode')[score]
    eps = np.array(sorted(d.episode.unique()))

    per_max = g.max().values
    per_med = g.median().values
    per_min = g.min().values
    per_mean = g.mean().values
    best_so_far = np.maximum.accumulate(per_max)

    if verbose == True:
        print(f'ceiling {best_so_far[0]:.3f} -> {best_so_far[-1]:.3f}  ({100*(best_so_far[-1]/best_so_far[0]-1):+.1f}%)')
        print(f'median  {per_med[:25].mean():.3f} -> {per_med[-25:].mean():.3f}  '
            f'({100*(per_med[-25:].mean()/per_med[:25].mean()-1):+.1f}%)')
        print(f'min     {per_min[:25].mean():.3f} -> {per_min[-25:].mean():.3f}')
        print(f'mean    {per_mean[:25].mean():.3f} -> {per_mean[-25:].mean():.3f}')

    fig, ax = plt.subplots(figsize=figsize)
    x = eps[w - 1:]
    ax.plot(x, _moving_average(per_max, w), lw=1.6, color='tab:red', label='per-episode max')
    ax.plot(x, _moving_average(per_med, w), lw=1.6, color='tab:blue', label='per-episode median')
    ax.plot(x, _moving_average(per_min, w), lw=1.6, color='tab:grey', label='per-episode min')
    ax.plot(eps, best_so_far, lw=1.2, color='black', ls='--', label='best so far')

    ax.set_xlabel('Episode')
    ax.set_ylabel(score)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    return fig


def truncate(df, n_episodes=1000):
    """Keep the first n_episodes episodes, by episode number rather than row count."""
    eps = sorted(df.episode.unique())[:n_episodes]
    return df[df.episode.isin(eps)].copy()


if __name__=='__main__':
    
    import pandas as pd
    df_tree_full = pd.read_parquet('nas_tree_candidates.parquet')
    df = pd.concat([truncate(df_tree_full), pd.read_parquet('nas_chain_candidates.parquet')], ignore_index=True)


    long = node_long(df)
    leaves = long[(long.nr_nodes == 3) & (long.depth == 1)]
    block_frequency(df, position='root_module')#.savefig('fig_leaf_frequency.pdf')
    print(depth_frequency(long, nr_nodes=3, depth=0))



    block_frequency(df, position='leaf_modules').savefig('fig_block_frequency_leaf.pdf')

    score_distribution(df).savefig('fig_score_distribution.pdf')
    penalised_share(df).savefig('fig_penalised_share.pdf')
    search_progress(df_tree_full, score='score', verbose=True).savefig('fig_search_progress.pdf')   # tree only, 1,313
