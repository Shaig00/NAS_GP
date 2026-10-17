"""Point-forecast metrics for the NAS trees and their ensemble.

    df = tree_table(range(10), [48, 96, 192], parent_dir)      # long, one row per metric set
    summarise(df)                                              # per tree, per horizon

    ens = ensemble_table(range(10), [48, 96, 192], parent_dir)
"""

import pickle
import numpy as np
import pandas as pd
from permetrics.regression import RegressionMetric

N_RUNS = 10
METRICS = ['RMSE', 'MSE', 'MAE', 'MedAE']   # the four TOPSIS uses

THRESH = 100.0
KEY = ['horizon', 'tree', 'run']

def _path(parent_dir, folder_name, tree_idx, seq_len, run, prefix='tree'):
    return (f'{parent_dir}/{folder_name}/'
            f'{prefix}_{tree_idx}_EU_Energy_Prices_Time_Series_{seq_len}_{run}.pkl')


def _run_failed(rmse):
    rmse = np.asarray(rmse, dtype=float)
    diverged = np.any(rmse > THRESH)
    degenerate = np.any(np.isclose(rmse, 1.0, rtol=0.0, atol=1e-9))
    return bool(diverged or degenerate)


def failed_runs(df):
    """-> boolean Series indexed by (horizon, tree, run)."""
    return df.groupby(KEY).RMSE.apply(_run_failed)


def filter_runs(df):
    """Drop failed runs in full. Returns (clean df, discard summary)."""
    failed = failed_runs(df)
    clean = df.set_index(KEY).loc[failed[~failed].index].reset_index()

    summary = (failed.rename('failed').reset_index()
               .groupby(['horizon', 'tree']).failed
               .agg(discarded='sum', total='size'))
    summary['surviving'] = summary.total - summary.discarded
    return clean, summary


def survivors(clean):
    """-> {(horizon, run): [tree ids surviving in that run]}"""
    return (clean.groupby(['horizon', 'run']).tree
            .unique().apply(sorted).to_dict())


def ensemble_filtered(clean, seq_lens, parent_dir, load_run, split_metrics,
                      prefix='chain', n_runs=10):
    """Ensemble over surviving members, formed within each run.

    Membership varies by run, so n_members is carried in the output and must be
    reported alongside the metrics.
    """
    surv = survivors(clean)
    rows = []
    for seq_len in seq_lens:
        for run in range(1, n_runs + 1):
            trees = surv.get((seq_len, run), [])
            if not trees:
                continue
            members = [load_run(parent_dir, t, seq_len, run, prefix) for t in trees]
            for subset in members[0]:
                true = members[0][subset][0]
                pred = np.mean([m[subset][1] for m in members], axis=0)
                rows.append({'tree': 'ensemble', 'run': run, 'horizon': seq_len,
                             'split': subset, 'n_members': len(trees),
                             **split_metrics(true, pred)})
    return pd.DataFrame(rows)


def _stack(batches):
    """Batch list -> one array. The last batch is short, so this is not reshape."""
    if isinstance(batches, (list, tuple)):
        return np.concatenate([np.asarray(b) for b in batches], axis=0)
    return np.asarray(batches)


def load_run(parent_dir, folder_name, tree_idx, seq_len, run, prefix='tree'):
    """-> {subset: (true, pred)}, each concatenated over batches."""
    with open(_path(parent_dir, folder_name, tree_idx, seq_len, run, prefix), 'rb') as f:
        r = pickle.load(f)

    out = {}
    for subset, entry in r['Results'].items():
        true, pred = _stack(entry[1]), _stack(entry[2])
        assert true.shape == pred.shape, f'{subset}: {true.shape} vs {pred.shape}'
        out[subset] = (true, pred)
    return out


def split_metrics(true, pred):
    m = RegressionMetric(np.asarray(true).reshape(-1), np.asarray(pred).reshape(-1))
    return {'RMSE': m.root_mean_squared_error(),
            'MSE': m.mean_squared_error(),
            'MAE': m.mean_absolute_error(),
            'MedAE': m.median_absolute_error()}


def tree_table(tree_idxs, seq_lens, parent_dir, folder_name, n_runs=N_RUNS, prefix='tree'):
    """Long table: one row per (tree, run, horizon, split)."""
    rows = []
    for tree_idx in tree_idxs:
        for seq_len in seq_lens:
            for run in range(1, n_runs + 1):
                for subset, (true, pred) in load_run(parent_dir, folder_name, tree_idx, seq_len, run, prefix).items():
                    rows.append({'tree': tree_idx, 'run': run, 'horizon': seq_len,
                                 'split': subset, **split_metrics(true, pred)})
    return pd.DataFrame(rows)


def ensemble_table(tree_idxs, seq_lens, parent_dir, folder_name, n_runs=N_RUNS):
    """Ensemble formed within each run, so the run-to-run spread survives."""
    rows = []
    for seq_len in seq_lens:
        for run in range(1, n_runs + 1):
            members = [load_run(parent_dir, folder_name, t, seq_len, run) for t in tree_idxs]
            for subset in members[0]:
                true = members[0][subset][0]
                pred = np.mean([m[subset][1] for m in members], axis=0)
                rows.append({'tree': 'ensemble', 'run': run, 'horizon': seq_len,
                             'split': subset, **split_metrics(true, pred)})
    return pd.DataFrame(rows)

def pretrained_table(seq_lens, parent_dir, folder_name, models = ['Sundial', 'GPT4TS', 'TimeMoe', 'TimesFM'], 
                     n_runs=N_RUNS, key='EU_Energy_Prices_Time_Series'):
    """Long table: one row per (run, horizon, split)."""
    rows = []
    for seq_len in seq_lens:
        for model in models:
            if model in ['TimeMoe', 'TimesFM']:
                with open(f'{parent_dir}/{folder_name}/{model}_{key}_{seq_len}_1.pkl', 'rb') as f:
                    r = pickle.load(f)
                split_idx = 1
                for subset, entry in r['Results'].items():
                    true, pred = _stack(entry[0]), _stack(entry[1])
                    assert true.shape == pred.shape, f'{subset}: {true.shape} vs {pred.shape}'
                    rows.append({'tree': model, 'run': 1, 'horizon': seq_len,
                                    'split': f'Train_Test{split_idx}', **split_metrics(true, pred)})
                    split_idx += 1
            else:
                for run in range(1, n_runs + 1):
                    with open(f'{parent_dir}/{folder_name}/{model}_{key}_{seq_len}_{run}.pkl', 'rb') as f:
                        r = pickle.load(f)
                    split_idx = 1
                    for subset, entry in r['Results'].items():
                        true, pred = _stack(entry[0]), _stack(entry[1])
                        assert true.shape == pred.shape, f'{subset}: {true.shape} vs {pred.shape}'
                        rows.append({'tree': model, 'run': run, 'horizon': seq_len,
                                        'split': f'Train_Test{split_idx}', **split_metrics(true, pred)})
                        split_idx += 1
    return pd.DataFrame(rows)

def multivariate_table(seq_lens, parent_dir, folder_name, 
                       models = ['Basisformer', 'Dlinear', 'PatchTST', 'TSMixer', 'MTSMixer'], 
                       n_runs=N_RUNS, key='EU_Energy_Prices_Time_Series'):
    """Long table: one row per (run, horizon, split)."""
    rows = []
    for seq_len in seq_lens:
        for model in models:
            for run in range(1, n_runs + 1):
                with open(f'{parent_dir}/{folder_name}/{model}_{key}_{seq_len}_{run}.pkl', 'rb') as f:
                    r = pickle.load(f)
                split_idx = 1
                for subset, entry in r['Results'].items():
                    true, pred = _stack(entry[1]), _stack(entry[2])
                    assert true.shape == pred.shape, f'{subset}: {true.shape} vs {pred.shape}'
                    rows.append({'tree': model, 'run': run, 'horizon': seq_len,
                                    'split': f'Train_Test{split_idx}', **split_metrics(true, pred)})
                    split_idx += 1
    return pd.DataFrame(rows)

def autogluon_table(parent_dir, folder_name, seq_lens, n_runs=N_RUNS, key='NASTrain_Energy_Prices_Data'):
    """Long table: one row per (run, horizon, split)."""
    rows = []
    for run in range(1, n_runs + 1):
        with open(f'{parent_dir}/{folder_name}/AutoGluonTS_NAS_Res_RMSE_{run}.pkl', 'rb') as f:
            r = pickle.load(f)
        for seq_len in seq_lens:
            split_idx = 2
            for split, entry in r[key][seq_len]['Results'].items():
                true, pred = _stack(entry['True_Values']), _stack(entry['Avg_Predictions'])
                rows.append({'tree': 'AutoGluon', 'run': run, 'horizon': seq_len,
                             'split': f'Train_Test{split_idx}', **split_metrics(true, pred)})
                split_idx += 1
    return pd.DataFrame(rows)




def summarise(df, by=('tree', 'horizon')):
    """Mean over splits and runs, with both sources of spread kept apart."""
    by = list(by)

    per_run = df.groupby(by + ['run'])[METRICS].mean()          # averaged over splits
    per_split = df.groupby(by + ['split'])[METRICS].mean()      # averaged over runs

    out = per_run.groupby(by).mean()
    out = out.join(per_run.groupby(by).std(), rsuffix='_sd_run')
    out = out.join(per_split.groupby(by).std(), rsuffix='_sd_split')
    return out