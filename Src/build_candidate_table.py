"""
Build a flat one-row-per-architecture table from a NAS sampling history.

Usage
-----
    df_tree  = build_table('Results_Evaluation/WV_GED_NAS_Sampling_History.pkl', 'tree')
    df_chain = build_table('.../history_dict.pkl', 'chain')
    df = pd.concat([df_tree, df_chain], ignore_index=True)

Notes
-----
* Score is r1 + r2

"""

import pickle
import hashlib
import json
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append("..")
parent_dir = Path.cwd().parent

PRE = 'Rewards_Episode_'

# per-episode arrays; runtime key casing differs from the others in the history
ARRAY_KEYS = {
    'r1':      'Rewards_Episode_{}',
    'r2':      'Rewards_SecondSubset_Episode_{}',
    'ged':     'Sampled_Networks_Episode_GED_{}',
    'entropy': 'Sampled_Networks_Episode_Entropy_{}',
    'runtime': ('Sampled_networks_Episode_Runtime_{}',
                'Sampled_Networks_Episode_Runtime_{}'),
}

# config fields holding one value per node, keyed by 'node_i'
DICT_FIELDS = ['kernel_size_nodes', 'd_model_nodes', 'patch_len_nodes',
               'train_pe_nodes', 'stride_nodes', 'dff_nodes', 'sampling_nodes',
               'activations_second_nodes']

# config fields holding one value per node, in node order
LIST_FIELDS = ['ts_module_nodes', 'revin_nodes', 'dropout_nodes',
               'activations_nodes']


def _load(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
        return loaded[0]


def _get(h, spec, ep):
    keys = (spec,) if isinstance(spec, str) else spec
    for k in keys:
        k = k.format(ep)
        if k in h:
            return np.asarray(h[k], dtype=float).ravel()
    return None


def inspect(path, ep=None):
    """Print the config keys of one sampled architecture. Use on an unknown run."""
    h = _load(path)
    eps = sorted(int(k[len(PRE):]) for k in h
                 if k.startswith(PRE) and k[len(PRE):].isdigit())
    ep = ep if ep is not None else eps[len(eps) // 2]
    cfg = h[f'Sampled_Networks_Episode_{ep}'][0]
    print(f'{len(eps)} episodes, {eps[0]}..{eps[-1]}')
    print(f'array keys present for episode {ep}:')
    for name, spec in ARRAY_KEYS.items():
        v = _get(h, spec, ep)
        print(f'  {name:8s} {"missing" if v is None else f"len {len(v)}"}')
    print(f'config keys:')
    for k, v in cfg.items():
        print(f'  {k:28s} {type(v).__name__:5s} {v}')
    return cfg


def _flatten_config(cfg):
    """One config dict -> one flat dict of columns."""
    out = {'nr_nodes': cfg.get('nr_nodes'),
           'prediction_projections': cfg.get('prediction_projections')}

    mods = list(cfg.get('ts_module_nodes', []))
    out['module_pattern'] = ','.join(mods)
    out['n_distinct_modules'] = len(set(mods))
    out['homogeneous'] = len(set(mods)) == 1
    out['root_module'] = mods[0] if mods else None
    out['leaf_modules'] = ','.join(mods[1:]) if len(mods) > 1 else ''

    for f in LIST_FIELDS:
        for i, v in enumerate(cfg.get(f, [])):
            out[f'node{i}_{f.replace("_nodes", "")}'] = v
    for f in DICT_FIELDS:
        for k, v in (cfg.get(f) or {}).items():
            out[f'{k.replace("node_", "node")}_{f.replace("_nodes", "")}'] = v

    # aggregates that are convenient for groupbys
    drops = cfg.get('dropout_nodes') or []
    out['dropout_mean'] = float(np.mean(drops)) if len(drops) else np.nan
    ks = list((cfg.get('kernel_size_nodes') or {}).values())
    out['kernel_mean'] = float(np.mean(ks)) if ks else np.nan
    dm = list((cfg.get('d_model_nodes') or {}).values())
    out['d_model_mean'] = float(np.mean(dm)) if dm else np.nan

    out['config_hash'] = hashlib.md5(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return out


def build_table(path, space, exclude_reward=None):
    h = _load(path)
    eps = sorted(int(k[len(PRE):]) for k in h
                 if k.startswith(PRE) and k[len(PRE):].isdigit())

    rows = []
    for ep in eps:
        cfgs = h[f'Sampled_Networks_Episode_{ep}']
        arrays = {n: _get(h, s, ep) for n, s in ARRAY_KEYS.items()}
        n = len(cfgs)
        for name, v in arrays.items():
            if v is not None:
                assert len(v) == n, f'episode {ep}: {name} has {len(v)}, {n} configs'

        for t, cfg in enumerate(cfgs):
            r = {'space': space, 'episode': ep, 'timestep': t + 1}
            for name, v in arrays.items():
                r[name] = float(v[t]) if v is not None else np.nan
            r['wv'] = -2.0 * (r['r1'] - r['r2']) if r['r1'] > r['r2'] else 0.0
            r['penalised'] = r['wv'] < 0
            r['score'] = r['r1'] + r['r2']
            r['score_r1wv'] = r['r1'] + r['wv']
            if (exclude_reward is not None 
                and (abs(r['score']- exclude_reward) < 1e-6 or 
                     abs(r['r1']- exclude_reward/2) < 1e-6 or
                     abs(r['r2']- exclude_reward/2) < 1e-6)):
                continue

            r.update(_flatten_config(cfg))
            rows.append(r)

    return pd.DataFrame(rows)


def summarise(df):
    """The checks worth running before anything is plotted."""
    for space, d in df.groupby('space'):
        print(f'\n=== {space}: {len(d)} architectures, '
              f'{d.episode.nunique()} episodes ===')

        dup = len(d) - d.config_hash.nunique()
        print(f'exact duplicate configs      : {dup}')

        if d.nr_nodes.notna().any():
            print('node-count share and score   :')
            print(d.groupby('nr_nodes').agg(
                n=('score', 'size'), share=('score', lambda s: len(s) / len(d)),
                mean=('score', 'mean'), best=('score', 'max')).round(3))

        print(f'penalised share, first 50 eps: '
              f'{d[d.episode <= d.episode.min() + 49].penalised.mean():.3f}')
        print(f'penalised share, last 50 eps : '
              f'{d[d.episode >= d.episode.max() - 49].penalised.mean():.3f}')

        print(f'mean GED, timestep 1         : {d[d.timestep == 1].ged.mean():.4f}')
        print(f'mean GED, timestep 10        : {d[d.timestep == 10].ged.mean():.4f}')

        # block frequency, early (near-uniform sampling) vs late
        early = d[d.episode <= d.episode.min() + 49]
        late = d[d.episode >= d.episode.max() - 49]
        freq = pd.DataFrame({
            'early': early.root_module.value_counts(normalize=True),
            'late': late.root_module.value_counts(normalize=True)}).fillna(0)
        print('root block frequency, early vs late:')
        print(freq.round(3))


def truncate(df, n_episodes=1000):
    """Keep the first n_episodes episodes, by episode number rather than row count."""
    eps = sorted(df.episode.unique())[:n_episodes]
    return df[df.episode.isin(eps)].copy()


if __name__ == '__main__':


    df_tree  = build_table(str(parent_dir)+'/Results_Evaluation/WV_GED_NAS_Sampling_History.pkl', 'tree')
    df_chain = build_table(str(parent_dir)+'/Results_Evaluation/WV_GED_NAS_Sampling_History_8_999_linear.pkl', 'chain', exclude_reward=6.0)
    df_tree.to_parquet('nas_tree_candidates.parquet')
    df_chain.to_parquet('nas_chain_candidates.parquet')

    summarise(pd.concat([truncate(df_tree), df_chain], ignore_index=True))

