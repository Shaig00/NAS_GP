"""Diebold-Mariano tests on the stored predictions, via epftoolbox.

H0: the errors of p_pred_2 are LARGER than p_pred_1's, so a small p-value says
p_pred_2 won. dm_matrix reports both directions, so it cannot be misread.

Storage layouts, matching the metric loaders:
    trees / chains / multivariate   Results[split] = [loss, true, pred]
    pre-trained                     Results[split] = [true, pred]
    AutoGluon      [key][seq_len]['Results'][split] = {'True_Values', 'Avg_Predictions'}
AutoGluon also stores arrays flattened to (n_origins*seq_len, 19); as_origins fixes that.

Sequences are cut with stride 1, so consecutive origins share all but one
forecast hour. Pass stride=seq_len to get non-overlapping origins, which is
what DM assumes. At 48h that is 7 origins per split, 105 in total.

Splits are renumbered by position. AutoGluon starts at 2 and ends short, so
`stack` intersects the split sets and prints what survived.

Assembly is explicit on purpose: see __main__.
"""
import pickle
import numpy as np
import pandas as pd

from epftoolbox.evaluation import DM



AG_KEY = 'NASTrain_Energy_Prices_Data'
MAX_SPLIT = 15
IDX = {'tree': (1, 2), 'chain': (1, 2), 'multivariate': (1, 2), 'pretrained': (0, 1)}




TREE_MEMBERS = list(range(10))
CHAIN_MEMBERS = {48: [1, 3, 4, 5, 6, 8], 96: [1, 3, 4, 5, 6, 8],
                 192: [1, 3, 4, 5, 6, 8]}
MULTIVARIATE = ['Basisformer', 'Dlinear', 'PatchTST', 'TSMixer', 'MTSMixer']
PRETRAINED = ['Sundial', 'GPT4TS', 'TimeMoe', 'TimesFM']
SINGLE_RUN = {'TimeMoe', 'TimesFM'}




def as_origins(x, seq_len):
    """-> (n_origins, seq_len, n_channels)."""
    a = (np.concatenate([np.asarray(b) for b in x], 0)
         if isinstance(x, (list, tuple)) else np.asarray(x))
    return a if a.ndim == 3 else a.reshape(-1, seq_len, a.shape[-1])


def load(path, layout, seq_len):
    """-> {split_number: (true, pred)}"""
    with open(path, 'rb') as f:
        r = pickle.load(f)
    i, j = IDX[layout]
    return {pos: (as_origins(entry[i], seq_len), as_origins(entry[j], seq_len))
            for pos, entry in enumerate(r['Results'].values(), 1) if pos <= MAX_SPLIT}


def load_autogluon(path, seq_len, start=2, key=AG_KEY):
    with open(path, 'rb') as f:
        r = pickle.load(f)
    return {pos: (as_origins(entry['True_Values'], seq_len), as_origins(entry['Avg_Predictions'], seq_len))
            for pos, entry in enumerate(r[key][seq_len]['Results'].values(), start)
            if pos <= MAX_SPLIT}


def ensemble(members):
    """[{split: (true, pred)}, ...] -> {split: (true, mean pred)}"""
    splits = set.intersection(*[set(m) for m in members])
    return {split: (members[0][split][0], np.mean([m[split][1] for m in members], 0))
            for split in splits}


def stack(models, stride):
    """{name: {split: (true, pred)}} -> {name: (true, pred)} on common splits."""
    splits = sorted(set.intersection(*[set(m) for m in models.values()]))
    out = {}
    ref = None
    for name, model in models.items():
        true = np.concatenate([model[split][0][::stride] for split in splits], 0)
        pred = np.concatenate([model[split][1][::stride] for split in splits], 0)
        if ref is None:
            ref = true
        assert np.allclose(true, ref, atol=1e-6), f'{name}: different targets'
        out[name] = (ref, pred)
    print(f'splits {splits[0]}-{splits[-1]} ({len(splits)}), {ref.shape[0]} origins')
    return out


def dm(true, other, mine, norm):
    """One multivariate p-value. Small => `mine` is better."""
    flat = lambda a: a.reshape(a.shape[0], -1)
    return float(DM(p_real=flat(true), p_pred_1=flat(other), p_pred_2=flat(mine),
                    norm=norm, version='multivariate'))


def dm_matrix(fc, focus, norm=2):
    rows = [{'vs': name,
             'p_focus_better': round(dm(fc[focus][0], other, fc[focus][1], norm), 4),
             'p_other_better': round(dm(fc[focus][0], fc[focus][1], other, norm), 4)}
            for name, (_, other) in fc.items() if name != focus]
    return pd.DataFrame(rows).sort_values('p_focus_better').reset_index(drop=True)

