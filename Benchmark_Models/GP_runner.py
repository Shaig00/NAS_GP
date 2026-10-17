"""
GP calibration over STORED point forecasts.  No backbone training.

Why this replaces the training loop in the benchmark.  The GP conditions on
[z_hat, h] -- the forecast and the horizon position -- not on any hidden
representation of the network (see `z = eval_inputs[:, :C]` in the original
benchmark).  The forecasts are already on disk from the point-forecast run, so
running a backbone again only reproduces them, at the cost of hours and of a
slight mismatch: the z the GP would see comes from a separately retrained model
rather than from the forecasts reported in the results tables.

One runner covers both cases through `members`:

    members = [3]              one tree, per-tree result
    members = list(range(10))  the mean of ten, the ensemble

The ensemble is formed WITHIN a run, so each member is exactly ten trained
models and the run-to-run spread survives, matching ensemble_table() in
point_metrics.py.

Leak safety per split, unchanged: the GP is fitted on blocks strictly before
the validation block, calibrated on the validation block, and evaluated on the
current block, which is strictly after both.

    python gp_runner.py
"""
import sys
sys.path.append("..")
import pickle
import re
import time
from pathlib import Path

import numpy as np
import torch

from Benchmark_Models.GP_Last_Layer_LMC import (
    fit_gp_on_cache, gp_predict_cached, extract_task_covariance,
)
from gp_calibration import CALIBRATORS, evaluate_split

parent_dir = Path.cwd().parent

N_TREES = 10
RESULTS = f'/home/RDC/khankiss/H/Results_Point_Forecasts_new'


# ---------------------------------------------------------------------------
# loading stored point forecasts
# ---------------------------------------------------------------------------
def _stack(batches):
    """Batch list -> one array.  The last batch is short, so this is not reshape."""
    if isinstance(batches, (list, tuple)):
        return np.concatenate([np.asarray(b) for b in batches], axis=0)
    return np.asarray(batches)


def load_run(tree_idx, seq_len, run):
    path = (f'{RESULTS}/tree_{tree_idx}_EU_Energy_Prices_Time_Series_'
            f'{seq_len}_{run}.pkl')
    with open(path, 'rb') as f:
        r = pickle.load(f)
    return {k: (_stack(v[1]), _stack(v[2])) for k, v in r['Results'].items()}


def load_members(members, seq_len, run):
    """
    -> {subset: (true, z)} in chronological order, z averaged over `members`.

    With more than one member the ten `true` arrays are checked for agreement.
    If they disagree the dataloaders were not aligned and averaging the
    predictions would be meaningless.
    """
    loaded = [load_run(t, seq_len, run) for t in members]
    subsets = sorted(loaded[0], key=lambda s: int(re.search(r'\d+', s).group()))

    out = {}
    for s in subsets:
        true = loaded[0][s][0]
        for i, m in enumerate(loaded[1:], start=1):
            if not np.allclose(true, m[s][0], atol=1e-5):
                raise ValueError(
                    f'member {members[i]} disagrees with member {members[0]} '
                    f'on `true` at {s} (seq_len {seq_len}, run {run})')
        out[s] = (true, np.mean([m[s][1] for m in loaded], axis=0))
    return out


# ---------------------------------------------------------------------------
# GP cache, reproducing build_gp_cache's layout
# ---------------------------------------------------------------------------
def make_cache(true, z, add_horizon=True):
    """
    true, z : [N, T, C]  ->  inputs [N*T, C(+1)], residuals [N*T, C], T

    Layout must match build_gp_cache: forecast in the first C columns, horizon
    position last, flattening row-major over (N, T) so that
    inputs[:, :C].reshape(-1, T, C) recovers z.
    """
    N, T, C = z.shape
    inputs = z.reshape(N * T, C).astype('float32')
    if add_horizon:
        h = np.arange(T, dtype='float32') / max(T - 1, 1)      # (t-1)/(T-1)
        inputs = np.concatenate([inputs, np.tile(h, N)[:, None]], axis=1)
    residuals = (true - z).reshape(N * T, C).astype('float32')

    # cheap round trip; catches a transposed flatten immediately
    assert np.allclose(inputs[:, :C].reshape(N, T, C), z, atol=1e-5)
    return torch.from_numpy(inputs), torch.from_numpy(residuals), T


# ---------------------------------------------------------------------------
# the walk-forward loop, without any backbone
# ---------------------------------------------------------------------------
def run_one(members, seq_len, run, device='cuda', calib_mode='per_channel',
            structure='icm', num_latents=1, num_inducing=200, gp_epochs=30,
            patience=8, add_horizon=True, store_arrays=True, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    calibrate = CALIBRATORS[calib_mode]

    data = load_members(members, seq_len, run)
    C = next(iter(data.values()))[0].shape[-1]
    tag = 'ensemble' if len(members) > 1 else f'tree {members[0]}'

    results, bank = {}, []
    t0, gp_time = time.time(), 0.0

    for subset, (true, z) in data.items():
        eval_inputs, eval_residuals, T = make_cache(true, z, add_horizon)

        if len(bank) >= 2:
            val_inputs, val_residuals = bank[-1]
            fit_inputs = torch.cat([b[0] for b in bank[:-1]], 0)
            fit_residuals = torch.cat([b[1] for b in bank[:-1]], 0)

            t_gp = time.time()
            gp, likelihood, info = fit_gp_on_cache(
                fit_inputs, fit_residuals, val_inputs, val_residuals,
                n_channels=C, structure=structure, num_latents=num_latents,
                num_inducing=num_inducing, n_epochs=gp_epochs, device=device,
                patience=patience, seed=seed)

            calib = calibrate(gp, likelihood,
                              torch.cat([b[0] for b in bank], 0),
                              torch.cat([b[1] for b in bank], 0),
                              val_inputs, val_residuals, device, seed=seed)
            gp_time += time.time() - t_gp

            mu, var = gp_predict_cached(gp, likelihood, eval_inputs, device)
            mu = mu.reshape(-1, T, C).numpy().astype('float32')
            var = var.reshape(-1, T, C).numpy().astype('float32')

            metrics = evaluate_split(z, true, mu, var, calib)
            B = extract_task_covariance(gp, likelihood)

            entry = {'metrics': metrics, 'calibration': calib,
                     'lambda_val': info['lambda'], 'best_epoch': info['best_epoch'],
                     'n_fit_points': int(fit_inputs.shape[0]),
                     'task_cov': {'B': B['B'], 'task_noise': B['task_noise'],
                                  'trace_signal': B['trace_signal'],
                                  'trace_noise': B['trace_noise']}}
            if store_arrays:
                entry['arrays'] = {'z': z.astype('float32'),
                                   'y': true.astype('float32'),
                                   'mu': mu, 'var': var}
            results[subset] = entry

            nu = np.atleast_1d(calib['nu'])
            print(f"[{tag} run {run}] {subset}: "
                  f"RMSE {metrics['rmse_point']:.4f} | "
                  f"PICP90 {metrics['picp_t'][0.9]:.3f} "
                  f"(spread {metrics['picp_spread'][0.9]:.3f}) | "
                  f"nu {nu.min():.2f}-{nu.max():.2f} | "
                  f"rho {metrics['spearman_var_err']:+.3f}", flush=True)

        # bank AFTER evaluating, BEFORE the window rolls over it
        bank.append((eval_inputs, eval_residuals))

    return {'GP_Results': results, 'Total_Time': time.time() - t0,
            'GP_Head_Time': gp_time,
            'Axes': {'members': list(members), 'run': run,
                     'calib_mode': calib_mode, 'structure': structure,
                     'num_latents': num_latents, 'add_horizon': add_horizon,
                     'gp_fit_mode': 'bank',
                     'source': 'stored_point_forecasts'}}


# ---------------------------------------------------------------------------
def run_many(timesteps=(96,), runs=(1,), per_tree=True, ensemble=True,
             calib_mode='per_channel', tag=None, **kw):
    """
    per_tree : one GP per architecture -- the robustness evidence that the
        calibration holds across architectures rather than on one lucky tree
    ensemble : one GP on the mean of the ten -- the headline, and the only
        configuration where the predictive is centred on the point forecast
        that is actually reported
    """
    tag = tag or calib_mode
    jobs = []
    if per_tree:
        jobs += [[t] for t in range(N_TREES)]
    if ensemble:
        jobs += [list(range(N_TREES))]

    for ts in timesteps:
        for run in runs:
            for members in jobs:
                res = run_one(members, ts, run, calib_mode=calib_mode, **kw)
                name = 'ensemble' if len(members) > 1 else f'tree_{members[0]}'
                out = (f'{RESULTS}/GP_{tag}_{name}_'
                       f'EU_Energy_Prices_Time_Series_{ts}_{run}.pkl')
                with open(out, 'wb') as f:
                    pickle.dump(res, f)
                print(f'saved -> {out}\n')


if __name__ == '__main__':

    # the method, everything
    run_many(timesteps=(48, 96, 192), runs=(1,), per_tree=False, ensemble=True)
    # run_many(timesteps=(48, 96, 192), runs=(1,), per_tree=True, ensemble=False)

    # the ablation: one tail index for every market
    # run_many(timesteps=(48,96,192), runs=(1,), per_tree=False, calib_mode='shared')
    pass