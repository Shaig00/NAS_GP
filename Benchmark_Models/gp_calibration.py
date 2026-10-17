"""
Post-hoc calibration and evaluation for the GP head.

Lifted out of the benchmark runner so that the per-tree run, the ensemble run
and the ablations all call the same code.  Two calibration modes:

    calibrate_per_channel   the method: nu and t_scale fitted per market
    calibrate_shared        the ablation: one nu and one t_scale for all markets

Both return a dict with the same keys, so `evaluate_split` is indifferent to
which produced it: `nu` and `t_scale` are either [C] arrays or scalars, and
every multiplier is built with np.asarray so it broadcasts either way.
"""

import numpy as np
import torch
from scipy import stats

from Benchmark_Models.GP_Last_Layer_LMC import gp_predict_cached

LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
MAX_FIT_POINTS = 200_000          # subsample for the Student-t MLE


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def _subsampler(seed):
    rng = np.random.default_rng(seed)

    def _f(a):
        if a.size > MAX_FIT_POINTS:
            return a[rng.choice(a.size, MAX_FIT_POINTS, replace=False)]
        return a
    return _f


def _scale_and_standardise(gp, likelihood, pool_inputs, pool_residuals,
                           recent_inputs, recent_residuals, device):
    """Shared first stage: per-channel variance rescaling, then standardise."""
    _, var_recent = gp_predict_cached(gp, likelihood, recent_inputs, device)
    scale = (recent_residuals.var(dim=0)
             / var_recent.mean(dim=0).clamp_min(1e-12)).clamp_min(1e-6)   # [C]

    _, var_pool = gp_predict_cached(gp, likelihood, pool_inputs, device)
    sd_pool = torch.sqrt(var_pool * scale).clamp_min(1e-12)
    sd_recent = torch.sqrt(var_recent * scale).clamp_min(1e-12)

    return (scale,
            (pool_residuals / sd_pool).numpy(),        # [n_pool, C]
            (recent_residuals / sd_recent).numpy())    # [n_recent, C]


def calibrate_per_channel(gp, likelihood, pool_inputs, pool_residuals,
                          recent_inputs, recent_residuals, device, seed=0):
    """
    Three post-hoc corrections, each fitted on the data most appropriate to it.
    All of it precedes the evaluation block, so the procedure stays leak-safe.

    scale : per-channel variance rescaling, from the MOST RECENT banked block.
        The GP's predictive variance is near-constant within a channel, and its
        allocation ACROSS channels is poor enough that the raw variance ranks
        errors worse than a per-channel constant does.  Rescaling to match the
        empirical per-channel residual variance repairs that.  Recent data is
        used because relative channel volatility drifts.

    nu : Student-t tail index, PER CHANNEL, fitted on the WHOLE bank.  The tail
        index is a structural property of a market -- it is near-identical
        across architectures on the same block -- but it is NOT common across
        markets: fitted values range from about 1.3 in the less liquid
        Southeast European markets to about 21 in Iberia, ordered by market
        liquidity and interconnection.  A single shared value gives coverage
        that looks correct at the 90 per cent level while being wrong in
        opposite directions at 50 and 99 per cent.  Pooling over the bank rather
        than a single block is still necessary, because nu is estimated almost
        entirely from the tail and a single 17k-point block is too few.

    t_scale : interval width, PER CHANNEL, refitted on the MOST RECENT block
        with that channel's nu held fixed.  Pooling the width as well makes
        every block inherit the tail conditions of the worst block ever seen,
        which over-covers on calm blocks; the width is regime-dependent and
        should track recent data.  With nu fixed this is a one-parameter fit,
        so it is stable even on a single block.

    Six markets fit nu below 2, where the Student-t has no finite variance.  The
    lower clip is therefore 1.05 rather than 2, and intervals are built from
    quantiles rather than from any variance, which would be undefined there.
    """
    sub = _subsampler(seed)
    scale, u_pool, u_recent = _scale_and_standardise(
        gp, likelihood, pool_inputs, pool_residuals,
        recent_inputs, recent_residuals, device)
    C = u_pool.shape[1]

    nus = np.empty(C)
    t_scales = np.empty(C)
    kurt = np.empty(C)

    for i in range(C):
        s_i = sub(u_pool[:, i])
        nu_i, _, _ = stats.t.fit(s_i, floc=0.0)
        # Clip low enough to keep the genuinely infinite-variance markets: six
        # channels fit below nu = 2, and a floor at 2 would flatten them all
        # onto the same value and erase the difference between them.
        nu_i = float(np.clip(nu_i, 1.05, 200.0))
        kurt[i] = float(stats.kurtosis(s_i, fisher=True))

        # Width from the recent block with nu fixed: a one-parameter fit uses
        # the whole sample rather than only its tail, so it is precise even on
        # 17k points, whereas estimating nu jointly there was not.
        r_i = sub(u_recent[:, i])
        try:
            _, _, ts_i = stats.t.fit(r_i, fdf=nu_i, floc=0.0)
        except Exception:
            # older scipy without fdf: match the interquartile range, which is
            # defined for every nu, unlike the variance
            q75 = np.percentile(np.abs(r_i), 75)
            ts_i = float(q75 / max(stats.t.ppf(0.875, nu_i), 1e-6))

        nus[i], t_scales[i] = nu_i, float(ts_i)

    return {"scale": scale.numpy().astype("float32"),
            "nu": nus.astype("float32"),                 # [C]
            "t_scale": t_scales.astype("float32"),       # [C]
            "nu_pooled": float(np.median(nus)),          # reporting only
            "excess_kurtosis_pool": kurt.astype("float32"),
            "n_pool": int(u_pool.shape[0]),
            "n_recent": int(u_recent.shape[0]),
            "mode": "per_channel"}


def calibrate_shared(gp, likelihood, pool_inputs, pool_residuals,
                     recent_inputs, recent_residuals, device, seed=0):
    """
    ABLATION: one tail index and one width for every market.

    Kept so the contrast can be reported.  Pooled coverage under this variant
    looks correct at the 90 per cent level while the spread across channels
    reaches 0.21 at 50 per cent, and the correlation between a channel's tail
    index and its coverage error changes sign at almost exactly 90 per cent --
    which is why a single-level PICP cannot reveal the problem.
    """
    sub = _subsampler(seed)
    scale, u_pool, u_recent = _scale_and_standardise(
        gp, likelihood, pool_inputs, pool_residuals,
        recent_inputs, recent_residuals, device)

    s_pool = sub(u_pool.reshape(-1))
    nu, _, t_scale_pool = stats.t.fit(s_pool, floc=0.0)
    nu = float(np.clip(nu, 1.05, 200.0))

    s_recent = sub(u_recent.reshape(-1))
    try:
        _, _, t_scale = stats.t.fit(s_recent, fdf=nu, floc=0.0)
    except Exception:
        q75 = np.percentile(np.abs(s_recent), 75)
        t_scale = float(q75 / max(stats.t.ppf(0.875, nu), 1e-6))

    return {"scale": scale.numpy().astype("float32"),
            "nu": float(nu),
            "t_scale": float(t_scale),
            "t_scale_pooled": float(t_scale_pool),
            "excess_kurtosis_pool": float(stats.kurtosis(s_pool, fisher=True)),
            "excess_kurtosis_recent": float(stats.kurtosis(s_recent, fisher=True)),
            "n_pool": int(s_pool.size),
            "n_recent": int(s_recent.size),
            "mode": "shared"}


CALIBRATORS = {"per_channel": calibrate_per_channel, "shared": calibrate_shared}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def interval_multiplier(level, nu, t_scale):
    """
    Half-width multiplier on sigma for a central interval at `level`.

    nu and t_scale may be [C] arrays or scalars; np.asarray makes the result
    broadcast against sd of shape [N, T, C] either way.
    """
    nu = np.asarray(nu, dtype=np.float64)
    t_scale = np.asarray(t_scale, dtype=np.float64)
    return stats.t.ppf(0.5 + level / 2.0, nu) * t_scale


def picp_np(err, sd, mult):
    return float((np.abs(err) <= mult * sd).mean())


def picp_per_channel(err, sd, mult):
    inside = (np.abs(err) <= mult * sd)
    return inside.reshape(-1, inside.shape[-1]).mean(axis=0)


def pinball_np(err, sd, q, nu, t_scale):
    """err = y - centre.  Quantile of the predictive at level q."""
    nu = np.asarray(nu, dtype=np.float64)
    t_scale = np.asarray(t_scale, dtype=np.float64)
    qp = stats.t.ppf(q, nu) * t_scale * sd
    d = err - qp
    return float(np.maximum(q * d, (q - 1.0) * d).mean())


def nll_student_np(err, sd, nu, t_scale):
    nu = np.asarray(nu, dtype=np.float64)
    t_scale = np.asarray(t_scale, dtype=np.float64)
    z = err / np.maximum(sd, 1e-12)
    return float(-np.mean(stats.t.logpdf(z, nu, loc=0.0, scale=t_scale)
                          - np.log(np.maximum(sd, 1e-12))))


def evaluate_split(z, y, mu, var, calib):
    """
    z   : point forecast             [N, T, C]   (one tree, or the ensemble)
    y   : truth                      [N, T, C]
    mu  : GP residual mean           [N, T, C]
    var : GP predictive variance     [N, T, C]

    Coverage is reported both pooled and per channel.  A shared tail index
    produced a pooled PICP that looked correct at 90 per cent while the spread
    across channels reached 0.21 at 50 per cent, so the pooled figure alone
    cannot show whether the intervals are right.
    """
    nu, t_scale = calib["nu"], calib["t_scale"]
    sd_raw = np.sqrt(var)
    sd = np.sqrt(var * calib["scale"])

    err0 = y - z                      # deployed model, lambda = 0
    err1 = y - (z + mu)               # lambda = 1, reported as an ablation

    mult = {c: interval_multiplier(c, nu, t_scale) for c in LEVELS}
    cov_chan = {c: picp_per_channel(err0, sd, mult[c]) for c in LEVELS}

    mpiw90 = 2.0 * float(np.mean(mult[0.9]
                                 * sd.reshape(-1, sd.shape[-1]).mean(axis=0)))

    return {
        # --- the deployed intervals ---------------------------------------
        "picp_t": {c: picp_np(err0, sd, mult[c]) for c in LEVELS},
        "picp_t_per_channel": {c: cov_chan[c].astype("float32") for c in LEVELS},
        "picp_spread": {c: float(cov_chan[c].max() - cov_chan[c].min())
                        for c in LEVELS},
        "picp_mae": {c: float(np.abs(cov_chan[c] - c).mean()) for c in LEVELS},

        # --- ablations ----------------------------------------------------
        "picp_gauss": {c: picp_np(err0, sd, stats.norm.ppf(0.5 + c / 2))
                       for c in LEVELS},
        "picp_gauss_raw": {c: picp_np(err0, sd_raw, stats.norm.ppf(0.5 + c / 2))
                           for c in LEVELS},
        "picp_lam1": {c: picp_np(err1, sd, mult[c]) for c in LEVELS},

        # --- proper scoring rules ----------------------------------------
        "pinball": {q: pinball_np(err0, sd, q, nu, t_scale) for q in QUANTILES},
        "nll_t": nll_student_np(err0, sd, nu, t_scale),

        # --- point and sharpness ------------------------------------------
        "rmse_point": float(np.sqrt((err0 ** 2).mean())),
        "rmse_lam1": float(np.sqrt((err1 ** 2).mean())),
        "sharpness_mpiw_90": mpiw90,
        "spearman_var_err": float(stats.spearmanr(
            sd.ravel(), np.abs(err0).ravel()).statistic),

        # --- the fitted tail indices, for the results table ---------------
        "nu_per_channel": np.atleast_1d(np.asarray(nu, dtype="float32")),
        "nu_median": float(np.median(np.atleast_1d(nu))),
    }