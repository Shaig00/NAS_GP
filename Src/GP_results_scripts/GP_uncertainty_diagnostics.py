"""
Is the GP's variance USEFUL, and does the Student-t correction fit?

Two questions, deliberately separate:

  1. Usefulness.  Does the predictive variance pick out the forecasts that
     turn out to be wrong?  Coverage can be perfect while the variance carries
     no information about WHICH hours are hard -- the intervals would then be
     the right width on average and the wrong width everywhere in particular.

  2. Shape.  Are the standardised residuals Student-t, as the deployed
     calibration assumes, and does the Gaussian alternative fail?

Reads the output of gp_runner.py, which stores z, y, mu and var per split.
Everything is evaluated on the DEPLOYED quantities: the error against the point
forecast (lambda = 0) and the rescaled standard deviation sqrt(var * scale),
not the GP's raw variance.  That matters -- before rescaling, the raw variance
ranks errors slightly WORSE than a per-channel constant does.

    python gp_uncertainty_diagnostics.py
"""

import pickle
import re
from pathlib import Path

import numpy as np
from scipy import stats

PARENT = Path.cwd().parent
RESULTS_DIR = PARENT / "Results_Evaluation"

TAG = "per_channel"
NAME = "ensemble"              # or "tree_0", ...
SEQ_LEN = 48
RUN = 1

NOMINAL = (0.5, 0.8, 0.9, 0.95, 0.99)
REMOVED_FRACTIONS = np.linspace(0.0, 0.9, 19)

trapezoid = getattr(np, "trapezoid", np.trapz)      # renamed in NumPy 2.0


# =============================================================================
# loading
# =============================================================================

def load_splits(tag, name, seq_len, run):
    """-> [(split_key, z, y, mu, var, calib)] in chronological order."""
    path = (RESULTS_DIR /
            f"GP_{tag}_{name}_EU_Energy_Prices_Time_Series_{seq_len}_{run}.pkl")
    with open(path, "rb") as f:
        results = pickle.load(f)["GP_Results"]

    keys = sorted(results, key=lambda s: int(re.search(r"\d+", s).group()))
    out = []
    for k in keys:
        entry = results[k]
        if "arrays" not in entry:
            raise KeyError(f"{k} has no stored arrays -- rerun gp_runner.py "
                           f"with store_arrays=True")
        a = entry["arrays"]
        out.append((k, a["z"], a["y"], a["mu"], a["var"], entry["calibration"]))
    return out


def deployed(z, y, var, calib):
    """
    The quantities the forecast actually uses.

    error : y - z, since the point forecast is the model's own (lambda = 0)
    sd    : sqrt(var * scale), the per-channel rescaling from calibration
    """
    return y - z, np.sqrt(np.maximum(var * calib["scale"], 1e-12))


# =============================================================================
# 1. is the variance useful?
# =============================================================================

def sparsification(error, sd, fractions=REMOVED_FRACTIONS):
    """
    Remove the fraction of points the model is LEAST confident about, then
    recompute RMSE on what remains.

        model   ordered by predicted sd      what can actually be done
        oracle  ordered by realised error    the best any ordering could do
        random  ordered arbitrarily          the no-information baseline

    A variance carrying information sits between oracle and random.  One lying
    on top of random is calibrated but uninformative.

    AUSE is the area between model and oracle over the area between random and
    oracle: 0 matches the oracle, 1 is no better than random.
    """
    squared = error.ravel() ** 2
    predicted = sd.ravel()
    n = squared.size

    orders = {"model": np.argsort(-predicted),
              "oracle": np.argsort(-squared),
              "random": np.random.default_rng(0).permutation(n)}

    curves = {}
    for label, order in orders.items():
        reordered = squared[order]
        curves[label] = np.array([
            np.sqrt(reordered[int(f * n):].mean()) for f in fractions])

    above_oracle = trapezoid(curves["model"] - curves["oracle"], fractions)
    random_gap = trapezoid(curves["random"] - curves["oracle"], fractions)
    ause = float(above_oracle / random_gap) if random_gap > 0 else np.nan

    return curves, ause


def association(error, sd):
    """Rank and linear association between predicted sd and realised error."""
    absolute = np.abs(error).ravel()
    predicted = sd.ravel()
    return (float(stats.spearmanr(predicted, absolute).statistic),
            float(stats.pearsonr(predicted, absolute).statistic))


def error_by_decile(error, sd):
    """Mean absolute error within each decile of predicted sd."""
    absolute = np.abs(error).ravel()
    predicted = sd.ravel()
    edges = np.quantile(predicted, np.linspace(0, 1, 11))

    means = []
    for i in range(10):
        upper = predicted <= edges[i + 1] if i == 9 else predicted < edges[i + 1]
        mask = (predicted >= edges[i]) & upper
        means.append(float(absolute[mask].mean()) if mask.any() else np.nan)
    return np.array(means)


# =============================================================================
# 2. does the Student-t fit?
# =============================================================================

def standardise(error, sd):
    return (error / sd).ravel()


def tail_summary(standardised):
    """Moments, a refitted Student-t, and KS against both candidates."""
    nu, _, scale = stats.t.fit(standardised, floc=0.0)
    return {
        "excess_kurtosis": float(stats.kurtosis(standardised, fisher=True)),
        "skew": float(stats.skew(standardised)),
        "nu": float(nu),
        "t_scale": float(scale),
        "ks_normal": float(stats.kstest(
            standardised, "norm", args=(0.0, standardised.std())).statistic),
        "ks_student_t": float(stats.kstest(
            standardised, "t", args=(nu, 0.0, scale)).statistic),
    }


def pit_histogram(error, sd, calib=None, bins=10):
    """
    Probability integral transform.  Uniform means the predictive distribution
    is right; a U shape means it is too narrow, a hump means too wide.

    With `calib`, the deployed per-channel Student-t is used; without it, a
    standard Gaussian, which is the ablation.
    """
    if calib is None:
        u = stats.norm.cdf(error / sd)
    else:
        nu = np.asarray(calib["nu"], dtype=float)
        t_scale = np.asarray(calib["t_scale"], dtype=float)
        u = stats.t.cdf(error / (sd * t_scale), nu)      # broadcasts over [C]

    counts, _ = np.histogram(u.ravel(), bins=bins, range=(0, 1))
    return counts / counts.sum()


# =============================================================================
# reporting
# =============================================================================

def report_usefulness(per_split, curves, ause, deciles):
    print("=" * 68)
    print("1. IS THE VARIANCE USEFUL?")
    print("=" * 68)
    print(f"{'split':>16}{'AUSE':>9}{'spearman':>11}{'pearson':>10}")
    for row in per_split:
        print(f"{row['split']:>16}{row['ause']:>9.3f}"
              f"{row['spearman']:>11.3f}{row['pearson']:>10.3f}")
    print(f"{'mean':>16}"
          f"{np.mean([r['ause'] for r in per_split]):>9.3f}"
          f"{np.mean([r['spearman'] for r in per_split]):>11.3f}"
          f"{np.mean([r['pearson'] for r in per_split]):>10.3f}")

    print(f"\npooled AUSE = {ause:.3f}   (0 matches the oracle, "
          f"1 is no better than random)")

    print("\nRMSE after removing the most uncertain x per cent:")
    print(f"{'removed':>9}{'model':>9}{'oracle':>9}{'random':>9}")
    for i in range(0, len(REMOVED_FRACTIONS), 3):
        print(f"{REMOVED_FRACTIONS[i]:>8.0%} "
              f"{curves['model'][i]:>9.4f}{curves['oracle'][i]:>9.4f}"
              f"{curves['random'][i]:>9.4f}")

    print("\nmean absolute error by predicted-sd decile (low to high):")
    print("  " + "  ".join(f"{d:.3f}" for d in deciles))
    print(f"  top / bottom decile = {deciles[-1] / deciles[0]:.2f}"
          f"   (1.0 means the variance separates nothing)")


def report_shape(summary, pit_t, pit_gauss, deployed_nu):
    print("\n" + "=" * 68)
    print("2. DOES THE STUDENT-t FIT?")
    print("=" * 68)
    print(f"  excess kurtosis   {summary['excess_kurtosis']:8.3f}   (Gaussian = 0)")
    print(f"  skew              {summary['skew']:8.3f}")
    print(f"  refitted nu       {summary['nu']:8.2f}   pooled over channels")
    print(f"  deployed nu       {np.min(deployed_nu):8.2f} to "
          f"{np.max(deployed_nu):.2f}   per channel")
    print(f"  KS vs normal      {summary['ks_normal']:8.4f}")
    print(f"  KS vs Student-t   {summary['ks_student_t']:8.4f}")

    print("\nPIT histogram, 10 bins, uniform = 0.100 each")
    print("  deployed t:  " + "  ".join(f"{h:.3f}" for h in pit_t))
    print("  Gaussian  :  " + "  ".join(f"{h:.3f}" for h in pit_gauss))
    print("\n  A U shape means the distribution is too narrow, a hump too wide.")
    print(f"  max deviation from uniform: t {np.abs(pit_t - 0.1).max():.3f}, "
          f"Gaussian {np.abs(pit_gauss - 0.1).max():.3f}")


# =============================================================================
def main():
    splits = load_splits(TAG, NAME, SEQ_LEN, RUN)
    print(f"{NAME}, seq_len {SEQ_LEN}, run {RUN}: {len(splits)} splits\n")

    per_split, errors, sds = [], [], []
    pit_t_parts, pit_gauss_parts = [], []

    for key, z, y, mu, var, calib in splits:
        error, sd = deployed(z, y, var, calib)
        _, ause = sparsification(error, sd)
        rho_rank, rho_linear = association(error, sd)

        per_split.append({"split": key, "ause": ause,
                          "spearman": rho_rank, "pearson": rho_linear})
        errors.append(error)
        sds.append(sd)
        pit_t_parts.append(pit_histogram(error, sd, calib))
        pit_gauss_parts.append(pit_histogram(error, sd, None))

    error = np.concatenate(errors)
    sd = np.concatenate(sds)

    curves, ause = sparsification(error, sd)
    deciles = error_by_decile(error, sd)
    report_usefulness(per_split, curves, ause, deciles)

    summary = tail_summary(standardise(error, sd))
    report_shape(summary,
                 np.mean(pit_t_parts, axis=0),
                 np.mean(pit_gauss_parts, axis=0),
                 np.atleast_1d(splits[-1][5]["nu"]))

    out = {"per_split": per_split, "ause": ause, "curves": curves,
           "fractions": REMOVED_FRACTIONS, "deciles": deciles,
           "tail": summary,
           "pit_t": np.mean(pit_t_parts, axis=0),
           "pit_gauss": np.mean(pit_gauss_parts, axis=0)}
    path = Path(f"gp_uncertainty_diagnostics_{NAME}_{SEQ_LEN}_{RUN}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()