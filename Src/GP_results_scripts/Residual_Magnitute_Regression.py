"""
Is the residual MAGNITUDE predictable?

The GP's predictive variance is effectively constant within a channel, so it
cannot say which hours are hard.  Before concluding that a different GP would
fix that, the prior question has to be asked: given some set of features, can
log(residual^2) be predicted out of sample at all?

Four nested feature sets, fitted on earlier blocks, scored on later ones:

    M0   channel identity only          what the GP effectively does now
    M1   M0 + forecast + horizon        what the GP is actually given
    M2   M1 + hour, volatility, spread  price features it has never seen
    M3   M2 + day-ahead load            information outside the price series

Reads stored point forecasts; nothing is trained.  Set MEMBERS to one tree for
a single architecture, or to all ten for the ensemble.

    python residual_magnitude_regression.py
"""

import pickle
import re
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# settings
# =============================================================================

SEQ_LEN = 48
MEMBERS = range(10)        # [0] for one tree, range(10) for the ensemble
RUN = 1
N_SPLITS = 12              # fewer than the full walk is plenty here
N_EVAL_BLOCKS = 3         # last k blocks held out for scoring
MAX_ROWS_PER_BLOCK = 40_000
RIDGE_ALPHAS = (1.0, 10.0, 100.0)

PARENT = Path.cwd().parent
RESULTS_DIR = PARENT / "Results_Evaluation"
DATA_DIR = PARENT / "Energy_Production_Data" / "EU_Time_Series"

# rows of the rolling train window and of each test block, per setting.
# Hardcoding one horizon's values silently misaligns the others.
TRAIN_ROWS = {48: 1904, 96: 2000, 192: 2192}
TEST_ROWS = {48: 404, 96: 500, 192: 692}
SPLIT_INDICES = {48: list(range(4, 21)),
                 96: list(range(3, 20)),
                 192: list(range(3, 18))}

FEATURE_LABELS = {"M0": "channel only",
                  "M1": "+ forecast, horizon (the GP's input)",
                  "M2": "+ hour, volatility, spread",
                  "M3": "+ day-ahead load"}


# =============================================================================
# loading
# =============================================================================

def concat_batches(x):
    """Stored arrays are lists of per-batch chunks; the last one is short."""
    if isinstance(x, (list, tuple)):
        return np.concatenate([np.asarray(b) for b in x], axis=0)
    return np.asarray(x)


def read_tree(tree_idx, seq_len, run):
    path = (RESULTS_DIR /
            f"tree_{tree_idx}_EU_Energy_Prices_Time_Series_{seq_len}_{run}.pkl")
    with open(path, "rb") as f:
        stored = pickle.load(f)["Results"]
    return {k: (concat_batches(v[1]), concat_batches(v[2]))
            for k, v in stored.items()}


def read_forecasts(members, seq_len, run):
    """
    -> {split_index: (truth, forecast)}, the forecast averaged over members.

    The stored keys are a renumbering: the benchmark writes Train_Test{k} for
    the k-th evaluated block, not for block k.  They are mapped back onto the
    real split indices here, because every load and hour feature is positioned
    from the split index.
    """
    trees = [read_tree(t, seq_len, run) for t in members]
    keys = sorted(trees[0], key=lambda s: int(re.search(r"\d+", s).group()))
    splits = SPLIT_INDICES[seq_len]

    if len(keys) != len(splits):
        raise RuntimeError(
            f"{len(keys)} stored blocks against {len(splits)} split indices "
            f"for seq_len {seq_len}")

    forecasts = {}
    for key, split in zip(keys, splits):
        truth = trees[0][key][0]
        for i, other in enumerate(trees[1:], start=1):
            if not np.allclose(truth, other[key][0], atol=1e-5):
                raise ValueError(
                    f"member {members[i]} disagrees with member {members[0]} "
                    f"on the truth at {key}")
        forecasts[split] = (truth,
                            np.mean([t[key][1] for t in trees], axis=0))
    return forecasts


def read_frames(seq_len):
    with open(DATA_DIR / f"EU_Energy_Prices_Time_Series_{seq_len}_Train.pkl", "rb") as f:
        prices = pickle.load(f)
    with open(DATA_DIR / f"EU_Total_Load_Time_Series_{seq_len}_Train.pkl", "rb") as f:
        load = pickle.load(f)
    return prices, load


# =============================================================================
# exogenous load
# =============================================================================

def country_of(column):
    return column.split(" ")[0].lower()


def align_load(price_frame, load_frame, scaler_rows):
    """
    Put the load frame in the price frame's channel order and standardise it.

    Countries absent from the load frame are left at zero, which after the
    regression's own standardisation is the column mean and so contributes
    nothing.  Slovakia is the expected case: its day-ahead series ends
    2024-06-30.

    Standardisation uses only the first `scaler_rows` hours, which precede
    every test block.  Channels must be on comparable units because the
    regression pools rows across channels.
    """
    price_countries = [country_of(c) for c in price_frame.columns
                       if c not in ("Subset", "date")]
    load_columns = {country_of(c): c for c in load_frame.columns
                    if c not in ("Subset", "date")}

    aligned = np.zeros((len(price_frame), len(price_countries)), dtype=np.float32)
    missing = []

    for j, country in enumerate(price_countries):
        if country not in load_columns:
            missing.append(country)
            continue
        series = load_frame[load_columns[country]].to_numpy(np.float32)
        mean = series[:scaler_rows].mean()
        sd = max(series[:scaler_rows].std(), 1e-6)
        aligned[:, j] = (series - mean) / sd

    print(f"[load] {len(price_countries) - len(missing)} of "
          f"{len(price_countries)} channels have load; "
          f"zero-filled: {missing or 'none'}")
    return np.nan_to_num(aligned, nan=0.0)


def trailing_mean(series, window=24):
    """Mean over the last `window` rows inclusive, expanding at the start."""
    padded = np.vstack([np.zeros((1, series.shape[1]), np.float32), series])
    cumulative = np.cumsum(padded, axis=0)

    out = np.empty_like(series)
    for i in range(series.shape[0]):
        lo = max(0, i - window + 1)
        out[i] = (cumulative[i + 1] - cumulative[lo]) / (i + 1 - lo)
    return out


def load_at_targets(load, target_rows):
    """
    Load features at the forecast target hours, each [N, T, C].

    All three are known when the forecast is made: the day-ahead publication
    covers the whole horizon.
    """
    previous = np.roll(load, 1, axis=0)
    previous[0] = load[0]
    baseline = trailing_mean(load)

    level = load[target_rows]
    ramp = level - previous[target_rows]
    deviation = level - baseline[target_rows]
    return level, ramp, deviation


# =============================================================================
# features
# =============================================================================

def block_start_row(split, seq_len):
    """First row of the test block for this split, in the full frame."""
    return TRAIN_ROWS[seq_len] + TEST_ROWS[seq_len] * (split - 1)


def target_rows(split, seq_len, n_windows, horizon):
    """
    Absolute row of each (window, horizon step) pair.

    Window w of this split starts at block_start + w and its input covers
    seq_len rows, so horizon step t targets block_start + w + seq_len + t.
    """
    start = block_start_row(split, seq_len)
    return (start
            + np.arange(n_windows)[:, None]
            + seq_len
            + np.arange(horizon)[None, :])


def rebuild_input_windows(prices, split, seq_len, n_windows):
    """
    The scaled input window behind each forecast.

    Scaling reproduces the pipeline's: per-channel statistics from this split's
    own training window, the `train_rows` rows immediately before the block.
    """
    start = block_start_row(split, seq_len)
    train = prices[start - TRAIN_ROWS[seq_len]:start]
    mean = train.mean(axis=0)
    sd = np.maximum(train.std(axis=0), 1e-6)

    rows = (start + np.arange(n_windows)[:, None] + np.arange(seq_len)[None, :])
    return ((prices[rows] - mean) / sd).astype(np.float32)


def build_features(prices, load, truth, forecast, split, seq_len, rng):
    """One row per (window, horizon step, channel)."""
    residual = truth - forecast
    n_windows, horizon, n_channels = forecast.shape
    shape = (n_windows, horizon, n_channels)

    rows = target_rows(split, seq_len, n_windows, horizon)
    if rows.max() >= load.shape[0]:
        raise IndexError(
            f"split {split}: target row {rows.max()} beyond the frame "
            f"({load.shape[0]} rows)")

    windows = rebuild_input_windows(prices, split, seq_len, n_windows)

    window_idx = np.arange(n_windows)[:, None, None]
    step_idx = np.arange(horizon)[None, :, None]
    spread = lambda a: np.broadcast_to(a, shape)

    # --- time position ------------------------------------------------------
    horizon_pos = spread(step_idx / max(horizon - 1, 1))
    clock_hour = (window_idx + step_idx) % 24          # up to a phase offset
    hour_sin = spread(np.sin(2 * np.pi * clock_hour / 24))
    hour_cos = spread(np.cos(2 * np.pi * clock_hour / 24))

    # --- dispersion ---------------------------------------------------------
    own_volatility = spread(windows.std(axis=1)[:, None, :])
    mean_volatility = spread(windows.std(axis=1).mean(1)[:, None, None])
    cross_channel_spread = spread(forecast.std(axis=2)[:, :, None])

    # --- identity and forecast ----------------------------------------------
    forecast_vector = np.broadcast_to(forecast[:, :, None, :],
                                      shape + (n_channels,))
    channel_onehot = np.zeros(shape + (n_channels,), dtype=np.float32)
    channel_onehot[..., np.arange(n_channels), np.arange(n_channels)] = 1.0

    # --- exogenous ----------------------------------------------------------
    load_level, load_ramp, load_deviation = load_at_targets(load, rows)

    n_rows = n_windows * horizon * n_channels
    flat = lambda a: np.asarray(a, np.float32).reshape(n_rows, -1)

    m0 = flat(channel_onehot)
    m1 = np.hstack([m0, flat(forecast_vector), flat(horizon_pos)])
    m2 = np.hstack([m1,
                    flat(hour_sin), flat(hour_cos),
                    flat(own_volatility), flat(mean_volatility),
                    flat(cross_channel_spread)])
    m3 = np.hstack([m2,
                    flat(load_level), flat(load_ramp), flat(load_deviation)])

    target = np.log(flat(residual).ravel() ** 2 + 1e-6)
    abs_residual = np.abs(flat(residual).ravel())

    designs = {"M0": m0, "M1": m1, "M2": m2, "M3": m3}
    if n_rows > MAX_ROWS_PER_BLOCK:
        keep = rng.choice(n_rows, MAX_ROWS_PER_BLOCK, replace=False)
        designs = {k: v[keep] for k, v in designs.items()}
        target, abs_residual = target[keep], abs_residual[keep]

    return designs, target, abs_residual


# =============================================================================
# fitting
# =============================================================================

def choose_alpha(x_train, y_train):
    """Pick the ridge penalty on a held-out tail of the training rows."""
    cut = int(0.8 * len(y_train))
    scores = [(Ridge(alpha=a).fit(x_train[:cut], y_train[:cut])
               .score(x_train[cut:], y_train[cut:]), a)
              for a in RIDGE_ALPHAS]
    return max(scores)[1]


def fit_and_score(x_train, y_train, x_eval, y_eval, abs_residual_eval):
    scaler = StandardScaler().fit(x_train)
    x_train, x_eval = scaler.transform(x_train), scaler.transform(x_eval)

    alpha = choose_alpha(x_train, y_train)
    predicted = Ridge(alpha=alpha).fit(x_train, y_train).predict(x_eval)

    r2 = 1.0 - np.sum((y_eval - predicted) ** 2) / np.sum((y_eval - y_train.mean()) ** 2)
    implied_sd = np.exp(predicted / 2.0)

    return {"r2": float(r2),
            "alpha": alpha,
            "spearman": float(stats.spearmanr(implied_sd, abs_residual_eval).statistic),
            "sd_cv": float(implied_sd.std() / implied_sd.mean())}


# =============================================================================
# reporting
# =============================================================================

def print_results(results):
    print("\n" + "=" * 72)
    print("OUT-OF-SAMPLE PREDICTABILITY OF log(residual^2)")
    print("=" * 72)
    print(f"{'model':<6}{'features':>9}{'R2':>9}{'spearman':>10}{'sd CV':>8}   label")

    for name, res in results.items():
        print(f"{name:<6}{res['n_features']:>9}{res['r2']:>9.4f}"
              f"{res['spearman']:>10.3f}{res['sd_cv']:>8.3f}   "
              f"{FEATURE_LABELS[name]}")

    print("-" * 72)
    names = list(results)
    for earlier, later in zip(names, names[1:]):
        gain = results[later]["r2"] - results[earlier]["r2"]
        print(f"gain {later} - {earlier}: {gain:+.4f}   "
              f"{FEATURE_LABELS[later]}")

    print("""
Reading:
  gains all small   error magnitude is close to unpredictable from anything
                    available.  Report the limitation; it was tested, not
                    assumed.
  M3 - M2 positive  exogenous load says something about when the forecast
                    fails, and the GP does not see it.
  M3 - M2 negative  the load relationship does not transfer from the fitting
                    period to the evaluation period.  Check how load-price
                    correlation moves across blocks before reading anything
                    else into it.
""")

def split_is_covered(split, seq_len, covered, n_windows=308):
    """
    True if load exists for every row this split touches.

    The split reaches from the start of its own training window to the last
    forecast target, so both ends have to be inside the covered region.
    """
    start = block_start_row(split, seq_len)
    first = start - TRAIN_ROWS[seq_len]
    last = int(target_rows(split, seq_len, n_windows, seq_len).max())
    if first < 0 or last >= len(covered):
        return False
    return bool(covered[first:last + 1].all())

# =============================================================================
def main():
    rng = np.random.default_rng(0)
    members = list(MEMBERS)
    tag = "ensemble" if len(members) > 1 else f"tree_{members[0]}"

    price_frame, load_frame = read_frames(SEQ_LEN)
    load_frame = load_frame.reindex(price_frame.index)
    covered = load_frame.notna().all(axis=1).to_numpy()
    # if not price_frame.index.equals(load_frame.index):
    #     raise RuntimeError(
    #         "price and load frames are not index-aligned.  At seq_len 192 the "
    #         "price frame runs past the end of the load series, so this horizon "
    #         "cannot be used without dropping the uncovered blocks.")
    print(f"[frames] {len(price_frame)} hours,load covers {covered.sum()} "
          f"({100 * covered.mean():.1f}%)")

    prices = price_frame[[c for c in price_frame.columns
                          if c not in ("Subset", "date")]].to_numpy(np.float32)
    load = align_load(price_frame, load_frame, TRAIN_ROWS[SEQ_LEN])

    forecasts = read_forecasts(members, SEQ_LEN, RUN)
    all_splits = SPLIT_INDICES[SEQ_LEN]
    splits = [s for s in all_splits if split_is_covered(s, SEQ_LEN, covered)]
    dropped = [s for s in all_splits if s not in splits]
    if dropped:
        print(f"[coverage] dropping splits{dropped} spli: load ends before them")

    splits = splits[:N_SPLITS]
    print(f"[forecasts] {tag}, seq_len {SEQ_LEN}, run {RUN}; splits {splits}\n")

    blocks = []
    for split in splits:
        truth, forecast = forecasts[split]
        blocks.append(build_features(prices, load, truth, forecast,
                                     split, SEQ_LEN, rng))

    train_blocks = blocks[:-N_EVAL_BLOCKS]
    eval_blocks = blocks[-N_EVAL_BLOCKS:]

    y_train = np.concatenate([b[1] for b in train_blocks])
    y_eval = np.concatenate([b[1] for b in eval_blocks])
    abs_residual_eval = np.concatenate([b[2] for b in eval_blocks])
    print(f"[rows] {len(y_train):,} train, {len(y_eval):,} eval "
          f"(last {N_EVAL_BLOCKS} blocks held out)")

    results = {}
    for name in ("M0", "M1", "M2", "M3"):
        x_train = np.vstack([b[0][name] for b in train_blocks])
        x_eval = np.vstack([b[0][name] for b in eval_blocks])
        results[name] = fit_and_score(x_train, y_train, x_eval, y_eval,
                                      abs_residual_eval)
        results[name]["n_features"] = x_train.shape[1]

    print_results(results)

    out_path = Path(f"residual_magnitude_regression_{tag}_{SEQ_LEN}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()