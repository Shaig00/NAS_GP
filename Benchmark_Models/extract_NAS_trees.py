
import pickle
import json
import numpy as np

"""
Extract the top-10 architectures from the NAS sampling history, pooled
across ALL episodes.

The history was saved as:
    [history_dict, actor_critic_model, optimizer_actor_critic,
     embedding_matrix, embedding_none_matrix, current_episode,
     best_embedded_representation_episode_start, episode_start_cell,
     best_reward_so_far]

and history_dict has, per episode N:
    'Sampled_Networks_Episode_N'          -> list of 10 tree_config dicts
    'Rewards_Episode_N'                   -> list of 10 first-subset rewards
    'Rewards_SecondSubset_Episode_N'      -> list of 10 second-subset rewards
    'AvgNetworkProba_Episode_N'           -> controller action probability
    'Sampled_Networks_Episode_Entropy_N'  -> per-network entropy
    'Sampled_Networks_Episode_GED_N'      -> per-network GED
    'Sampled_networks_Episode_Runtime_N'  -> per-network runtime

All 10 entries in each list are index-aligned: network i's first-subset
reward is Rewards_Episode_N[i], etc.
"""


def load_history(path='Results_Evaluation/WV_GED_NAS_Sampling_History.pkl'):
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    history_dict = saved[0]
    current_episode = saved[5]
    return history_dict, current_episode


def _episode_indices(history_dict, truncate=1000):
    """Recover the set of episode numbers actually present in the dict, by
    parsing the 'Sampled_Networks_Episode_{N}' keys (robust to gaps / to not
    knowing the exact final episode count)."""
    eps = []
    prefix = 'Sampled_Networks_Episode_'
    for k in history_dict.keys():
        # match 'Sampled_Networks_Episode_12' but NOT '..._Entropy_12' / '..._GED_12'
        if k.startswith(prefix):
            suffix = k[len(prefix):]
            if suffix.isdigit():
                eps.append(int(suffix))
    return sorted(eps)[:truncate] if truncate is not None else sorted(eps)


def _config_signature(config):
    """A hashable, canonical signature of a tree_config, so we can detect
    duplicate architectures sampled in different episodes. Uses a sorted
    JSON dump so key/order differences don't create false distinctions."""
    return json.dumps(config, sort_keys=True, default=str)


def extract_top_k(history_dict, k=10,
                  reward_mode='second',
                  dedupe=True,
                  min_convergence=None, 
                  convergence_weight=0.0,
                  exclude_reward=None,
                  truncate=1000):
                                           
    """
    Pool every sampled network across all episodes, rank, return the top-k.

    reward_mode:
      'second' -> rank by second-subset (walk-forward) reward  [recommended]
      'first'  -> rank by first-subset reward
      'sum'    -> rank by first + second
      'min'    -> rank by min(first, second)  [most conservative]

    dedupe: keep only the best occurrence of each distinct architecture.

    CONVERGENCE FILTERING/WEIGHTING (new):
    Each architecture was sampled in some episode, which has an AvgNetworkProba
    (the controller confidence at that point). An architecture sampled in a
    HIGH-confidence episode is one the controller had actually converged toward
    -- versus a high-reward architecture sampled early during exploration, which
    may be a lucky one-off the controller never learned to prefer.

      min_convergence: if set (e.g. 0.4), DROP any architecture sampled in an
          episode whose AvgNetworkProba is below this threshold. Hard filter --
          keeps only architectures from episodes where the controller was
          reasonably confident.

      convergence_weight: if > 0, blend convergence into the ranking score:
              combined = (1 - w) * norm_reward + w * convergence
          where norm_reward is min-max normalized reward across the pool and
          convergence is the episode's AvgNetworkProba (already in [0,1]).
          0.0 = rank by reward only (default). Try ~0.2-0.3 to let convergence
          break ties / nudge toward architectures the controller preferred.
    """
    episodes = _episode_indices(history_dict)

    if truncate is not None:
        episodes = episodes[:truncate]

    # per-episode controller confidence
    def _episode_conv(ep):
        key = f'AvgNetworkProba_Episode_{ep}'
        if key in history_dict:
            return float(np.mean(history_dict[key]))
        return np.nan

    pooled = []
    for ep in episodes:
        nets = history_dict[f'Sampled_Networks_Episode_{ep}']
        first = history_dict[f'Rewards_Episode_{ep}']
        second = history_dict.get(f'Rewards_SecondSubset_Episode_{ep}', [None] * len(nets))
        conv = _episode_conv(ep)

        # hard convergence filter
        if min_convergence is not None and (not np.isfinite(conv) or conv < min_convergence):
            continue

        for i, config in enumerate(nets):
            r1 = first[i] if first[i] is not None else np.nan
            r2 = second[i] if second[i] is not None else np.nan

            if reward_mode == 'second':
                score = r2
            elif reward_mode == 'first':
                score = r1
            elif reward_mode == 'sum':
                score = np.nansum([r1, r2])
            elif reward_mode == 'min':
                score = np.nanmin([r1, r2])
            else:
                raise ValueError(f"unknown reward_mode: {reward_mode}")

            if not np.isfinite(score):
                continue

            if (exclude_reward is not None
                and (abs((r1+r2)- exclude_reward) < 1e-6 or 
                    abs(r1 - exclude_reward/2) < 1e-6 or
                    abs(r2 - exclude_reward/2) < 1e-6)):
                continue

            pooled.append({
                'config': config,
                'first_reward': r1,
                'second_reward': r2,
                'reward_score': float(score),
                'convergence': conv,
                'episode': ep,
                'idx_in_episode': i,
            })


    # ranking score: reward only, or blended with convergence
    if convergence_weight > 0.0 and pooled:
        rewards = np.array([e['reward_score'] for e in pooled])
        rmin, rmax = rewards.min(), rewards.max()
        rng = (rmax - rmin) if (rmax > rmin) else 1.0
        for e in pooled:
            norm_r = (e['reward_score'] - rmin) / rng
            conv = e['convergence'] if np.isfinite(e['convergence']) else 0.0
            e['score'] = (1 - convergence_weight) * norm_r + convergence_weight * conv
    else:
        for e in pooled:
            e['score'] = e['reward_score']

    pooled.sort(key=lambda d: d['score'], reverse=True)

    if dedupe:
        seen = set()
        deduped = []
        for entry in pooled:
            sig = _config_signature(entry['config'])
            if sig not in seen:
                seen.add(sig)
                deduped.append(entry)
        pooled = deduped

    return pooled[:k]


def extract_top_k_from_best_episodes(history_dict, k=10, n_episodes=10,
                                     reward_mode='second', dedupe=True, exclude_reward=None, truncate=1000):
    """
    Select architectures from the N episodes with the highest AvgNetworkProba
    (the most-converged episodes), pool their trees, and return the top-k by
    reward within that pool.

    Rationale: the most-converged episodes are where the controller had the
    strongest, most-settled preferences -- so their sampled architectures
    reflect 'what the search converged to' rather than early exploration. Pooling
    the top-N episodes (rather than just the single best) preserves diversity and
    lets reward break ties within the converged region.

    n_episodes: how many top-convergence episodes to pool trees from.
    """
    episodes = _episode_indices(history_dict)

    if truncate is not None:
        episodes = episodes[:truncate]

    # rank episodes by convergence, take the top n_episodes
    ep_conv = []
    for ep in episodes:
        key = f'AvgNetworkProba_Episode_{ep}'
        conv = float(np.mean(history_dict[key])) if key in history_dict else np.nan
        if np.isfinite(conv):
            ep_conv.append((ep, conv))
    ep_conv.sort(key=lambda t: t[1], reverse=True)
    best_episodes = [ep for ep, _ in ep_conv[:n_episodes]]

    print(f"Pooling trees from the {len(best_episodes)} highest-convergence episodes: "
          f"{sorted(best_episodes)}")
    print(f"  their AvgNetworkProba range: "
          f"{ep_conv[min(n_episodes, len(ep_conv))-1][1]:.3f} .. {ep_conv[0][1]:.3f}")

    # pool trees only from those episodes, then rank by reward
    pooled = []
    for ep in best_episodes:
        nets = history_dict[f'Sampled_Networks_Episode_{ep}']
        first = history_dict[f'Rewards_Episode_{ep}']
        second = history_dict.get(f'Rewards_SecondSubset_Episode_{ep}', [None] * len(nets))
        key = f'AvgNetworkProba_Episode_{ep}'
        conv = float(np.mean(history_dict[key])) if key in history_dict else np.nan

        for i, config in enumerate(nets):
            r1 = first[i] if first[i] is not None else np.nan
            r2 = second[i] if second[i] is not None else np.nan
            if reward_mode == 'second':
                score = r2
            elif reward_mode == 'first':
                score = r1
            elif reward_mode == 'sum':
                score = np.nansum([r1, r2])
            elif reward_mode == 'min':
                score = np.nanmin([r1, r2])
            else:
                raise ValueError(f"unknown reward_mode: {reward_mode}")
            if not np.isfinite(score):
                continue
            if (exclude_reward is not None
                and (abs((r1+r2)- exclude_reward) < 1e-6 or 
                    abs(r1 - exclude_reward/2) < 1e-6 or
                    abs(r2 - exclude_reward/2) < 1e-6)):
                continue
            pooled.append({
                'config': config, 'first_reward': r1, 'second_reward': r2,
                'reward_score': float(score), 'score': float(score),
                'convergence': conv, 'episode': ep, 'idx_in_episode': i,
            })

    pooled.sort(key=lambda d: d['score'], reverse=True)

    if dedupe:
        seen = set()
        deduped = []
        for entry in pooled:
            sig = _config_signature(entry['config'])
            if sig not in seen:
                seen.add(sig)
                deduped.append(entry)
        pooled = deduped

    return pooled[:k]


def diagnose_reward_distribution(history_dict, reward_mode='second', truncate=1000):
    """Check whether rewards cluster at a ceiling (e.g. many at ~3), which would
    make reward-ranking uninformative and points to a flat reward landscape /
    possible cap or fallback artifact."""
    all_scores = []
    episodes = _episode_indices(history_dict)
    if truncate is not None:
        episodes = episodes[:truncate]

    for ep in episodes:
        nets = history_dict[f'Sampled_Networks_Episode_{ep}']
        first = history_dict[f'Rewards_Episode_{ep}']
        second = history_dict.get(f'Rewards_SecondSubset_Episode_{ep}', [None] * len(nets))
        for i in range(len(nets)):
            r1 = first[i] if first[i] is not None else np.nan
            r2 = second[i] if second[i] is not None else np.nan
            s = r2 if reward_mode == 'second' else r1
            if np.isfinite(s):
                all_scores.append(float(s))
    a = np.array(all_scores)
    print(f"\n=== reward distribution ({reward_mode} subset, {len(a)} finite trees) ===")
    print(f"  min {a.min():.3f}  p25 {np.percentile(a,25):.3f}  median {np.median(a):.3f}  "
          f"p75 {np.percentile(a,75):.3f}  max {a.max():.3f}")
    # how concentrated near the max?
    top = a.max()
    near_top = np.mean(a >= top - 0.05 * abs(top))
    print(f"  fraction within 5% of max ({top:.3f}): {near_top:.1%}  "
          f"({'CLUSTERED near a ceiling -> reward-ranking weak, lean on convergence' if near_top > 0.3 else 'spread out -> reward-ranking is informative'})")
    # histogram-ish
    for lo, hi in [(a.min(), np.percentile(a,25)), (np.percentile(a,25), np.median(a)),
                   (np.median(a), np.percentile(a,75)), (np.percentile(a,75), a.max())]:
        print(f"    [{lo:6.3f}, {hi:6.3f}): {np.mean((a>=lo)&(a<hi)):.1%}")

import pandas as pd
def summarize_top_k(top_k):
    """Print a compact table of the selected architectures."""
    print(f"{'rank':>4}  {'score':>7}  {'reward':>7}  {'conv':>6}  "
          f"{'ep':>4}  {'nodes':>5}  modules")
    rows =[]
    for rank, e in enumerate(top_k, 1):
        cfg = e['config']
        modules = ','.join(m.replace('Dnonlinear_', 'Dnl_') for m in cfg['ts_module_nodes'])
        conv = e['convergence'] if np.isfinite(e['convergence']) else float('nan')
        print(f"{rank:>4}  {e['score']:>7.4f}  {e['reward_score']:>7.4f}  "
              f"{conv:>6.3f}  {e['episode']:>4}  {cfg['nr_nodes']:>5}  {modules}")
        rows.append([rank,  e['score'], e['reward_score'], conv, e['episode'], cfg['nr_nodes']])
    return pd.DataFrame(rows, columns=['rank', 'score', 'reward_score', 'conv', 'episode', 'nodes'])


def get_top_k_configs(top_k):
    """Return just the list of config dicts, ready to hand to build_model_fn
    / retrain_top10 / nas_top10_forecasting."""
    return [e['config'] for e in top_k]


def _module_signature(config):
    """Signature based ONLY on the module pattern and node count. Collapses
    architectures that share the same modules but differ in hyperparameters,
    so it identifies distinct DESIGNS rather than distinct configs. Use this
    when you want a benchmark set of genuinely different architectures rather
    than many hyperparameter variants of the same module pattern."""
    return json.dumps({
        'nr_nodes': config.get('nr_nodes'),
        'ts_module_nodes': config.get('ts_module_nodes'),
    }, sort_keys=True, default=str)
 
 
def count_module_patterns(top_entries):
    """Report how many distinct module patterns vs distinct full configs are
    present in a ranked list of entries. Use this to decide whether a
    'best per module pattern' selection would yield a useful number of
    distinct designs before committing to it."""
    module_sigs, config_sigs = set(), set()
    for entry in top_entries:
        module_sigs.add(_module_signature(entry['config']))
        config_sigs.add(_config_signature(entry['config']))
    print(f"  entries examined:            {len(top_entries)}")
    print(f"  distinct full configs:       {len(config_sigs)}")
    print(f"  distinct module patterns:    {len(module_sigs)}")
    # also show the patterns and how many of each
    from collections import Counter
    counts = Counter()
    for entry in top_entries:
        nodes = entry['config'].get('nr_nodes')
        mods = ','.join(str(m) for m in entry['config'].get('ts_module_nodes', []))
        counts[f"[{nodes}] {mods}"] += 1
    print("  module patterns (count):")
    for pat, n in counts.most_common():
        print(f"    {n:>3}x  {pat}")
    return {'n_configs': len(config_sigs), 'n_patterns': len(module_sigs)}
 
 
def extract_best_per_module_pattern(top_entries, k=None):
    """From a ranked (best-first) list of entries, keep only the best-scoring
    entry for each distinct module pattern. Returns distinct DESIGNS, best-first.
 
    Pass in a list that is ALREADY sorted best-first (e.g. the output of
    extract_top_k or extract_top_k_from_best_episodes with a large k), so that
    the first occurrence of each pattern is its best-scoring instance."""
    seen = set()
    out = []
    for entry in top_entries:
        sig = _module_signature(entry['config'])
        if sig not in seen:
            seen.add(sig)
            out.append(entry)
    return out[:k] if k else out



if __name__ == '__main__':
    history_dict, current_episode = load_history(
        'Results_Evaluation/WV_GED_NAS_Sampling_History.pkl'
    )
    print(f"Loaded history up to episode {current_episode}, "
          f"{len(_episode_indices(history_dict))} episodes present.\n")

    # --- diagnose reward distribution first (is it clustered at a ceiling?) ---
    diagnose_reward_distribution(history_dict, reward_mode='second')

    # --- reward only (original behaviour) ---
    print("\n=== Top-10 by reward only ===")
    top_reward = extract_top_k(history_dict, k=10, reward_mode='second', dedupe=True)
    summarize_top_k(top_reward)

    # --- pool trees from the 10 highest-convergence episodes, rank by reward ---
    print("\n=== Top-10 pooled from the 10 highest-convergence episodes ===")
    top_best_eps = extract_top_k_from_best_episodes(
        history_dict, k=10, n_episodes=10, reward_mode='second', dedupe=True)
    summarize_top_k(top_best_eps)

    # choose which selection to save
    chosen = top_best_eps   # <- using the converged-episodes pool; change if desired
    top_k_configs = get_top_k_configs(chosen)
    with open('Results_Evaluation/top10_configs.pkl', 'wb') as f:
        pickle.dump(top_k_configs, f)
    print(f"\nSaved {len(top_k_configs)} configs to Results_Evaluation/top10_configs.pkl")