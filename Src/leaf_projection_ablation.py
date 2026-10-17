"""
What does the leaf-combination projection actually learn?

Each tree concatenates its L leaves into [batch, seq_len, L*C] and maps that to
C channels with a single bias-free Linear(L*C -> C).  That layer does two jobs at
once: it combines leaves, and it decides which input channel feeds which output
channel.  Nothing constrains it to preserve channel identity, and nothing stops
it from simply averaging the leaves -- which would make the tree an ensemble with
extra steps rather than a set of specialists.

This script trains each of the top-10 configurations on a single split and then
reads the fitted weight matrix, reshaped to [L, C, C] as
[leaf, output channel, input channel].  Four diagnostics:

  norm share   fraction of total Frobenius norm carried by each leaf.  A leaf
               near zero is dead: the tree is effectively shallower than it looks.

  diagonality  share of a leaf's block mass sitting on its diagonal.  Near 1 means
               channel identity is preserved and the layer is only combining
               leaves; near 1/C means output channels are arbitrary mixtures of
               input channels.

  leaf cosine  mean cosine similarity between the vectorised blocks of different
               leaves.  Near 1 means the leaves are being averaged rather than
               selected between, so specialisation is not being used.

  gate spread  standard deviation of the per-timestep routing probabilities,
               collected by hooking the softmax modules.  Near 0 means every
               timestep is routed half-and-half and the leaves never specialise
               in the first place.

Taken together these say whether a constrained, per-channel leaf combination
(C x L weights instead of C x LC) would change anything, or whether the free
layer has already learned that solution on its own.

Usage:
    python inspect_leaf_projection.py
"""

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append("..")
parent_dir = Path.cwd().parent

from Benchmark_Models.utils import (_acquire_device, _select_criterion, create_args,
                                    EarlyStopping, make_param_groups, get_peak_lr)
from Benchmark_Models.Build_NAS_models import default_build_model_fn

# ---------------------------------------------------------------------------
SEQ_LEN    = 48
SPLIT      = 4            # any split; the layer is inspected after training on it
N_CHANNELS = 19
TREES      = range(10)
SAVE       = 'leaf_projection_diagnostics_channel.pkl'
GATE_TEMPERATURE = 1.0
ROUTING = 'timestep'

TOP10_PATH = f'{parent_dir}/Results_Evaluation/top10_ensemble_configs.pkl'
DATA_DIR   = f'{parent_dir}/Energy_Production_Data/EU_Time_Series'


def train_one_split(model, args, train_loader, criterion, device, optim):
    model.train()
    losses = []
    path = os.path.join(args.checkpoints, 'leaf_projection_probe')
    os.makedirs(path, exist_ok=True)

    steps = len(train_loader)
    warm = max(1, steps * args.warmup_epochs)
    warmup = (torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: min(1.0, s / warm))
        if args.warmup_epochs > 0 else None)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.8, patience=1, threshold=1e-4)
    stopper = EarlyStopping(patience=args.patience, delta=1e-4, verbose=False)

    step = 0
    for _ in range(args.train_epochs):
        for items in train_loader:
            x = items[0].float().to(device)
            y = items[1].float().to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            losses.append(loss.item())
            loss.backward()

            if step == 0:
                for d, level in enumerate(model.path_proba_modules):
                    for n, gate in enumerate(level):
                        g = gate.grad if isinstance(gate, torch.nn.Parameter) else gate.weight.grad
                        print(f'  [grad] depth {d} node {n}: '
                              f'{g.abs().mean().item():.3e}'
                              if g is not None else
                              f'  [grad] depth {d} node {n}: None')

            optim.step()
            if warmup is not None and step < warm:
                warmup.step()
            step += 1
        if step >= warm:
            plateau.step(np.mean(losses))
        stopper(np.mean(losses), model, path)
        if stopper.early_stop:
            break
    return model


def projection_blocks(model, C):
    """
    Fitted weight as [L, C_out, C_in].

    The leaves are concatenated along the feature axis as
    [leaf0_ch0..leaf0_chC-1, leaf1_ch0, ...], and nn.Linear stores weight as
    [out_features, in_features], so the input axis reshapes to (L, C).
    """
    W = model.output_leaf_projection.weight.detach().cpu().numpy()   # [C, L*C]
    C_out, in_features = W.shape
    L = in_features // C
    if in_features % C:
        raise RuntimeError(f'in_features {in_features} is not a multiple of {C}')
    return W.reshape(C_out, L, C).transpose(1, 0, 2), L              # [L, C, C]





def diagnostics(blocks, selected_features):
    L, C, _ = blocks.shape
    perm = np.asarray(selected_features)          # position j holds channel perm[j]
    inv = np.argsort(perm)
    aligned = blocks[:, :, inv]                   # columns back to channel order

    norms = np.linalg.norm(blocks.reshape(L, -1), axis=1) 
    share = norms / max(norms.sum(), 1e-12) 

    diagonality = np.array([
        np.abs(np.diag(aligned[l])).sum() / max(np.abs(aligned[l]).sum(), 1e-12)
        for l in range(L)])

    vecs = blocks.reshape(L, -1)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    sim = vecs @ vecs.T
    off = sim[~np.eye(L, dtype=bool)]
    leaf_cos = float(off.mean()) if L > 1 else float('nan')

    # what a per-channel (block-diagonal) layer would keep: the diagonals
    per_channel_weights = np.stack([np.diag(blocks[l]) for l in range(L)], axis=1)

    return {'norm_share': share, 'diagonality': diagonality,
            'leaf_cosine': leaf_cos, 'per_channel_weights': per_channel_weights,
            'n_leaves': L}


@torch.no_grad()
def gate_statistics(model, loader, device, max_batches=4):
    captured = []
    def hook(_m, _inp, out):
        p = torch.softmax(out.detach()/ model.gate_temperature, dim=-1)
        captured.append(p.detach().float().cpu().numpy())

    handles = []
    for level in model.path_proba_modules:
        for gate in level:
            handles.append(gate.register_forward_hook(hook))
    if not handles:
        return None

    model.eval()
    for i, items in enumerate(loader):
        model(items[0].float().to(device))
        if i + 1 >= max_batches:
            break
    for h in handles:
        h.remove()

    p = np.concatenate([c[..., 0].reshape(-1) for c in captured])
    return {'mean': float(p.mean()), 'std': float(p.std()),
            'frac_near_half': float(np.mean(np.abs(p - 0.5) < 0.05))}

@torch.no_grad()
def test_rmse(model,loader, device):
    model.eval()
    se, n = 0.0, 0
    for items in loader:
        x = items[0].float().to(device)
        y = items[1].float().to(device)
        e = (model(x) - y)**2
        se +=e.sum().item()
        n += e.numel()
    return float(np.sqrt(se/n))

@torch.no_grad()
def channel_gate_statistics(model):
    rows = []
    for d, level in enumerate(model.path_proba_modules):
        print(f'    [debug] level {d}: {type(level).__name__}, len {len(level)}')
        for n, gate in enumerate(level):
            print(f'    [debug] gate {n}: {type(gate).__name__}, shape {tuple(gate.shape)}')
            p = torch.softmax(gate/ model.gate_temperature, dim=-1)[:, 0]
            p = p.detach().cpu().numpy()
            rows.append({'depth': d, 'node': n, 'p': p,
                         'std': float(p.std()),
                         'frac_near_half': float(np.mean(np.abs(p - 0.5) < 0.05)),
                         'n_committed': int(np.sum(np.sum(p-0.5) > 0.25))})
    return rows


def channel_names(frame):
    return [c.split(' ')[0] for c in frame.columns if c not in ('Subset', 'date')]
# ---------------------------------------------------------------------------
def main():

    # run over 3 seeds
    for seed in range(0,3):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        with open(f'{DATA_DIR}/EU_Energy_Prices_Dataloader_torch_{SEQ_LEN}_Train.pkl',
                'rb') as f:
            loaders = pickle.load(f)
        with open(f'{DATA_DIR}/EU_Energy_Prices_Time_Series_{SEQ_LEN}_Train.pkl',
                'rb') as f:
            frame = pickle.load(f)
        names = channel_names(frame)
        with open(TOP10_PATH, 'rb') as f:
            configs = pickle.load(f)

        args = create_args(current_model='NAS_Tree', df=frame, use_gpu=True,
                        seq_len=SEQ_LEN, label_len=SEQ_LEN // 2, pred_len=SEQ_LEN)
        args.patience = 1
        device = _acquire_device(args=args)

        train_loader = loaders[f'Train_Test{SPLIT}']['Train_Set_Loader']
        criterion = _select_criterion()

        rows, store = [], {}
        for t in TREES:
            cfg = configs[t]
            model = default_build_model_fn(tree_config=cfg, nr_inputs=N_CHANNELS,
                                        nr_outputs=N_CHANNELS, seq_len=SEQ_LEN,
                                        pred_len=SEQ_LEN, tau=GATE_TEMPERATURE, routing_mode=ROUTING).float().to(device)
            optim = torch.optim.Adam(make_param_groups(
                model=model, peak_lr=get_peak_lr(n_nodes=cfg['nr_nodes'], base_lr=7e-4), path_lr_mult=1.0))
            model = train_one_split(model, args, train_loader, criterion, device, optim)

            test_loader = loaders[f'Train_Test{SPLIT}']['Test_Set_Loader']
            rmse = test_rmse(model, test_loader, device)

            # one batch pass with debug gates
            model.eval()
            model.debug_gates = True
            with torch.no_grad():
                model(next(iter(train_loader))[0].float().to(device))
            model.debug_gates = False
            if isinstance(model.output_leaf_projection, torch.nn.Parameter):
                # per-channel merge: channel identity is preserved by construction,
                # so diagonality is 1.0 and the leaf blocks do not exist.  What the
                # layer learns is a [C, L] table of how much each channel relies on
                # each leaf.
                W = model.output_leaf_projection.detach().cpu().numpy()   # [C, L]
                L = W.shape[1]
                d = {'norm_share': np.abs(W).sum(axis=0) / max(np.abs(W).sum(), 1e-12),
                    'diagonality': np.array([1.0] * L),
                    'leaf_cosine': float('nan'),
                    'per_channel_weights': W,
                    'n_leaves': L}
                print(f' leaf weights per channel (L={L}):') 
                for c in range(W.shape[0]): 
                    w = W[c] 
                    print(f' {names[c]:<16}' + ' '.join(f'{v:+.3f}' for v in w) + f' dominant leaf {int(np.argmax(np.abs(w)))}')
                
            else:
                blocks, L = projection_blocks(model, N_CHANNELS)
                d = diagnostics(blocks, model.selected_features.cpu().numpy())
                d['blocks'] = blocks.astype('float32')

            if getattr(model, 'routing_mode', 'timestep') == 'channel':
                gate_rows = channel_gate_statistics(model)
                g = {'std': float(np.mean([r['std'] for r in gate_rows])),
                    'frac_near_half': float(np.mean([r['frac_near_half'] for r in gate_rows])),
                    'mean': float(np.mean([r['p'] for r in gate_rows]))}

                perm = model.selected_features.cpu().numpy()
                for r in gate_rows:
                    order = np.argsort(r['p'])
                    print(f"    depth {r['depth']} node {r['node']}: "
                        f"p sd {r['std']:.3f}, {r['n_committed']}/{len(r['p'])} committed")
                    print("     left: ", [names[perm[j]] for j in order[:5]])
                    print("     right: ", [names[perm[j]] for j in order[-5:]])
            else:
                g = gate_statistics(model, train_loader, device)
            d['gates'] = g
            d['nr_nodes'] = cfg['nr_nodes']
            store[t] = {**d}

            rows.append((t, cfg['nr_nodes'], L, d['norm_share'].min(),
                        d['norm_share'].max(), d['diagonality'].mean(),
                        d['leaf_cosine'],
                        g['std'] if g else np.nan,
                        g['frac_near_half'] if g else np.nan,
                        rmse
                        ))
            print(f'tree {t} done ({cfg["nr_nodes"]} nodes, {L} leaves)', flush=True)

        print('\n' + '=' * 92)
        print(f"{'tree':>5}{'nodes':>7}{'leaves':>8}{'min share':>11}{'max share':>11}"
            f"{'diagonality':>13}{'leaf cos':>10}{'gate sd':>9}{'p~0.5':>8}{'RMSE':>8}")
        print('-' * 92)
        for r in rows:
            print(f'{r[0]:>5}{r[1]:>7}{r[2]:>8}{r[3]:>11.3f}{r[4]:>11.3f}'
                f'{r[5]:>13.3f}{r[6]:>10.3f}{r[7]:>9.3f}{r[8]:>8.3f}{r[9]:>8.3f}')

        arr = np.array([[r[5], r[6]] for r in rows])
        print('-' * 92)
        print(f"{'mean':>5}{'':>7}{'':>8}{'':>11}{'':>11}"
            f"{arr[:, 0].mean():>13.3f}{arr[:, 1].mean():>10.3f}")
        print(f'\nreference: diagonality of a channel-preserving layer = 1.000, '
            f'of an arbitrary mixture = {1.0/N_CHANNELS:.3f}')

    print("""
Reading:
  diagonality high, leaf cosine low
      -> the free layer already behaves like a per-channel leaf combination and
         constraining it would change little.  The tree is using its leaves.
  diagonality low
      -> output channels are mixtures of input channels; the constrained form is
         a real design change worth testing.
  leaf cosine near 1, or one leaf holding nearly all the norm
      -> the leaves are being averaged, or all but one is dead.  Either way the
         hierarchical composition is not doing the work the search space assumes,
         which is a finding in its own right.
  gate sd near 0 with p~0.5 high
      -> routing never commits, so the leaves cannot specialise regardless of how
         the projection is parameterised.  Fix routing before the projection.
""")

    with open(SAVE, 'wb') as f:
        pickle.dump(store, f)
    print(f'saved -> {SAVE}')


if __name__ == '__main__':
    main()

