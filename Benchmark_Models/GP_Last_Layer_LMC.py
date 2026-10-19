"""
Bayesian last-layer (GP) head for a trained, frozen, NAS-selected TreeModel.

Assumes TreeModel exposes:
    get_pooled_representation(x) -> [batch, seq_len, 4n]
and has trained attributes output_leaf_projection, pred_len_projection,
prediction_projections.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import gpytorch
from scipy.stats import norm
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader


def fit_gp_on_cache(fit_inputs, fit_residuals, val_inputs, val_residuals,
                    n_channels=19, structure="icm", num_latents=1,
                    num_inducing=200, n_epochs=30, device="cuda",
                    gp_lr=1e-2, likelihood_lr=1e-2, gp_batch_points=1536,
                    min_noise=1e-4, signal_fraction=0.1, patience=8,
                    seed=0, verbose=False):
    """Core GP fit on already-cached (input, residual) tensors."""
    gp, likelihood = build_icm_lmc_gp(
        fit_inputs, fit_residuals, num_tasks=n_channels,
        num_inducing=min(num_inducing, fit_inputs.shape[0]),
        structure=structure, num_latents=num_latents, device=device,
        min_noise=min_noise, signal_fraction=signal_fraction, seed=seed,
    )
 
    loader = DataLoader(TensorDataset(fit_inputs, fit_residuals),
                        batch_size=gp_batch_points, shuffle=True)
    optimizer = torch.optim.Adam([
        {"params": gp.parameters(), "lr": gp_lr},
        {"params": likelihood.parameters(), "lr": likelihood_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=int(fit_inputs.shape[0]))
 
    best_nll, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for epoch in range(n_epochs):
        gp.train(); likelihood.train()
        for gp_input, residual in loader:
            gp_input = gp_input.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(gp(gp_input), residual)
            loss.backward()
            optimizer.step()
        scheduler.step()
 
        m = validate(gp, likelihood, val_inputs, val_residuals, device)
        if m["nll"] < best_nll - 1e-5:
            best_nll, best_epoch, bad = m["nll"], epoch + 1, 0
            best_state = (copy.deepcopy(gp.state_dict()),
                          copy.deepcopy(likelihood.state_dict()))
        else:
            bad += 1
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    ep {epoch+1:3d} | val NLL {m['nll']:.4f} | lam {m['lambda']:.3f}")
        if bad >= patience:
            break
 
    if best_state is not None:
        gp.load_state_dict(best_state[0])
        likelihood.load_state_dict(best_state[1])
 
    lam = validate(gp, likelihood, val_inputs, val_residuals, device)["lambda"]
    return gp, likelihood, {"best_epoch": best_epoch, "best_val_nll": best_nll, "lambda": lam}
# ---------------------------------------------------------------------------
# Tree forecast
# ---------------------------------------------------------------------------

def configure_tree_for_gp(tree_model):
    """Freeze every TreeModel parameter.  The GP head never fine-tunes the tree."""
    for p in tree_model.parameters():
        p.requires_grad = False
    tree_model.eval()
    return tree_model


def get_tree_forecast(tree_model, x):
    """Final tree forecast after all trained projection layers."""
    pooled = tree_model.get_pooled_representation(x)          # [B, seq_len, concat_dim]
    z = tree_model.output_leaf_projection(pooled)             # [B, seq_len, 19]

    if (getattr(tree_model, 'prediction_projections', False) and
            getattr(tree_model, 'pred_len_projection', None) is not None):
        z = tree_model.pred_len_projection(z.permute(0, 2, 1)).permute(0, 2, 1)
    return z


def add_normalized_horizon(z):
    """z: [B, pred_len, C] -> [B, pred_len, C + 1], horizon in [0, 1]."""
    bsz, pred_len, _ = z.shape
    if pred_len == 1:
        horizon = torch.zeros(1, 1, 1, device=z.device, dtype=z.dtype)
    else:
        horizon = torch.linspace(0.0, 1.0, pred_len,
                                 device=z.device, dtype=z.dtype).view(1, pred_len, 1)
    horizon = horizon.expand(bsz, -1, -1)
    return torch.cat([z, horizon], dim=-1)


# ---------------------------------------------------------------------------
# Cached GP dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_gp_cache(tree_model, data_loader, add_horizon, device):
    """
    Run the frozen tree once over a loader and cache the GP training pairs.

    Returns
    -------
    gp_inputs : [N * pred_len, C (+1)]   on CPU
    residuals : [N * pred_len, C]        on CPU
    pred_len  : int
    """
    tree_model.eval()
    inputs, residuals = [], []
    pred_len = None

    for items in data_loader:
        x = items[0].float().to(device, non_blocking=True)
        y = items[1].float().to(device, non_blocking=True)

        z = get_tree_forecast(tree_model, x)
        if z.shape != y.shape:
            raise RuntimeError(
                f"Tree forecast shape {tuple(z.shape)} != target shape {tuple(y.shape)}"
            )

        residual = y - z
        gp_input = add_normalized_horizon(z) if add_horizon else z

        bsz, pred_len, D = gp_input.shape
        inputs.append(gp_input.reshape(bsz * pred_len, D).cpu())
        residuals.append(residual.reshape(bsz * pred_len, -1).cpu())

    return torch.cat(inputs, 0), torch.cat(residuals, 0), pred_len


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TreeICMLMCGP(gpytorch.models.ApproximateGP):
    """
    structure="icm": one latent GP  -> rank-one task mixing
    structure="lmc": several latent GPs -> linear model of coregionalisation
    """

    def __init__(self, inducing_points, num_tasks=19, num_latents=1,
                 structure="icm", ard_num_dims=None):
        if structure not in {"icm", "lmc"}:
            raise ValueError("structure must be either 'icm' or 'lmc'")
        if structure == "icm":
            num_latents = 1

        self.num_tasks = num_tasks
        self.num_latents = num_latents
        self.structure = structure

        M = inducing_points.shape[-2]

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=M, batch_shape=torch.Size([num_latents])
        )
        base_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            base_strategy, num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
        # ARD: one lengthscale per input dimension, so the head can weight the
        # horizon coordinate separately from the individual price channels.
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([num_latents]),
                ard_num_dims=ard_num_dims,
            ),
            batch_shape=torch.Size([num_latents]),
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def initialize_inducing_points(gp_inputs, num_inducing, num_latents, device,
                               max_kmeans_points=100_000, seed=0):
    n = gp_inputs.shape[0]
    idx = torch.randperm(n)[:min(max_kmeans_points, n)]
    sample = gp_inputs[idx].cpu().numpy()

    kmeans = KMeans(n_clusters=num_inducing, n_init=10, random_state=seed)
    kmeans.fit(sample)

    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    return centers.unsqueeze(0).repeat(num_latents, 1, 1)


def median_lengthscale(gp_inputs, max_points=2000, seed=0):
    """Median pairwise distance, a standard RBF lengthscale initialisation."""
    g = torch.Generator().manual_seed(seed)
    n = gp_inputs.shape[0]
    idx = torch.randperm(n, generator=g)[:min(max_points, n)]
    sample = gp_inputs[idx].float()
    d = torch.cdist(sample, sample)
    med = d[d > 0].median().item()
    return max(med, 1e-3)


def set_likelihood_noise(likelihood, value_per_task):
    """
    Set the observation noise from the empirical residual variance.
    MultitaskGaussianLikelihood exposes task noises differently across GPyTorch
    versions, so fall back to the shared noise if the per-task path is absent.
    """
    v = torch.as_tensor(value_per_task, dtype=torch.float32)
    try:
        likelihood.task_noises = v.to(likelihood.raw_task_noises.device)
        return "task_noises"
    except Exception:
        try:
            likelihood.noise = float(v.mean())
            return "noise"
        except Exception:
            return "unset"


def _read_noise(likelihood):
    for attr in ("task_noises", "noise"):
        try:
            val = getattr(likelihood, attr)
            return attr, val.detach().cpu().flatten()
        except Exception:
            continue
    return "unknown", torch.tensor([float('nan')])


def build_icm_lmc_gp(train_gp_inputs, train_residuals, num_tasks=19,
                     num_inducing=200, structure="icm", num_latents=8,
                     device='cuda', min_noise=1e-4, signal_fraction=0.1, seed=0):
    """
    signal_fraction : share of the empirical residual variance assigned to the
        GP prior at initialisation; the remainder goes to observation noise.
        Starting conservatively (most variance as noise) makes the head earn
        signal rather than assuming it, which is the failure mode that produces
        a confident non-transferring mean correction.
    """
    if structure == "icm":
        num_latents = 1
    elif structure == "lmc":
        if num_latents < 2:
            raise ValueError("num_latents must be >= 2 for LMC structure")
    else:
        raise ValueError("structure must be either 'icm' or 'lmc'")

    D = train_gp_inputs.shape[-1]

    inducing_points = initialize_inducing_points(
        train_gp_inputs, num_inducing=num_inducing,
        num_latents=num_latents, device=device, seed=seed
    )

    gp = TreeICMLMCGP(
        inducing_points=inducing_points,
        num_tasks=num_tasks,
        num_latents=num_latents,
        structure=structure,
        ard_num_dims=D,
    ).to(device)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks,
        noise_constraint=gpytorch.constraints.GreaterThan(min_noise),
    ).to(device)

    # --- variance-based initialisation ------------------------------------
    var_per_task = train_residuals.var(dim=0).clamp_min(min_noise)     # [C]
    total_var = float(var_per_task.mean())

    with torch.no_grad():
        gp.covar_module.outputscale = torch.full_like(gp.covar_module.outputscale, signal_fraction * total_var)
        ls_init = median_lengthscale(train_gp_inputs, seed=seed)
        gp.covar_module.base_kernel.lengthscale = torch.full_like(gp.covar_module.base_kernel.lengthscale, ls_init)
        where = set_likelihood_noise(likelihood, (1.0 - signal_fraction) * var_per_task)

    print(f"[init] residual var (mean over tasks) = {total_var:.4f} | "
          f"outputscale = {signal_fraction * total_var:.4f} | "
          f"lengthscale = {ls_init:.4f} | noise set via '{where}'")

    return gp, likelihood


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def gp_predict_cached(gp, likelihood, gp_inputs, device, batch_points=8192):
    """Residual posterior over a cached input tensor.  Returns tensors stored in CPU."""
    gp.eval()
    likelihood.eval()
    means, variances = [], []
    with gpytorch.settings.fast_pred_var():
        for i in range(0, gp_inputs.shape[0], batch_points):
            chunk = gp_inputs[i:i + batch_points].to(device, non_blocking=True)
            dist = likelihood(gp(chunk))
            means.append(dist.mean.cpu())
            variances.append(dist.variance.cpu())
    return torch.cat(means, 0), torch.cat(variances, 0)


def fit_lambda(residual, mu):
    """
    Closed-form least-squares shrinkage on the correction, clipped to [0, 1].

    The corrected error is (r - lambda * mu), so
        lambda* = <r, mu> / <mu, mu>
    lambda = 0 recovers the tree forecast exactly.
    """
    denom = float((mu * mu).sum())
    if denom <= 0:
        return 0.0
    lam = float((residual * mu).sum()) / denom
    return float(min(max(lam, 0.0), 1.0))


def rmse(e):
    return float(torch.sqrt((e ** 2).mean()))


def gaussian_nll(residual, mu, var):
    var = var.clamp_min(1e-8)
    return float((0.5 * (torch.log(2 * math.pi * var) + (residual - mu) ** 2 / var)).mean())


@torch.no_grad()
def validate(gp, likelihood, val_inputs, val_residuals, device):
    """Point diagnostic + NLL on held-out data."""
    mu, var = gp_predict_cached(gp, likelihood, val_inputs, device)

    rmse_tree = rmse(val_residuals)                 # error of z alone
    rmse_full = rmse(val_residuals - mu)            # error of z + mu_r
    lam = fit_lambda(val_residuals, mu)
    rmse_lam = rmse(val_residuals - lam * mu)
    nll = gaussian_nll(val_residuals, mu, var)

    return {
        "rmse_tree": rmse_tree,
        "rmse_full_correction": rmse_full,
        "lambda": lam,
        "rmse_shrunk": rmse_lam,
        "nll": nll,
        "mu_abs_mean": float(mu.abs().mean()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_icm_lmc(tree_model, fit_loader, val_loader=None, add_horizon=True,
                  n_channels=19, structure="icm", num_latents=1,
                  num_inducing=200, n_epochs=50, device='cuda',
                  gp_lr=1e-2, likelihood_lr=1e-2, gp_batch_points=1536,
                  min_noise=1e-4, signal_fraction=0.1, patience=8,
                  log_every=1, seed=0):
    """
    fit_loader : loader whose residuals the GP is fitted on. 
    val_loader : held-out loader for early stopping and the lambda fit.
    """
    configure_tree_for_gp(tree_model)

    # ---- cache the frozen tree's outputs once ----------------------------
    fit_inputs, fit_residuals, pred_len = build_gp_cache(
        tree_model, fit_loader, add_horizon, device
    )
    print(f"[cache] fit: {tuple(fit_inputs.shape)} inputs, pred_len = {pred_len}")

    val_inputs = val_residuals = None
    if val_loader is not None:
        val_inputs, val_residuals, _ = build_gp_cache(
            tree_model, val_loader, add_horizon, device
        )
        print(f"[cache] val: {tuple(val_inputs.shape)} inputs")
        if val_inputs.shape[-1] != fit_inputs.shape[-1]:
            raise RuntimeError("fit/val GP input dimension mismatch. Check add_horizon.")

    gp, likelihood = build_icm_lmc_gp(
        fit_inputs, fit_residuals, num_tasks=n_channels,
        num_inducing=num_inducing, structure=structure,
        num_latents=num_latents, device=device,
        min_noise=min_noise, signal_fraction=signal_fraction, seed=seed,
    )

    cached = DataLoader(
        TensorDataset(fit_inputs, fit_residuals),
        batch_size=gp_batch_points, shuffle=True, drop_last=False,
    )

    # Single parameter group: anything added to the model later trains too.
    optimizer = torch.optim.Adam([
        {'params': gp.parameters(), 'lr': gp_lr},
        {'params': likelihood.parameters(), 'lr': likelihood_lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # ELBO scaling: total number of GP observations = windows * pred_len.
    num_data = int(fit_inputs.shape[0])
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=num_data)

    best_nll, best_state, best_epoch, bad_epochs = float('inf'), None, -1, 0
    history = []

    for epoch in range(n_epochs):
        gp.train()
        likelihood.train()
        tree_model.eval()

        epoch_loss, n_points = 0.0, 0
        for gp_input, residual in cached:
            gp_input = gp_input.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = -mll(gp(gp_input), residual)
            loss.backward()
            optimizer.step()

            npts = gp_input.shape[0]
            epoch_loss += loss.item() * npts
            n_points += npts

        scheduler.step()
        epoch_loss /= max(1, n_points)

        ls = gp.covar_module.base_kernel.lengthscale.detach().cpu().flatten()
        noise_name, noise_val = _read_noise(likelihood)

        record = {
            "epoch": epoch + 1,
            "elbo_loss": epoch_loss,
            "lengthscale_mean": float(ls.mean()),
            "lengthscale_min": float(ls.min()),
            f"{noise_name}_mean": float(noise_val.mean()),
            f"{noise_name}_min": float(noise_val.min()),
        }

        if val_inputs is not None:
            record.update(validate(gp, likelihood, val_inputs, val_residuals, device))

            # Early stopping on validation NLL: the head is retained for
            # uncertainty, so calibration is the criterion, not point error.
            if record["nll"] < best_nll - 1e-5:
                best_nll = record["nll"]
                best_state = (copy.deepcopy(gp.state_dict()),
                              copy.deepcopy(likelihood.state_dict()))
                best_epoch = epoch + 1
                bad_epochs = 0
            else:
                bad_epochs += 1

        history.append(record)

        if (epoch + 1) % log_every == 0 or epoch == n_epochs - 1:
            msg = (f"Epoch {epoch+1}/{n_epochs} | ELBO {epoch_loss:.4f} "
                   f"| ls {record['lengthscale_mean']:.3f} "
                   f"| {noise_name} {record[f'{noise_name}_mean']:.4f}")
            if val_inputs is not None:
                msg += (f" | val NLL {record['nll']:.4f}"
                        f" | RMSE tree {record['rmse_tree']:.4f}"
                        f" -> full {record['rmse_full_correction']:.4f}"
                        f" | lam {record['lambda']:.3f}"
                        f" -> {record['rmse_shrunk']:.4f}")
            print(msg)

        if val_inputs is not None and bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch}, "
                  f"val NLL {best_nll:.4f})")
            break

    if best_state is not None:
        gp.load_state_dict(best_state[0])
        likelihood.load_state_dict(best_state[1])

    lam_final = 0.0
    if val_inputs is not None:
        final = validate(gp, likelihood, val_inputs, val_residuals, device)
        lam_final = final["lambda"]
        print("\n[final] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in final.items()))
        if final["rmse_full_correction"] >= final["rmse_tree"]:
            print("[final] full residual correction does NOT improve the point "
                  "forecast; lambda shrinkage selected instead.")

    return gp, likelihood, {"history": history, "lambda": lam_final,
                            "best_epoch": best_epoch, "pred_len": pred_len}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_icm_lmc(tree_model, gp, likelihood, x, add_horizon=True,
                    device='cuda', n_channels=19, lam=0.0):
    """
    lam : weight on the GP residual mean.
          lam = 0.0 -> point forecast is the tree's, GP supplies variance only
                       (the configuration reported in the thesis).
          lam = 1.0 -> full additive residual correction.

    Returns mean, var, both [batch, pred_len, n_channels].
    """
    tree_model.eval()
    gp.eval()
    likelihood.eval()

    x = x.float().to(device)
    z = get_tree_forecast(tree_model, x)

    gp_input = add_normalized_horizon(z) if add_horizon else z
    bsz, pred_len, D = gp_input.shape
    gp_input = gp_input.reshape(bsz * pred_len, D)

    with gpytorch.settings.fast_pred_var():
        residual_dist = likelihood(gp(gp_input))

    residual_mean = residual_dist.mean.reshape(bsz, pred_len, n_channels)
    residual_var = residual_dist.variance.reshape(bsz, pred_len, n_channels)

    mean = z + lam * residual_mean
    var = residual_var           # the tree forecast itself is deterministic
    return mean, var


# ---------------------------------------------------------------------------
# Coregionalisation matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_task_covariance(gp, likelihood):
    """
    B = A A^T from the LMC mixing coefficients, plus the likelihood task noises.

    With num_latents = 1 (ICM here) B is rank one by construction, so its
    eigenspectrum is uninformative -- refit with num_latents > 1 before reading
    regional structure into it.
    """
    A = gp.variational_strategy.lmc_coefficients.detach().cpu()   # [num_tasks, num_latents]
    B = (A @ A.T).numpy()
    d = np.sqrt(np.diag(B))
    corr = B / np.outer(d, d)

    _, noise = _read_noise(likelihood)
    noise = noise.numpy()

    return {
        "B": B,
        "correlation": corr,
        "task_noise": noise,
        "rank": int(A.shape[-1]),
        "trace_signal": float(np.trace(B)),
        "trace_noise": float(noise.sum()),
        "eigenvalues": np.linalg.eigvalsh(B)[::-1],
    }


# ---------------------------------------------------------------------------
# PICP / pinball
# ---------------------------------------------------------------------------

def picp(y_true, mean, std, confidence):
    z = norm.ppf(0.5 + confidence / 2)
    lower, upper = mean - z * std, mean + z * std
    return ((y_true >= lower) & (y_true <= upper)).float().mean().item()


def pinball_loss(y_true, mean, std, quantile):
    z = norm.ppf(quantile)
    q_pred = mean + z * std
    diff = y_true - q_pred
    return torch.maximum(quantile * diff, (quantile - 1) * diff).mean().item()