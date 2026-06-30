

# save trained weights
import torch
from torchgen import model


trained_weight = model.output_leaf_projection.weight.data.clone()
trained_bias   = model.output_leaf_projection.bias.data.clone()

# replace with BayesLinear — same dimensions, weight transfer works
model.output_leaf_projection = bnn.BayesLinear(
    prior_mu=0,
    prior_sigma=0.1,
    in_features=num_leaves * max_features,
    out_features=output_channels  # still 20
)

# transfer weights
model.output_leaf_projection.weight_mu.data = trained_weight
model.output_leaf_projection.bias_mu.data   = trained_bias

# add quantile heads on top
self.q10_head = nn.Linear(output_channels, output_channels)  # (20 → 20)
self.q25_head = nn.Linear(output_channels, output_channels)
self.q50_head = nn.Linear(output_channels, output_channels)
self.q75_head = nn.Linear(output_channels, output_channels)
self.q90_head = nn.Linear(output_channels, output_channels)


def forward(self, x):
    # ... tree computation ...
    leaf_concat = torch.cat(leaf_outputs, dim=-1)  # (B, L, num_leaves × C)
    
    merged = self.output_leaf_projection(leaf_concat)  # BayesLinear → (B, L, 20)

    q10 = self.q10_head(merged)  # (B, L, 20)
    q25 = self.q25_head(merged)
    q50 = self.q50_head(merged)
    q75 = self.q75_head(merged)
    q90 = self.q90_head(merged)
    
    quantiles = torch.stack([q10, q25, q50, q75, q90], dim=-2)  # (B, L, 5, 20)
    quantiles, _ = torch.sort(quantiles, dim=-2)  # enforce ordering
    
    return quantiles  # q10, q25, q50, q75, q90


#LOSS Function

kl_loss   = bnn.BKLLoss(reduction='mean', last_layer_only=False)
# last_layer_only=False because BayesLinear is no longer the last layer

def total_loss(quantiles, target, model, beta=0.01):
    # quantiles: (B, L, 5, 20)
    # target: (B, L, 20)
    
    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    heads = [q10, q25, q50, q75, q90]

    pinball_total = sum(
        pinball_loss(pred, target, tau)
        for pred, tau in zip(heads, taus)
    )

    loss = pinball_total + 0.01 * kl_loss(model)


# MC Inference

model.train()
samples = torch.stack([model(x) for _ in range(100)])
# shape: (100, B, L, 5, 20)

q10 = samples[:,:,:,0,:].mean(dim=0)  # (B, L, 20)
q25 = samples[:,:,:,1,:].mean(dim=0)
q50 = samples[:,:,:,2,:].mean(dim=0)  # point estimate
q75 = samples[:,:,:,3,:].mean(dim=0)
q90 = samples[:,:,:,4,:].mean(dim=0)

# epistemic: spread of median across MC samples
epistemic = samples[:,:,:,2,:].std(dim=0)   # (B, L, 20)

# aleatoric: mean interval width across MC samples
aleatoric = (samples[:,:,:,4,:] - samples[:,:,:,0,:]).mean(dim=0)