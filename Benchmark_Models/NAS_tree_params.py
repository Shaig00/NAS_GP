import sys
from pathlib import Path
# Parent of notebook current working directory
parent_dir = Path.cwd().parent
sys.path.append(str(Path.cwd().parent))
print(parent_dir)
import pickle
import pandas as pd
from Benchmark_Models.Build_NAS_models import default_build_model_fn, build_chain_model_fn
from pathlib import Path
parent_dir = str(Path.cwd().parent)

with open(str(parent_dir)+'/Results_Evaluation/top10_ensemble_configs.pkl', 'rb') as f:
    tree_configs = pickle.load(f)

with open(str(parent_dir)+'/Results_Evaluation/top10_ensemble_configs_linear.pkl', 'rb') as f:
    chain_configs = pickle.load(f)

def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

rows_tree = []
for i, cfg in enumerate(tree_configs):
    for seq_len in [48, 96, 192]:
        model = default_build_model_fn(cfg, 19, 19, seq_len, seq_len)
        rows_tree.append({'arm': 'tree', 'tree': i, 'horizon': seq_len,
                     'params': n_params(model)})

pd.DataFrame(rows_tree).to_pickle('tree_NAS_params.pkl')

rows_chain = []
for i, cfg in enumerate(chain_configs):
    for seq_len in [48, 96, 192]:
        model = build_chain_model_fn(cfg, 19, 19, seq_len, seq_len)
        rows_chain.append({'arm': 'chain', 'tree': i, 'horizon': seq_len,
                     'params': n_params(model)})

pd.DataFrame(rows_chain).to_pickle('chain_NAS_params.pkl')