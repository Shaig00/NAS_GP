from Neural_Architecture_Search.Framework import Custom_Torch_Model, TSMixer_ResBlock, MTSMixer_Model, Patching_Block, Dnonlinear_Model, create_args
# from Benchmark_Models. import 
import numpy as np
import  torch
import torch.nn as nn

from Neural_Architecture_Search.Framework_chain import Custom_Torch_Model as Custom_Chain_Model

def default_build_model_fn(tree_config, nr_inputs, nr_outputs, seq_len, pred_len, tau = 1.0, routing_mode='timestep'):
    """Build a Custom_Torch_Model (the NAS tree) from a sampled config dict.
    Mirrors the construction used during the NAS search itself, so a retrained
    top-10 tree is architecturally identical to what the search evaluated."""

    
    return Custom_Tree_Modelv2(
        input_channels=nr_inputs,
        output_channels=nr_outputs,
        seq_len=seq_len, pred_len=pred_len,
        nr_nodes=tree_config['nr_nodes'],
        ts_module_nodes=tree_config['ts_module_nodes'],
        revin_nodes=tree_config['revin_nodes'],
        dropout_nodes=tree_config['dropout_nodes'],
        activations_nodes=tree_config['activations_nodes'],
        # skip_connections=tree_config['skip_connections'],
        prediction_projections=tree_config['prediction_projections'],
        d_model_nodes=tree_config['d_model_nodes'],
        patch_len_nodes=tree_config['patch_len_nodes'],
        stride_nodes=tree_config['stride_nodes'],
        train_pe_nodes=tree_config['train_pe_nodes'],
        kernel_size_nodes=tree_config['kernel_size_nodes'],
        activations_second_nodes=tree_config['activations_second_nodes'],
        dff_nodes=tree_config['dff_nodes'],
        sampling_nodes=tree_config['sampling_nodes'],
        # leaves_operation='channel',
        gate_temperature = tau,
        routing_mode=routing_mode
        )


class Custom_Tree_Model(Custom_Torch_Model):
    def get_pooled_representation(self,x):
        activation = {}
        def hook(module, input, output):
            activation["pooled"] = input[0]
        handle = self.output_leaf_projection.register_forward_hook(hook)
        self(x)
        handle.remove()
        return activation['pooled']


class Custom_Tree_Modelv2(nn.Module):
    def __init__(self, input_channels,
                 output_channels,
                 seq_len,pred_len,
                 ts_module_nodes,
                 nr_nodes,
                 revin_nodes,prediction_projections,
                 activations_nodes,
                 activations_second_nodes,
                #  skip_connections,
                 dropout_nodes,
                 #forecast_sequence_difference,
                 d_model_nodes=None,
                #  n_heads_nodes=None,
                #  self_attention_type_nodes=None,
                 kernel_size_nodes=None,
                 patch_len_nodes=None,
                 stride_nodes=None,
                 train_pe_nodes=None,
                 sampling_nodes=None,
                 dff_nodes=None,
                 channels_ratio = 1,
                 fix_subsets = True,
                 operation = 'add', 
                 leaves_operation = 'concat',
                 merge_mode = 'linear',
                 gate_temperature=1.0,
                 routing_mode = 'timestep'):
        
        super(Custom_Tree_Modelv2,self).__init__()

        self.tree_depth = int(np.ceil(np.log2(nr_nodes+1)))
        self.num_leaves = 2 ** (self.tree_depth-1)
        self.nr_nodes = nr_nodes
        #Input dimensions=output dimensions, i.e., 
        #number of input timesteps per sequence=number of prediction timesteps
        #number of input channels=number of output channels
        self.input_channels = input_channels
        self.seq_len=seq_len
        self.max_features = int(self.input_channels*channels_ratio)
    
        # self.skip_connections=skip_connections
        self.prediction_projections=prediction_projections

        self.ts_module_nodes=ts_module_nodes
        #self.forecast_sequence_difference=forecast_sequence_difference

        
        #---------------------
        # fix subset and fix selected features
        self.subset_fixed = fix_subsets
        if self.subset_fixed:
            self.selected_features = torch.randperm(self.input_channels)[:self.max_features]
        #---------------------
        # latent variable merge operation
        self.operation = operation
        # leaves operation
        self.leaves_operation = leaves_operation
        #---------------------


        # Node split layers for each tree and depth
        self.path_proba_modules=[]
        self.ts_tree_modules=[]

        # self.node_components=[]

        self.routing_mode = routing_mode
        self.gate_temperature = float(gate_temperature)

        node_idx = 0
        for depth_idx in range(0,self.tree_depth):
            current_level_path_probas=[]
            current_level_ts_modules=[]

            for path_proba_idx in range(0,2**depth_idx):
                #Get path proba modules:
                if depth_idx<(self.tree_depth-1):
                    if routing_mode == 'channel':
                        gate = nn.Parameter(torch.zeros(self.max_features, 2))
                        current_level_path_probas.append(gate)
                    else:
                        current_level_path_probas.append(nn.Linear(self.max_features, 2))
                
                if self.ts_module_nodes[node_idx]=="TSMixer":
                    current_level_ts_modules.append(TSMixer_ResBlock(create_args(current_model='TSMixer',
                                                    input_channels=input_channels,nr_node=node_idx,seq_len=seq_len,
                                                    d_model=int(d_model_nodes['node_'+str(node_idx)]/(2**depth_idx)), # hidden dimension reduction based on level
                                                    revin=revin_nodes[node_idx],
                                                    dropout_rate=dropout_nodes[node_idx],
                                                    activation=activations_nodes[node_idx],
                                                    activation_second=activations_second_nodes['node_'+str(node_idx)])).float())
                elif self.ts_module_nodes[node_idx]=="MTSMixer":
                    current_level_ts_modules.append(MTSMixer_Model(create_args(current_model='MTSMixer',
                                                        input_channels=input_channels,nr_node=node_idx,seq_len=seq_len,
                                                        revin=revin_nodes[node_idx],
                                                        dropout_rate=dropout_nodes[node_idx],
                                                        d_model=int(d_model_nodes['node_'+str(node_idx)]/(2**depth_idx)), # hidden dim reduction
                                                        activation=activations_nodes[node_idx],
                                                        activation_second=activations_second_nodes['node_'+str(node_idx)],
                                                        dff=dff_nodes['node_'+str(node_idx)],
                                                        sampling=sampling_nodes['node_'+str(node_idx)])).float())
                    
                elif "Dnonlinear" in self.ts_module_nodes[node_idx]:
                    current_level_ts_modules.append(Dnonlinear_Model(create_args(current_model='Dnonlinear',
                                                        input_channels=input_channels,
                                                        nr_node=node_idx,seq_len=seq_len,
                                                        revin=revin_nodes[node_idx],
                                                        dropout_rate=dropout_nodes[node_idx],
                                                        decomposition_type=self.ts_module_nodes[node_idx].split('_')[1],#avg pooling or cnn
                                                        kernel_size=kernel_size_nodes['node_'+str(node_idx)],
                                                        activation=activations_nodes[node_idx],
                                                        activation_second=activations_second_nodes['node_'+str(node_idx)])).float())
                    
                else:#Patching
                    current_level_ts_modules.append(Patching_Block(create_args(current_model='Patching',
                                                                    input_channels=input_channels,
                                                                    nr_node=node_idx,seq_len=seq_len,
                                                                    revin=revin_nodes[node_idx],
                                                                    dropout_rate=dropout_nodes[node_idx],
                                                                    d_model=int(d_model_nodes['node_'+str(node_idx)]/(2**depth_idx)), # hidden dim reduction
                                                                    patch_len=patch_len_nodes['node_'+str(node_idx)],
                                                                    stride=stride_nodes['node_'+str(node_idx)],
                                                                    activation=activations_nodes[node_idx],
                                                                    train_pe=(train_pe_nodes['node_'+str(node_idx)] if train_pe_nodes!={} else False))).float())
                node_idx += 1
            if routing_mode == "channel":
                self.path_proba_modules.append(torch.nn.ParameterList(current_level_path_probas))
            else:
                self.path_proba_modules.append(torch.nn.ModuleList(current_level_path_probas))
            self.ts_tree_modules.append(torch.nn.ModuleList(current_level_ts_modules))

        # self.node_components=torch.nn.ModuleList( self.node_components)
        self.path_proba_modules=torch.nn.ModuleList(self.path_proba_modules)
        self.ts_tree_modules=torch.nn.ModuleList(self.ts_tree_modules)
        
        self.merge_mode = merge_mode
        if self.merge_mode == 'bnn':
            # self.output_leaf_projection = bnn.BayesLinear(prior_mu=0,
            #                                               prior_sigma=0.1,
            #                                               in_features=(self.num_leaves*self.max_features),
            #                                               out_features=output_channels)
            pass
        elif self.leaves_operation == 'channel':
            self.output_leaf_projection = nn.Parameter(torch.ones(output_channels, self.num_leaves)/ self.num_leaves)
        else:
            #To add Revin for output projection layers:
            self.output_leaf_projection = nn.Linear((self.num_leaves*self.max_features 
                                                        if self.leaves_operation=='concat' 
                                                        else self.max_features), output_channels, bias=False)
        
        if self.prediction_projections==True:
            self.pred_len_projection=nn.Linear(seq_len, pred_len)
            
            if not self.subset_fixed: # add ifelse shared
                self.concatenate_merge = nn.Linear(2*self.max_features, self.max_features)

                
            
    def forward(self, x):
        node_outputs=[]

        tree_outputs=[]
        previous_level_outputs={}
        if self.subset_fixed:
            selected_features = self.selected_features

        for tree_depth_idx in range(0,self.tree_depth):

            previous_level_outputs[tree_depth_idx]=[]
            current_level_ts_modules=self.ts_tree_modules[tree_depth_idx]

            #Each node at the previous level appends TWO entries: its left and its
            #right routed output.  So the list at level d-1 is indexed by BRANCH,
            #not by node, and holds 2^d entries -- exactly one per node at level d.
            #Node i therefore consumes entry i.  Halving the index, as a parent
            #lookup in a node-indexed tree would, sends both children of a node to
            #the same branch and orphans the other half of the tree: at depth 1
            #both nodes would read the root's left branch, and at depth 2 all four
            #leaves would descend from depth-1 node 0, leaving node 1 outside the
            #computation graph entirely.
            if tree_depth_idx > 0:
                assert len(previous_level_outputs[tree_depth_idx-1]) == len(current_level_ts_modules), (
                    f'depth {tree_depth_idx}: {len(current_level_ts_modules)} nodes but '
                    f'{len(previous_level_outputs[tree_depth_idx-1])} parent branches')

            for ts_module_idx in range(0,len(current_level_ts_modules)):
                # Select the current ts module:
                ts_module=current_level_ts_modules[ts_module_idx]

                # Select the subset of features in the current level:
                if not self.subset_fixed:
                    selected_features = torch.randperm(self.input_channels)[:self.max_features]

                # Add the information from the parent branch of the previous level:
                if tree_depth_idx > 0:
                    idx_parent_node=ts_module_idx

                if tree_depth_idx>0:
                    if not self.subset_fixed:
                        if self.operation == 'concat':
                            latent_x_subset=torch.concat((latent_x_subset, previous_level_outputs[tree_depth_idx-1][idx_parent_node]), dim=-1)
                            latent_x_subset = self.concatenate_merge(latent_x_subset)
                        elif self.operation == 'add':
                            latent_x_subset=latent_x_subset+previous_level_outputs[tree_depth_idx-1][idx_parent_node]
                        elif self.operation == 'avg':
                            latent_x_subset=(latent_x_subset+previous_level_outputs[tree_depth_idx-1][idx_parent_node])/2
                    else:
                        latent_x_subset = ts_module(previous_level_outputs[tree_depth_idx-1][idx_parent_node])

                else: # depth being 0
                    x_subset = x[:, :, selected_features]
                    # latent representation
                    latent_x_subset=ts_module(x_subset)

                if tree_depth_idx<self.tree_depth-1:#then prepare the softmax node splits for the next level:
                    gate=self.path_proba_modules[tree_depth_idx][ts_module_idx]

                    if getattr(self, 'routing_mode', 'timestep') == 'channel':
                        #A free [C, 2] parameter, softmaxed over the branch axis and
                        #broadcast across batch and time: each channel gets a fixed
                        #apportionment between the two branches, independent of the
                        #input.  Unlike the per-timestep gate it cannot be pulled
                        #toward an even split by averaging over timesteps.
                        channel_proba=torch.softmax(gate/self.gate_temperature, dim=-1)
                        left_proba=channel_proba[:,0].view(1,1,-1)
                        right_proba=channel_proba[:,1].view(1,1,-1)
                    else:
                        #Per-timestep routing: the softmax is applied here rather
                        #than inside the module so the logits can be divided by a
                        #temperature first.  gate_temperature = 1 recovers the
                        #original behaviour exactly.
                        gate_logits=gate(latent_x_subset)
                        softmax_proba=torch.softmax(gate_logits/self.gate_temperature, dim=-1)
                        left_proba=softmax_proba[:,:,0].unsqueeze(-1)
                        right_proba=softmax_proba[:,:,1].unsqueeze(-1)

                    if getattr(self, 'debug_gates', False):
                        p=(channel_proba[:,0] if getattr(self, 'routing_mode', 'timestep')=='channel'
                           else softmax_proba[:,:,0])
                        print(f'  depth {tree_depth_idx} node {ts_module_idx} | '
                              f'p: min {p.min().item():.3f} max {p.max().item():.3f} '
                              f'sd {p.std().item():.3f}')

                    left_latent_x_subset=left_proba*latent_x_subset
                    previous_level_outputs[tree_depth_idx].append(left_latent_x_subset)

                    right_latent_x_subset=right_proba*latent_x_subset
                    previous_level_outputs[tree_depth_idx].append(right_latent_x_subset)

                else:
                    tree_outputs.append(latent_x_subset)

                node_outputs.append(latent_x_subset)

        # Merge tree leaves with specified leaves operation:
        if self.merge_mode == 'bnn':    # bnn bypasses the leaves operation
            tree_outputs=torch.concat(tree_outputs,axis=-1)
            tree_outputs=self.output_leaf_projection(tree_outputs)

        else:
            if self.leaves_operation == 'channel':
                #Per-channel leaf mixture: each output channel learns its own
                #weights over the leaves, using only that channel's value in each
                #leaf.  C*L parameters rather than C*(L*C), and channel identity is
                #preserved by construction.
                tree_outputs=torch.stack(tree_outputs, dim=-1)          # [B, T, C, L]
                tree_outputs=(tree_outputs*self.output_leaf_projection).sum(dim=-1)
            elif self.leaves_operation == 'concat':
                tree_outputs=torch.concat(tree_outputs,axis=-1)
                tree_outputs=self.output_leaf_projection(tree_outputs)
            elif self.leaves_operation == 'add':
                tree_outputs=torch.stack(tree_outputs,axis=1)
                tree_outputs=torch.sum(tree_outputs, axis=1)
            elif self.leaves_operation == 'avg':
                tree_outputs=torch.stack(tree_outputs,axis=1)
                tree_outputs=torch.mean(tree_outputs, axis=1)

        # PREDICTION LAYER
        if self.prediction_projections==True:
            output_timesteps=self.pred_len_projection(tree_outputs.permute(0,2,1))
            output=output_timesteps.permute(0,2,1)
        else:
            output=tree_outputs

        return output

    
    def get_pooled_representation(self,x):
            activation = {}
            def hook(module, input, output):
                activation["pooled"] = input[0]
            handle = self.output_leaf_projection.register_forward_hook(hook)
            self(x)
            handle.remove()
            return activation['pooled']

    


#---------------------------------------------
#           CUSTOM TREE END
#---------------------------------------------

def build_chain_model_fn(chain_config, nr_inputs, nr_outputs, seq_len, pred_len):

    return Custom_Chain_Model(
        input_channels=nr_inputs,
        output_channels=nr_outputs,
        seq_len=seq_len, pred_len=pred_len,
        nr_nodes=chain_config['nr_nodes'],
        ts_module_nodes=chain_config['ts_module_nodes'],
        revin_nodes=chain_config['revin_nodes'],
        dropout_nodes=chain_config['dropout_nodes'],
        activations_nodes=chain_config['activations_nodes'],
        skip_connections=chain_config['skip_connections'],
        prediction_projections=chain_config['prediction_projections'],
        d_model_nodes=chain_config['d_model_nodes'],
        patch_len_nodes=chain_config['patch_len_nodes'],
        stride_nodes=chain_config['stride_nodes'],
        train_pe_nodes=chain_config['train_pe_nodes'],
        kernel_size_nodes=chain_config['kernel_size_nodes'],
        activations_second_nodes=chain_config['activations_second_nodes'],
        dff_nodes=chain_config['dff_nodes'],
        sampling_nodes=chain_config['sampling_nodes'],
        )