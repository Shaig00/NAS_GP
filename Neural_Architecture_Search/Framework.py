import sys
sys.path.append("..")

from simple_colors  import *
import warnings
from functools import wraps
import gc
import argparse
import sys
from collections import namedtuple
import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
from permetrics import RegressionMetric
import torch
from torch import nn
import torchbnn as bnn
import os
import argparse
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.distributions import Categorical
from dask.distributed import Client
        

#NAS-Framework: calls the classes defined below
class NAS_Framework(nn.Module):
    """
    A class implementing both the actor and the critic in one neural network model:
    
    
    Parameters:
    ----------
    embedding_dimension : int
        Dimensionality of continuous vectors representing the components of the search space
    actor_head_shapes : list of int values
        List of integer values corresponding to the number of possible values each actor head samples
    shared_units : int
        The number of neurons in the shared LSTM-cell(s) part of the actor-critic model.
    search_space_values : dict
        The mapping between the components in the search space (the keys) and the possible values to
        sample from (the values of the dictionary).
    
    
    Attributes:
    (in addition to all Parameters)
    ----------
    
    embedding_dimension=embedding_dimension
    max_number_episodes=max_number_episodes
    timesteps_per_episode=timesteps_per_episode
    ac_units=ac_units
    ac_lr=ac_lr

    search_space={}
    embedding_matrix={}
    embedding_none_matrix={}
    min_max_deltas={}
    actor_head_shapes=[]
    history_dict={}
    all_sampled_cells=[]
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
    device=_acquire_device(use_gpu=True)
    
    

    Methods
    -------
    create_search_space():
        Creates a categorical, high-dimensional search space.

    run_macro_nas():
        Performs macro NAS.
    
    
    """
    def __init__(self,
                embedding_dimension:int,
                max_number_episodes:int,
                timesteps_per_episode:int,
                ac_units:int,
                ac_lr:float):

        super(NAS_Framework, self).__init__()

        self.embedding_dimension=embedding_dimension
        self.search_space={}
        self.embedding_matrix={}
        self.embedding_none_matrix={}
        self.min_max_deltas={}
        self.actor_head_shapes=[]
        self.history_dict={}
        self.all_sampled_cells=[]

        
        #Set format for saving actions & critic values:
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.max_number_episodes=max_number_episodes
        self.timesteps_per_episode=timesteps_per_episode
        self.device=_acquire_device(use_gpu=True)
        self.ac_units=ac_units
        self.ac_lr=ac_lr
        
        

    def create_search_space(self,print_components=False):
        """Creates the categorical search space together with the continuous embeddings of each categorical component.

        Parameters
        ----------
        print_components: bool
        Whether to print an overview of all NAS macro network components together with their values, once the search space has been created.
            
        Returns
        -------
        None    
        """
        # layer of a tree
        list_layers = list(np.arange(2,4,1))

        self.search_space['nr_nodes']= [np.sum(2**(np.arange(1,nr_layers+1,1) - 1)) 
                                        for nr_layers in list_layers]
        self.embedding_matrix['nr_nodes']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['nr_nodes']),self.embedding_dimension])
        max_nr_nodes=self.search_space['nr_nodes'][-1]

        for node_idx in range(0,max_nr_nodes):
            self.search_space['node_'+str(node_idx)+'_ts_module']=["Dnonlinear_avgpool1d","Dnonlinear_conv1d","TSMixer","MTSMixer","Patching"]
            self.embedding_matrix['node_'+str(node_idx)+'_ts_module']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_ts_module']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_ts_module']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            
            self.search_space['node_'+str(node_idx)+'_kernel_size']=list(np.arange(3,39,2))
            self.embedding_matrix['node_'+str(node_idx)+'_kernel_size']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_kernel_size']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_kernel_size']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            
            self.search_space['node_'+str(node_idx)+'_d_model']=list(np.arange(200,6200,200))
            self.embedding_matrix['node_'+str(node_idx)+'_d_model']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_d_model']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_d_model']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])
            
            

            self.search_space['node_'+str(node_idx)+'_patch_len']=list(np.arange(8,44,4))
            self.embedding_matrix['node_'+str(node_idx)+'_patch_len']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_patch_len']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_patch_len']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            self.search_space['node_'+str(node_idx)+'_stride']=list(np.arange(4,22,2))
            self.embedding_matrix['node_'+str(node_idx)+'_stride']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_stride']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_stride']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            
            self.search_space['node_'+str(node_idx)+'_dff']=[2, 3, 4, 6]
            self.embedding_matrix['node_'+str(node_idx)+'_dff']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_dff']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_dff']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            self.search_space['node_'+str(node_idx)+'_sampling']=[2, 3, 4, 6, 8, 12]
            self.embedding_matrix['node_'+str(node_idx)+'_sampling']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_sampling']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_sampling']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            
            self.search_space['node_'+str(node_idx)+'_dropout']=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
            self.embedding_matrix['node_'+str(node_idx)+'_dropout']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_dropout']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_dropout']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])
            
            
            #First Activation Function:#########################################################################################################
            self.search_space['node_'+str(node_idx)+'_activation_1.component']=['relu','elu','mish','silu',"sine","cosine",'exp','erfcsoftplus',
                                                                                'leakysinelu','leakycosinelu',
                                                                                'tanh','sigmoid','gelu','linear']
            self.embedding_matrix['node_'+str(node_idx)+'_activation_1.component']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_activation_1.component']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_activation_1.component']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            self.search_space['node_'+str(node_idx)+'_activation_action']=['multiply','add','average',None]
            self.embedding_matrix['node_'+str(node_idx)+'_activation_action']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_activation_action']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_activation_action']=self.embedding_matrix['node_'+str(node_idx)+'_activation_action'][:,-1,:]
            
            self.search_space['node_'+str(node_idx)+'_activation_2.component']=['relu','elu','mish','silu',"sine","cosine",'exp','erfcsoftplus',
                                                                                'leakysinelu','leakycosinelu',
                                                                                'tanh','sigmoid','gelu','linear']
            self.embedding_matrix['node_'+str(node_idx)+'_activation_2.component']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_activation_2.component'])-1,self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_activation_2.component']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])
            #####################################################################################################################################

            
            
            #Second Activation Function:#########################################################################################################
            self.search_space['node_'+str(node_idx)+'_activation_second_1.component']=['relu','elu','mish','silu',"sine","cosine",'exp','erfcsoftplus',
                                                                            'leakysinelu','leakycosinelu',
                                                                            'tanh','sigmoid','gelu','linear']
            self.embedding_matrix['node_'+str(node_idx)+'_activation_second_1.component']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_activation_second_1.component']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_activation_second_1.component']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])

            self.search_space['node_'+str(node_idx)+'_activation_second_action']=['multiply','add','average',None]
            self.embedding_matrix['node_'+str(node_idx)+'_activation_second_action']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_activation_second_action']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_activation_second_action']=self.embedding_matrix['node_'+str(node_idx)+'_activation_second_action'][:,-1,:]

            self.search_space['node_'+str(node_idx)+'_activation_second_2.component']=['relu','elu','mish','silu',"sine","cosine",'exp','erfcsoftplus',
                                                                            'leakysinelu','leakycosinelu',
                                                                            'tanh','sigmoid','gelu','linear']
            self.embedding_matrix['node_'+str(node_idx)+'_activation_second_2.component']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(node_idx)+'_activation_second_2.component'])-1,self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(node_idx)+'_activation_second_2.component']=torch.normal(mean=0.0,std=1.0,size=[1,self.embedding_dimension])
            #####################################################################################################################################


        for skip_idx in range(1,self.search_space['nr_nodes'][-1]):
            # find depth
            node_depth = np.floor(np.log2(skip_idx+1))
            # get all nodes from previous levels
            self.search_space['node_'+str(skip_idx)+'_skip_connection']=list(np.arange(0,2**int(node_depth)-1,1))
            self.search_space['node_'+str(skip_idx)+'_skip_connection'].append(None)
            self.embedding_matrix['node_'+str(skip_idx)+'_skip_connection']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['node_'+str(skip_idx)+'_skip_connection']),self.embedding_dimension])
            self.embedding_none_matrix['node_'+str(skip_idx)+'_skip_connection']=self.embedding_matrix['node_'+str(skip_idx)+'_skip_connection'][:,-1,:]


        self.search_space['prediction_projections']=[True, False] 
        self.embedding_matrix['prediction_projections']=torch.normal(mean=0.0,std=1.0,size=[1,len(self.search_space['prediction_projections']),self.embedding_dimension])

        #Print the search space components if corresponding boolean condition evaluated to True:
        if print_components==True:
            for ss_key in self.search_space.keys():
                print(black(ss_key,['bold']),': ',self.search_space[ss_key],'\n')

        #Based on the initialized values of the search space components, fill in the min and max values relevant for the computation of GED:
        self.min_max_deltas['dropout']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_dropout']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_dropout'])}

        self.min_max_deltas['patch_len']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_patch_len']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_patch_len'])}
        self.min_max_deltas['stride']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_stride']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_stride'])}

        self.min_max_deltas['dff']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_dff']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_dff'])}


        self.min_max_deltas['sampling']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_sampling']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_sampling'])}

        self.min_max_deltas['kernel_size']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_kernel_size']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_kernel_size'])}
        self.min_max_deltas['d_model_nodes']={'min_value':np.min(self.search_space['node_'+str(node_idx)+'_d_model']),
                                'max_value':np.max(self.search_space['node_'+str(node_idx)+'_d_model'])}

        
        #Save embedding dimension per macro network component:
        for key in self.search_space.keys():
            self.actor_head_shapes.append(self.embedding_matrix[key].shape[1])

    def run_macro_nas(self,entropy_baseline,save_sampling_history):
        actor_critic_model=Policy(embedding_dimension=self.embedding_dimension,
                    actor_head_shapes=self.actor_head_shapes,
                    shared_units=self.ac_units,
                    search_space_values=self.search_space,
                    embedding_matrix_values=self.embedding_matrix,
                    embedding_none_matrix_values=self.embedding_none_matrix)

        #Initialize further variables relevant for the backward pass of the actor critic
        optimizer_actor_critic = torch.optim.Adam(actor_critic_model.parameters(), lr=self.ac_lr)
        eps = np.finfo(np.float32).eps.item()
        
        gamma = 0.99
        current_episode = 0
        dataloader_name='EU_Energy_Prices_Dataloader_torch_'
        reverse_rescale=False
        nr_inputs=(15 if 'Energy_Data_Dataloader_torch' in dataloader_name else 19)
        nr_targets=(15 if 'Energy_Data_Dataloader_torch' in dataloader_name else 19)
        input_seq_len=np.sum([48,96,192])
        edit_distance_alpha=10.0
        actor_critic_model.train()

        keep_trainig_rl=True
        while keep_trainig_rl:  # Run until solved
            #print('GPU Memory Usage & Processes Overview: ')
            #!nvidia-smi
            
            episode_reward = 0
            print(black('STARING A NEW EPISODE: ',['bold']),current_episode)
            cells_current_episode=[]
            
            cells_edit_distance_current_episode=[]
            cells_entropy_current_episode=[]
            embedded_cells_current_episode=[]
            memory_states_current_episode=[]
            episode_timestep_looping=True
            episode_timestep=0
            
            
            while episode_timestep_looping==True:
                #Randomly initialize the search :
                if current_episode==0 and episode_timestep==0:
                    episode_start_cell,embedded_representation_rs_episode_start=random_sampling_model(search_space=self.search_space,
                                                            embedding_matrix=self.embedding_matrix,
                                                            embedding_none_matrix=self.embedding_none_matrix)
                    
                    cell_entropy,cell_probabilities,actor_critic_sampled_network,environment_memory_state,embedded_representation_predicted_cell,log_prob_sum,critic_value_function=actor_critic_model(current_state_environment=embedded_representation_rs_episode_start)
                else:
                    if episode_timestep==0:
                        if np.random.random()>=0.7:#only in 30 of the cases
                            episode_start_cell,embedded_representation_rs_episode_start=random_sampling_model(search_space=self.search_space,
                                                                                                        embedding_matrix=self.embedding_matrix,
                                                                                                        embedding_none_matrix=self.embedding_none_matrix)
                            cell_entropy,cell_probabilities,actor_critic_sampled_network,environment_memory_state,embedded_representation_predicted_cell,log_prob_sum,critic_value_function=actor_critic_model(current_state_environment=embedded_representation_rs_episode_start,
                                                                                                        previous_memory_state=None)
                        else:#condition episode sampling on previously best perfomring cell
                            cell_entropy,cell_probabilities,actor_critic_sampled_network,environment_memory_state,embedded_representation_predicted_cell,log_prob_sum,critic_value_function=actor_critic_model(current_state_environment=best_embedded_representation_episode_start,
                                                                                                        previous_memory_state=None)
                    else:#if current timestep>0: take previously embedded representation stored in embedded cells current episode 
                        #the last embedding, which has passed the if else condition below:
                        cell_entropy,cell_probabilities,actor_critic_sampled_network,environment_memory_state,embedded_representation_predicted_cell,log_prob_sum,critic_value_function=actor_critic_model(current_state_environment=embedded_cells_current_episode[-1],
                                                                                                        previous_memory_state=memory_states_current_episode[-1])
                
                #Check if currently sampled cell is already in the list of visited NAS architectures:
                if (len(self.all_sampled_cells)==0) or (actor_critic_sampled_network not in self.all_sampled_cells):
                    
                    #Keep track of all sampled cells for the if-else condition accepting cells, which
                    #have never been visited before:
                    self.all_sampled_cells.append(actor_critic_sampled_network)
                    
                    #Save the embedded representation of the accepted cells and the corresponding
                    #memory state produced by the controller network: both will be used within the
                    #same episode at the beginning of the next timestep, but will be reset at the
                    #beginning of a new episode
                    embedded_cells_current_episode.append(embedded_representation_predicted_cell)
                    memory_states_current_episode.append(environment_memory_state)
                    
                    #Save corresponding log-prob sum of the accepted cell and the critic network's value,
                    #which will be used in the backward pass:
                    actor_critic_model.saved_actions.append(self.SavedAction(log_prob_sum,critic_value_function))

                    #Save the accepted cell in the current episode's sampled & accepted cells array:
                    cells_current_episode.append(actor_critic_sampled_network)
                    cells_entropy_current_episode.append(cell_entropy)
                    
                    #Compute edit distance:
                    if episode_timestep>1:
                        avg_episode_cell_edits_so_far=[]
                        for previous_episode_timestep in np.arange(0,episode_timestep,1):
                            current_edit_distance=compute_nas_edit_distance(first_sampled_cell=cells_current_episode[-1],second_sampled_cell=cells_current_episode[previous_episode_timestep],
                                                                            input_seq_len=input_seq_len,
                                                                            input_channels=nr_inputs,
                                                                            min_max_deltas=self.min_max_deltas)

                            avg_episode_cell_edits_so_far.append(current_edit_distance)     
                        
                        cells_edit_distance_current_episode.append(np.mean(avg_episode_cell_edits_so_far))
                        
                    elif episode_timestep==1:
                        current_edit_distance=compute_nas_edit_distance(first_sampled_cell=cells_current_episode[-1],second_sampled_cell=cells_current_episode[-2],
                                                                            input_seq_len=input_seq_len,
                                                                            input_channels=nr_inputs,
                                                                            min_max_deltas=self.min_max_deltas)
                        
                        cells_edit_distance_current_episode.append(current_edit_distance)
                    else:#current episode timestep==0:
                        current_edit_distance=compute_nas_edit_distance(first_sampled_cell=cells_current_episode[-1],second_sampled_cell=episode_start_cell,
                                                                            input_seq_len=input_seq_len,
                                                                            input_channels=nr_inputs,
                                                                            min_max_deltas=self.min_max_deltas)
                        cells_edit_distance_current_episode.append(current_edit_distance)
                    
                        
                    #Compute average probability of sampled cell: to be used as a callback for NAS,
                    # if avg. cell proba>0.9, then stop the search process:
                    avg_cell_proba=np.array([cell_proba.detach().numpy() for cell_proba in cell_probabilities]).mean()
                    print(black('Average Proba of '+str(episode_timestep+1)+'.sampled macro network: ',['bold']),avg_cell_proba,', ', black('with the architecture: ',['bold']))
                    for cell_component in actor_critic_sampled_network.keys():
                        print('- ',black(cell_component,['bold']),': ',actor_critic_sampled_network[cell_component],'\n')
                    print('\n')

                    if 'AvgNetworkProba_Episode_'+str(current_episode) in self.history_dict.keys():
                        self.history_dict['AvgNetworkProba_Episode_'+str(current_episode)].append(avg_cell_proba)
                    else:
                        self.history_dict['AvgNetworkProba_Episode_'+str(current_episode)]=[avg_cell_proba]
                    
                    
                    episode_timestep=episode_timestep+1
                    if episode_timestep==self.timesteps_per_episode:#
                        #then break episode timestep looping
                        episode_timestep_looping=False
                else:
                    continue    
                                
            
            client=Client(n_workers=5)
            print(black('Dask Client Dashboard Link: ',['bold']),client.dashboard_link)
            cells_chunks=[[cells_current_episode[0],cells_current_episode[1],cells_current_episode[2],cells_current_episode[3],cells_current_episode[4]],
                        [cells_current_episode[5],cells_current_episode[6],cells_current_episode[7],cells_current_episode[8],cells_current_episode[9] ] ]

            rewards_cells_history=[]
            for cells_chunk in cells_chunks:
                futures = client.map(environment_step,cells_chunk,
                                special_parameters={
                                        'nr_inputs':nr_inputs,
                                        'nr_targets':nr_targets,
                                        'epochs':50,
                                        'dataloader_name':dataloader_name,
                                        'use_amp':False,
                                        'checkpoints':'./checkpoints/',
                                        'train_test_split':'Train_Test1',
                                        'reverse_rescale':reverse_rescale,
                                        'device':self.device,
                                        'use_gpu':True})

                rewards_cells_history_chunk = client.gather(futures)
                for chunk_res_idx in range(0,len(rewards_cells_history_chunk)):
                    rewards_cells_history.append(rewards_cells_history_chunk[chunk_res_idx])
                del futures
                

            rewards_history=[rw_cl[0] for rw_cl in rewards_cells_history]
            rewards_second_subset_history=[rw_cl[1] for rw_cl in rewards_cells_history]
            both_rewards=np.array(rewards_history)+np.array(rewards_second_subset_history)
            rewards_training_testing_time=[rw_cl[3] for rw_cl in rewards_cells_history]
                
            if entropy_baseline==True:
                adj_rewards_history=both_rewards
                adj_rewards_nasdist_history=[adj_rewards_history[idx]+(edit_distance_alpha*cells_entropy_current_episode[idx]) for idx in range(0,len(adj_rewards_history))]
            
            else:
                adj_rewards_history=[]
                for rw_idx in range(0,len(rewards_history)):
                    if rewards_history[rw_idx]>rewards_second_subset_history[rw_idx]:#then sign of overfitting
                        rw_diff=2*(rewards_history[rw_idx]-rewards_second_subset_history[rw_idx])
                        adj_reward_diff=both_rewards[rw_idx]-rw_diff
                        adj_rewards_history.append((adj_reward_diff if adj_reward_diff>=0.0 else 0.0))
                    else:
                        adj_rewards_history.append(both_rewards[rw_idx])
                        
                #Add to the WV-penalized reward signal the 10-fold scaled GED component: 
                adj_rewards_nasdist_history=[adj_rewards_history[idx]+(edit_distance_alpha*cells_edit_distance_current_episode[idx]) for idx in range(0,len(adj_rewards_history))]
                
            #Add the edit distance terms:
            cells_history=[rw_cl[2] for rw_cl in rewards_cells_history]

            self.history_dict['Rewards_Episode_'+str(current_episode)]=rewards_history
            self.history_dict['Rewards_SecondSubset_Episode_'+str(current_episode)]=rewards_second_subset_history
            
            print(black('Summary current episode: ',['bold']))
            print(black('Rewards first validation subset: ',['bold']))
            print(np.around(rewards_history,5))
            print(black('Rewards second validation subset: ',['bold']))
            print(np.around(rewards_second_subset_history,5))
            print(black('Training testing time: ',['bold']))
            print(np.around(rewards_training_testing_time,5))
            
            print(black('Graph edit distance: ',['bold']))
            print(np.around(cells_edit_distance_current_episode,5))
            
            print(black('Entropy: ',['bold']))
            print(np.around(cells_entropy_current_episode,5))
            
            print(black('Adjusted rewards incl. edit distance for controller: ',['bold']))
            print(np.around(adj_rewards_nasdist_history,5),'\n')
            
            self.history_dict['Sampled_Networks_Episode_'+str(current_episode)]=cells_history
            
            self.history_dict['Sampled_Networks_Episode_Entropy_'+str(current_episode)]=cells_entropy_current_episode
            self.history_dict['Sampled_Networks_Episode_GED_'+str(current_episode)]=cells_edit_distance_current_episode
            self.history_dict['Sampled_networks_Episode_Runtime_'+str(current_episode)]=rewards_training_testing_time
            
            finish_episode(actor_critic_model=actor_critic_model,
                        optimizer_actor_critic=optimizer_actor_critic,
                        gamma=gamma,rewards_history=adj_rewards_nasdist_history)
                        
            #Initialize the embedding of the cell used as the input state from the previous timestep: 
            # set to the embeddding of the best performing cell sampled so far:
            if current_episode==0:
                idx_best_cell=np.argmax(adj_rewards_history)
                best_reward_so_far=adj_rewards_history[idx_best_cell]
                #Set the best embedding based on sampled cells in the first episode:
                best_embedding_so_far=embedded_cells_current_episode[idx_best_cell]
                best_embedded_representation_episode_start=best_embedding_so_far
                
                best_cell_so_far=cells_history[idx_best_cell]
                episode_start_cell=best_cell_so_far
            else:
                idx_best_cell=np.argmax(adj_rewards_history)
                if adj_rewards_history[idx_best_cell]>=best_reward_so_far:
                    
                    #Overwrite best reward so far and best embedding:
                    best_reward_so_far=adj_rewards_history[idx_best_cell]    
                    best_embedding_so_far=embedded_cells_current_episode[idx_best_cell]
                    best_embedded_representation_episode_start=best_embedding_so_far
                    
                    best_cell_so_far=cells_history[idx_best_cell]
                    episode_start_cell=best_cell_so_far
                else:
                    #For the initial state at the next episode take previously sampled best embedding
                    best_embedded_representation_episode_start=best_embedding_so_far
                    episode_start_cell=best_cell_so_far

            
            if save_sampling_history==True:

                if entropy_baseline==True:#NAS with entropy-based reward signal:
                    with open('Results_Evaluation/Entropy_NAS_Sampling_History.pkl','wb') as f:
                        pickle.dump([self.history_dict,actor_critic_model,optimizer_actor_critic,
                                    self.embedding_matrix,self.embedding_none_matrix,current_episode,
                                    best_embedded_representation_episode_start,episode_start_cell,
                                    best_reward_so_far],f)
                        
                else:#out approach, i.e., WV-GED NAS:
                    with open('Results_Evaluation/WV_GED_NAS_Sampling_History.pkl','wb') as f:
                        pickle.dump([self.history_dict,actor_critic_model,optimizer_actor_critic,
                                    self.embedding_matrix,self.embedding_none_matrix,current_episode,
                                    best_embedded_representation_episode_start,episode_start_cell,
                                    best_reward_so_far],f)
            
            #Close Dask Client:
            dask_client_cluster_open=True
            while dask_client_cluster_open==True:
                try:
                    client.close()
                    dask_client_cluster_open=False
                except:
                    print('Try again to shutdown and close Dask Client and Cluster!')
            
            #Once Dask Client completely closed:
            # also delete client and clister objects from current episode of AC 
            del client
            del adj_rewards_nasdist_history
            del cells_history
            del rewards_cells_history
            del cells_edit_distance_current_episode
            del embedded_cells_current_episode
            del memory_states_current_episode
            

            if np.mean(self.history_dict['AvgNetworkProba_Episode_'+str(current_episode)])>=0.90:  # Condition to consider the task solved
                print("Actor Critic Probabilities converged (Avg. Cell Proba>=90.00%)!")
                break

            current_episode += 1
            if current_episode>=self.max_number_episodes:
                print("Reached max number of episodes !")
                break




#Search Space Components:
class Torch_Cos(nn.Module):
    def __init__(self):
        super(Torch_Cos, self).__init__()
    def forward(self, x):
        cos_x=torch.cos(x)
        return cos_x
    
class Torch_Sin(nn.Module):
    def __init__(self):
        super(Torch_Sin, self).__init__()

    def forward(self, x):
        sin_x=torch.sin(x)
        return sin_x
    
#https://arxiv.org/pdf/2412.09889
class LeakySineLU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        return torch.where(x>0.0,torch.sin(x) ** 2 + x, 0.5 * (torch.sin(x) ** 2 + x))


class LeakyCosineLU(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, x):
        return torch.where(x>0.0,torch.cos(x) ** 2 + x, 0.5 * (torch.cos(x) ** 2 + x))

    
class Torch_Exp(nn.Module):
    def __init__(self):
        super(Torch_Exp, self).__init__()

    def forward(self, x):
        exp_x=torch.exp(x)
        return exp_x
    
    
class Torch_ErfcSoftplus(nn.Module):
    def __init__(self):
        super(Torch_ErfcSoftplus, self).__init__()

    def forward(self, x):
        erfcsoftplus_x=torch.square(torch.erfc(torch.nn.Softplus()(x)))
        return erfcsoftplus_x

def get_torch_activation(activation):
    
    if activation=='relu':
        return torch.nn.ReLU()
    
    elif activation=='gelu':
        return torch.nn.GELU()
    
    elif activation=='elu':
        return torch.nn.ELU()
    
    elif activation=='sine':
        return Torch_Sin()
    
    elif activation=='cosine':
        return Torch_Cos()
    
    elif activation=='tanh':
        return torch.nn.Tanh()
    
    elif activation=='sigmoid':
        return torch.nn.Sigmoid()
    
    elif activation=='exp':
        return Torch_Exp()
    
    elif activation=='erfcsoftplus':
        return Torch_ErfcSoftplus()

    elif activation=="leakysinelu":
        return LeakySineLU() 
        
    elif activation=="leakycosinelu":
        return LeakyCosineLU()
    
    elif activation=='linear':
        return torch.nn.Identity()
    
    elif activation=='mish':
        return torch.nn.Mish()
    
    else:#activation=='silu'
        return torch.nn.SiLU()


def create_args(current_model,input_channels,
                nr_node,seq_len,
                activation,dropout_rate,
                revin,
                self_attention_type=None,
                activation_second=None,
                kernel_size=None,
                decomposition_type=None,
                d_model=None,
                n_heads=None,
                patch_len=None,
                stride=None,
                train_pe=None,
                dff=None,
                sampling=None):
    
    #Parser TSMixer: ########################################################################################
    parser = argparse.ArgumentParser(description=current_model+'_'+str(nr_node))
    
    if "Dnonlinear" in current_model:
        parser.add_argument('--moving_avg', type=int, default=kernel_size, help='window size of moving average')
        parser.add_argument('--decomposition_type', type=str, default=decomposition_type, help='decomposition_type: avg pooling or cnn')
        
    if current_model in ["TSMixer","MTSMixer","Attention","Patching"]:
        parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
        
        if current_model=="MTSMixer":
            parser.add_argument('--individual', action='store_true', default=False, help='Linear layer for each variate(channel) individually')
            parser.add_argument('--norm', action='store_false', default=True, help='whether to apply LayerNorm')
            parser.add_argument('--sampling', type=int, default=sampling, help='the number of downsampling in factorized temporal interaction')
            parser.add_argument('--d_ff', type=int, default=int(input_channels/dff), help='dimension of fcn')
            #parser.add_argument('--fac_T', action='store_true', default=True, help='whether to apply factorized temporal interaction')
            #parser.add_argument('--fac_C', action='store_true', default=True, help='whether to apply factorized channel interaction')

        if current_model=="Attention":
            parser.add_argument('--n_heads', type=int, default=n_heads, help='the attention heads')
            parser.add_argument('--self_attention_type', type=str, default=self_attention_type, help='type of self-attention to apply')
            parser.add_argument('--pe', type=str, default='zeros', help='positional embeddings')
            parser.add_argument('--train_pe', type=bool, default=True, help='learn the positional embeddings')
            
        if current_model=="Patching":
            parser.add_argument('--patch_len', type=int, default=patch_len, help='length of patches')
            parser.add_argument('--stride', type=int, default=stride, help='stride for building patches')
            parser.add_argument('--train_pe', type=bool, default=train_pe, help='train positional embeddings for patches')
            parser.add_argument('--pe', type=str, default='zeros', help='type of positional embeddings if trained')
            parser.add_argument('--dropout_rate_pe', type=float, default=0.1, help='dropout rate for positional embeddings if trained')
            parser.add_argument('--padding_patch', type=str, default='end', help='padding patch')
            
            
    # Arguments relevant for all models:
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=seq_len, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=input_channels, help='encoder input size')
    parser.add_argument('--dropout_rate', type=float, default=dropout_rate, help='dropout')
    parser.add_argument('--activation', type=str, default=activation, help='activation')
    
    if current_model not in ["Attention","Patching"]:
        parser.add_argument('--activation_second', type=str, default=activation_second, help='activation_second')
    
    parser.add_argument('--revin', type=bool, default=revin, help='revin_layer')
    
    args = parser.parse_args()
    
    return args


#Revin layers:
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


#TSMixer:
class TSMixer_ResBlock(nn.Module):
    def __init__(self, configs):
        super(TSMixer_ResBlock, self).__init__()
        self.configs=configs
        if self.configs.revin==True:
            self.norm_denorm_layer=RevIN(num_features=self.configs.enc_in)
        
        #Temporal/Timesteps Dimension:
        if '_' not in configs.activation:
            self.temporal = nn.Sequential(
                nn.Linear(configs.seq_len, configs.d_model),
                get_torch_activation(activation=configs.activation),
                nn.Linear(configs.d_model, configs.seq_len),
                nn.Dropout(configs.dropout_rate))
        else:
            self.temporal_gated_activation_first,self.action_temporal_gate,self.temporal_gated_activation_second=configs.activation.split('_')
            self.temporal_gate_first = nn.Sequential(
                nn.Linear(configs.seq_len, configs.d_model),
                get_torch_activation(activation=self.temporal_gated_activation_first),
                nn.Linear(configs.d_model, configs.seq_len),
                nn.Dropout(configs.dropout_rate))
            self.temporal_gate_second = nn.Sequential(
                nn.Linear(configs.seq_len, configs.d_model),
                get_torch_activation(activation=self.temporal_gated_activation_second),
                nn.Linear(configs.d_model, configs.seq_len),
                nn.Dropout(configs.dropout_rate))

        #Channels Dimension:
        if '_' not in configs.activation_second:
            self.channel = nn.Sequential(
                nn.Linear(configs.enc_in, configs.d_model),
                get_torch_activation(activation=configs.activation_second),
                nn.Linear(configs.d_model, configs.enc_in),
                nn.Dropout(configs.dropout_rate))
        else:
            self.channel_gated_activation_first,self.action_channel_gate,self.channel_gated_activation_second=configs.activation_second.split('_')
            self.channel_gate_first = nn.Sequential(
                nn.Linear(configs.enc_in, configs.d_model),
                get_torch_activation(activation=self.channel_gated_activation_first),
                nn.Linear(configs.d_model, configs.enc_in),
                nn.Dropout(configs.dropout_rate))
            
            self.channel_gate_second = nn.Sequential(
                nn.Linear(configs.enc_in, configs.d_model),
                get_torch_activation(activation=self.channel_gated_activation_second),
                nn.Linear(configs.d_model, configs.enc_in),
                nn.Dropout(configs.dropout_rate))

    def forward(self, x):
        # x: [B, L, D]
        
        if self.configs.revin==True:
            self.norm_denorm_layer(x=x,mode="norm")
        
        #Temporal/Timesteps Dimension:
        if '_' not in self.configs.activation:
            x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        else:
            if self.action_temporal_gate=='multiply':
                x = x + self.temporal_gate_first(x.transpose(1, 2)).transpose(1, 2)*self.temporal_gate_second(x.transpose(1, 2)).transpose(1, 2)
            elif self.action_temporal_gate=='add':
                x = x + self.temporal_gate_first(x.transpose(1, 2)).transpose(1, 2)+self.temporal_gate_second(x.transpose(1, 2)).transpose(1, 2)
            else:#self.action_temporal_gate=='average':
                x = x + (self.temporal_gate_first(x.transpose(1, 2)).transpose(1, 2)+self.temporal_gate_second(x.transpose(1, 2)).transpose(1, 2))/2.0
        
        #Channels Dimension:
        if '_' not in self.configs.activation_second: 
            x = x + self.channel(x)
        else:
            if self.action_channel_gate=='multiply':
                x = x + self.channel_gate_first(x)*self.channel_gate_second(x)
            elif self.action_channel_gate=='add':
                x = x + self.channel_gate_first(x)+self.channel_gate_second(x)
            else:#self.action_channel_gate=='average':
                x = x + (self.channel_gate_first(x)+self.channel_gate_second(x))/2.0
                
        if self.configs.revin==True:
            self.norm_denorm_layer(x=x,mode="denorm")

        return x
    
#MTSMixer:
class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim,mlp_activation,dropout_rate) :
        super(MLPBlock,self).__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.mlp_activation=get_torch_activation(activation=mlp_activation)
        self.fc2 = nn.Linear(mlp_dim, input_dim)
        self.dropout=nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.dropout(self.fc2(self.mlp_activation(self.fc1(x))))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling,temporal_mixing_activation,dropout_rate) :
        super(FactorizedTemporalMixing,self).__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList([
            MLPBlock(input_dim // sampling, mlp_dim,temporal_mixing_activation,dropout_rate) for _ in range(sampling)
        ])

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx::self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx::self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim,channel_mixing_activation,dropout_rate) :
        super(FactorizedChannelMixing,self).__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim,channel_mixing_activation,dropout_rate)
        

    def forward(self, x):

        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, tokens_hidden_dim, channels_hidden_dim, 
                 #fac_T, fac_C, 
                 sampling, norm_flag,configs):
        super(MixerBlock,self).__init__()
        self.configs=configs

        #Tokens Mixing Dimension:
        if '_' not in self.configs.activation:
            self.tokens_mixing = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling,configs.activation,configs.dropout_rate)# if fac_T else MLPBlock(tokens_dim, tokens_hidden_dim)
        else:
            self.tokens_mixing_gated_activation_first,self.action_token_mixing_gate,self.tokens_mixing_gated_activation_second=configs.activation.split('_')
            self.tokens_mixing_gate_first = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling,self.tokens_mixing_gated_activation_first,configs.dropout_rate)
            self.tokens_mixing_gate_second = FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling,self.tokens_mixing_gated_activation_second,configs.dropout_rate)
              
        #Channels Mixing Dimension:
        if '_' not in self.configs.activation_second:   
            self.channels_mixing = FactorizedChannelMixing(channels_dim, channels_hidden_dim,configs.activation_second,configs.dropout_rate)# if fac_C else None
        else:
            self.channels_mixing_gated_activation_first,self.action_channel_mixing_gate,self.channels_mixing_gated_activation_second=configs.activation_second.split('_')
            self.channels_mixing_gate_first = FactorizedChannelMixing(channels_dim, channels_hidden_dim,self.channels_mixing_gated_activation_first,configs.dropout_rate)# if fac_C else None
            self.channels_mixing_gate_second = FactorizedChannelMixing(channels_dim, channels_hidden_dim,self.channels_mixing_gated_activation_second,configs.dropout_rate)# if fac_C else None
        
        
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self,x):
        # token-mixing [B, D, #tokens]
        #y = self.norm(x) if self.norm else x
        #y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        #Token mixing dimension:
        if '_' not in self.configs.activation:
            y = self.tokens_mixing(x.transpose(1, 2)).transpose(1, 2)
        else:
            if self.action_token_mixing_gate=='multiply':
                y = self.tokens_mixing_gate_first(x.transpose(1, 2)).transpose(1, 2)*self.tokens_mixing_gate_second(x.transpose(1, 2)).transpose(1, 2)
            elif self.action_token_mixing_gate=='add':
                y = self.tokens_mixing_gate_first(x.transpose(1, 2)).transpose(1, 2)+self.tokens_mixing_gate_second(x.transpose(1, 2)).transpose(1, 2)
            else:#self.action_token_mixing_gate=='average:
                y = (self.tokens_mixing_gate_first(x.transpose(1, 2)).transpose(1, 2)+self.tokens_mixing_gate_second(x.transpose(1, 2)).transpose(1, 2))/2.0

        #Channel mixing dimension:
        y += x
        res = y
        y = self.norm(y) if self.norm else y
        if '_' not in self.configs.activation_second:
            y = res + self.channels_mixing(y)
        else:
            if self.action_channel_mixing_gate=='multiply':
                y = res + self.channels_mixing_gate_first(y)*self.channels_mixing_gate_second(y)
            elif self.action_channel_mixing_gate=='add':
                y = res + self.channels_mixing_gate_first(y)+self.channels_mixing_gate_second(y)
            else:#self.action_channel_mixing_gate=='average':
                y = res + (self.channels_mixing_gate_first(y)+self.channels_mixing_gate_second(y))/2.0

        return y


class MTSMixer_Model(nn.Module):
    def __init__(self, configs):
        super(MTSMixer_Model,self).__init__()
        self.configs=configs
        
        self.mlp_block=MixerBlock(configs.seq_len, configs.enc_in, configs.d_model, configs.d_ff, 
                                  #configs.fac_T, configs.fac_C, 
                                  configs.sampling, configs.norm,configs)
        if configs.revin==True:
            self.norm_denorm_layer=RevIN(num_features=self.configs.enc_in)
        
       
    def forward(self, x):
        if self.configs.revin==True:
            x=self.norm_denorm_layer(x=x,mode='norm')
       
        x=self.mlp_block(x)
        
        if self.configs.revin==True:
            x=self.norm_denorm_layer(x=x,mode='denorm')

        return x

#Dnonlinear:
class Dnonlinear_Model(nn.Module):
    """
    Decomposition-Nonlinear
    """
    def __init__(self, configs):
        super(Dnonlinear_Model, self).__init__()
        self.configs=configs
        self.seq_len = configs.seq_len
        self.decomposition_type=configs.decomposition_type
        
        # Decompsition Kernel Size
        self.kernel_size = configs.moving_avg#by default set to 25
        self.channels = configs.enc_in
        
        #Seasonality:
        if '_' not in self.configs.activation:
            self.Seasonal_Component = nn.Sequential(
                nn.Linear(self.seq_len,self.seq_len),
                get_torch_activation(activation=configs.activation),
                nn.Dropout(self.configs.dropout_rate))
        else:
            self.seasonality_gated_activation_first,self.action_seasonality_gate,self.seasonality_gated_activation_second=configs.activation.split('_')
            self.Seasonal_Component_gate_first = nn.Sequential(
                nn.Linear(self.seq_len,self.seq_len),
                get_torch_activation(activation=self.seasonality_gated_activation_first),
                nn.Dropout(self.configs.dropout_rate))
            
            self.Seasonal_Component_gate_second = nn.Sequential(
                nn.Linear(self.seq_len,self.seq_len),
                get_torch_activation(activation=self.seasonality_gated_activation_second),
                nn.Dropout(self.configs.dropout_rate))
        
        if self.decomposition_type=='avgpool1d':
            resulting_shape=(self.seq_len-self.kernel_size)+1
            padding=int((self.seq_len-resulting_shape)/2)
            self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=padding)
            
            #Trend:
            if '_' not in self.configs.activation_second:
                self.Trend_Component = nn.Sequential(
                    nn.Linear(self.seq_len,self.seq_len),
                    get_torch_activation(activation=configs.activation_second),
                    nn.Dropout(self.configs.dropout_rate))
            else:
                self.trend_gated_activation_first,self.action_trend_gate,self.trend_gated_activation_second=configs.activation_second.split('_')
                self.Trend_Component_gate_first = nn.Sequential(
                    nn.Linear(self.seq_len,self.seq_len),
                    get_torch_activation(activation=self.trend_gated_activation_first),
                    nn.Dropout(self.configs.dropout_rate))
                
                self.Trend_Component_gate_second = nn.Sequential(
                    nn.Linear(self.seq_len,self.seq_len),
                    get_torch_activation(activation=self.trend_gated_activation_second),
                    nn.Dropout(self.configs.dropout_rate))
        else:
            if '_' not in self.configs.activation_second:
                self.Trend_Component = nn.Sequential(
                    nn.Conv1d(in_channels=self.channels,out_channels=self.channels,
                                 kernel_size=self.kernel_size, stride=1, padding='same'),
                    get_torch_activation(activation=configs.activation_second),
                    nn.Dropout(self.configs.dropout_rate))
            else:
                self.trend_gated_activation_first,self.action_trend_gate,self.trend_gated_activation_second=configs.activation_second.split('_')
                
                self.Trend_Component_gate_first = nn.Sequential(
                    nn.Conv1d(in_channels=self.channels,out_channels=self.channels,
                                 kernel_size=self.kernel_size, stride=1, padding='same'),
                    get_torch_activation(activation=self.trend_gated_activation_first),
                    nn.Dropout(self.configs.dropout_rate))
                
                self.Trend_Component_gate_second = nn.Sequential(
                    nn.Conv1d(in_channels=self.channels,out_channels=self.channels,
                                 kernel_size=self.kernel_size, stride=1, padding='same'),
                    get_torch_activation(activation=self.trend_gated_activation_second),
                    nn.Dropout(self.configs.dropout_rate))
        
            
        if configs.revin==True:
            self.norm_denorm_layer=RevIN(num_features=self.configs.enc_in)
        
        
    def forward(self, x):
        if self.configs.revin==True:
            x=self.norm_denorm_layer(x=x,mode='norm')
            
        #Decomposition:
        #Trend modelling:
        if self.decomposition_type=='avgpool1d':
            trend_init = self.avg(x.permute(0, 2, 1))
            #Seasonality modelling as the remainder:
            seasonal_init=x.permute(0, 2, 1)-trend_init
        else:
            if '_' not in self.configs.activation_second:
                trend_output = self.Trend_Component(x.permute(0, 2, 1))   
            else:
                if self.action_trend_gate=='multiply':
                    trend_output = self.Trend_Component_gate_first(x.permute(0, 2, 1))*self.Trend_Component_gate_second(x.permute(0, 2, 1))
                elif self.action_trend_gate=='add':
                    trend_output = self.Trend_Component_gate_first(x.permute(0, 2, 1))+self.Trend_Component_gate_second(x.permute(0, 2, 1))
                else:#self.action_trend_gate=='average':
                    trend_output = (self.Trend_Component_gate_first(x.permute(0, 2, 1))+self.Trend_Component_gate_second(x.permute(0, 2, 1)))/2.0
                    
                    
            
            #Seasonality modelling as the remainder:
            seasonal_init=x.permute(0, 2, 1)-trend_output
        
        #Seasonality:
        if '_' not in self.configs.activation:
            seasonal_output = self.Seasonal_Component(seasonal_init)
        else:
            if self.action_seasonality_gate=='multiply':
                seasonal_output = self.Seasonal_Component_gate_first(seasonal_init)*self.Seasonal_Component_gate_second(seasonal_init)
            elif self.action_seasonality_gate=='add':
                seasonal_output = self.Seasonal_Component_gate_first(seasonal_init)+self.Seasonal_Component_gate_second(seasonal_init)
            else:#self.action_seasonal_gate=='average:
                seasonal_output = (self.Seasonal_Component_gate_first(seasonal_init)+self.Seasonal_Component_gate_second(seasonal_init))/2.0
        
        if self.decomposition_type=='avgpool1d':
            #Trend:
            if '_' not in self.configs.activation_second:
                trend_output = self.Trend_Component(trend_init)
            else:
                if self.action_trend_gate=='multiply':
                    trend_output = self.Trend_Component_gate_first(trend_init)*self.Trend_Component_gate_second(trend_init)
                elif self.action_trend_gate=='add':
                    trend_output = self.Trend_Component_gate_first(trend_init)+self.Trend_Component_gate_second(trend_init)
                else:#self.action_trend_gate=='average':
                    trend_output = (self.Trend_Component_gate_first(trend_init)+self.Trend_Component_gate_second(trend_init))/2.0


        x = seasonal_output + trend_output
        x=x.permute(0,2,1)
        
        if self.configs.revin==True:
            x=self.norm_denorm_layer(x=x,mode='denorm')
            
        return x# to [Batch, Input length, Channel]

#Patching:
class Patching_Block(nn.Module):  #i means channel-independent
    def __init__(self,configs):
    
        super(Patching_Block,self).__init__()
        self.configs=configs
    
        self.revin=self.configs.revin
        self.dropout_rate=self.configs.dropout_rate
        self.d_model=self.configs.enc_in#d_model
        if self.revin==True:
            self.norm_denorm_layer=RevIN(num_features=self.d_model)
        
        self.patch_len=self.configs.patch_len
        self.stride=self.configs.stride
        self.context_window=self.configs.seq_len
        self.patch_num = int((self.context_window - self.patch_len)/self.stride + 1)
        self.padding_patch=self.configs.padding_patch
        
        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
            self.patch_num += 1

        
        
        self.learn_pe=bool(self.configs.train_pe)
        if self.learn_pe==True:
            self.W_P = nn.Linear(self.patch_len, self.d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
            self.pe=self.configs.pe
            self.q_len = self.patch_num
            #print(self.patch_num)
            # Positional encoding
            self.W_pos = positional_encoding(self.pe, self.learn_pe, self.q_len, self.d_model)
            self.dropout_pe=nn.Dropout(self.configs.dropout_rate_pe)

        self.nf=self.d_model * self.patch_num
        self.target_window=self.configs.seq_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_flatten = nn.Linear(self.nf, self.target_window)
        self.activation=self.configs.activation
        self.d_patching_model=self.configs.d_model

        #If sampled factorized patching:
        #apply the temporal factorization first, then transform into patches, then apply nonlinear patching transformation
        #then apply the linear flatten layer.

        if '_' not in self.activation:
            self.patching_transformation=nn.Sequential(
                nn.Linear((self.d_model if self.learn_pe==True else self.patch_len),self.d_patching_model),#potentially add here the option to add cnn transformation after fcn 
                get_torch_activation(activation=self.activation),
                nn.Linear(self.d_patching_model,self.d_model),
                nn.Dropout(self.dropout_rate)
            )
            
        else:
            self.gated_activation_first,self.action_gate,self.gated_activation_second=self.activation.split('_')
            self.patching_transformation_gate_first=nn.Sequential(
                nn.Linear((self.d_model if self.learn_pe==True else self.patch_len),self.d_patching_model),#potentially add here the option to add cnn transformation after fcn
                get_torch_activation(activation=self.gated_activation_first),
                nn.Linear(self.d_patching_model,self.d_model),
                nn.Dropout(self.dropout_rate)
            )
            self.patching_transformation_gate_second=nn.Sequential(
                nn.Linear((self.d_model if self.learn_pe==True else self.patch_len),self.d_patching_model),#potentially add here the option to add cnn transformation after fcn
                get_torch_activation(activation=self.gated_activation_second),
                nn.Linear(self.d_patching_model,self.d_model) ,
                nn.Dropout(self.dropout_rate)
            )
        
    def forward(self,z):
        if self.revin==True:
            z=self.norm_denorm_layer(x=z,mode='norm')

        #1. Transpose channel and temporal dimension as patching is done along temporal dimension:
        #print("Patching Block z-entry shape: ",z.shape)
        z=z.transpose(1,2)
        
        #2. Upsampling of timesteps as preparation for patching:
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
            #print("Shape after padding patch layer: ",z.shape,"\n")

        #3. Creating patches from 3. dimension: patch_num by patch_len
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        #print("Shape after unfolding: ",z.shape,"\n")

        n_vars = z.shape[1]
        #4. Applying linear transformation to dimension patch_len before injection of temporal/positional embeddings:
        if self.learn_pe==True:
            z = self.W_P(z)                                                          # x: [bs x nvars x patch_num x d_model]
            #print("Shape after linear patch_len by d_model: ",z.shape,"\n")

            #5. Reshaping as preration for injection of positional embeddings:
            z = torch.reshape(z, (z.shape[0]*z.shape[1],z.shape[2],z.shape[3]))      # u: [bs * nvars x patch_num x d_model]
            #print(z.shape)
            #if self.learn_pe==True:
            #print(W_pos.shape)

            z= self.dropout_pe(z + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]
            #print("Shape after reshaping and adding positional embeddings: ",u.shape,"\n")
        else:
            z = torch.reshape(z, (z.shape[0]*z.shape[1],z.shape[2],z.shape[3]))      # u: [bs * nvars x patch_num x patch_len]

        #6. Application of nonlinear transformation along d_model dimension (previouslz patch_len)
        # Encoder: some nonlinear transformation:
        #z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        if '_' not in self.activation:
            z=self.patching_transformation(z)
        else:
            if self.action_gate=='multiply':
                z=self.patching_transformation_gate_first(z)*self.patching_transformation_gate_second(z)
            elif self.action_gate=='add':
                z=self.patching_transformation_gate_first(z)+self.patching_transformation_gate_second(z)
            else:#self.action_gate=='average':
                z=(self.patching_transformation_gate_first(z)+self.patching_transformation_gate_second(z))/2.0
            
        #7. Reshaing back to four dimensions: batch size, number channels, number patches, d_model
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        #print("Shape after again reshaping: ",z.shape,"\n")

        #8. Permutation of last two axes and flattening:
        z = z.permute(0,1,3,2) 
        #print("Z after permutation: ",z.shape,"\n")
        z=self.flatten(z)
        #print("Shape after flattening: ",z.shape,"\n")

        #9. Linear projection of input sequence length to target sequence length:
        z=self.linear_flatten(z).transpose(1,2)
        ##print("Shape after linear nf (d_model*patch_num, target window): ",z.shape)

        if self.revin==True:
            z=self.norm_denorm_layer(x=z,mode='denorm')

        return z



#---------------------------------
# EDIT this to a tree model
#---------------------------------


class Custom_Torch_Model(nn.Module):
    def __init__(self, input_channels,
                 output_channels,
                 seq_len,pred_len,
                 ts_module_nodes,
                 nr_nodes,
                 revin_nodes,prediction_projections,
                 activations_nodes,
                 activations_second_nodes,
                 skip_connections,
                 dropout_nodes,
                 #forecast_sequence_difference,
                 d_model_nodes=None,
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
                 merge_mode = 'linear'):
        
        super(Custom_Torch_Model,self).__init__()

        self.tree_depth = int(np.ceil(np.log2(nr_nodes+1)))
        self.num_leaves = 2 ** (self.tree_depth-1)

        #Input dimensions=output dimensions, i.e., 
        #number of input timesteps per sequence=number of prediction timesteps
        #number of input channels=number of output channels
        self.input_channels = input_channels
        self.seq_len=seq_len
        self.max_features = int(self.input_channels*channels_ratio)
    
        self.skip_connections=skip_connections
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

        node_idx = 0
        for depth_idx in range(0,self.tree_depth):
            current_level_path_probas=[]
            current_level_ts_modules=[]

            for path_proba_idx in range(0,2**depth_idx):
                #Get path proba modules:
                if depth_idx<(self.tree_depth-1):
                    current_level_path_probas.append(nn.Sequential(
                                                        nn.Linear(self.max_features, 2),
                                                        torch.nn.Softmax(dim=-1)))
                
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

            self.path_proba_modules.append(torch.nn.ModuleList(current_level_path_probas))
            self.ts_tree_modules.append(torch.nn.ModuleList(current_level_ts_modules))

        # self.node_components=torch.nn.ModuleList( self.node_components)
        self.path_proba_modules=torch.nn.ModuleList(self.path_proba_modules)
        self.ts_tree_modules=torch.nn.ModuleList(self.ts_tree_modules)
        
        self.merge_mode = merge_mode
        if self.merge_mode == 'bnn':
            self.output_leaf_projection = bnn.BayesLinear(prior_mu=0,
                                                          prior_sigma=0.1,
                                                          in_features=(self.num_leaves*self.max_features),
                                                          out_features=output_channels)

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

            for ts_module_idx in range(0,len(current_level_ts_modules)):
                # Select the current ts module:
                ts_module=current_level_ts_modules[ts_module_idx]
                
                # Select the subset of features in the current level:
                if not self.subset_fixed:
                    selected_features = torch.randperm(self.input_channels)[:self.max_features]

                # Add the information from the parent nodes of the previous level:
                if tree_depth_idx > 0:
                    idx_parent_node=(ts_module_idx)//2


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


                    # nr_nodes_all_previous_levels = np.sum(2**np.arange(0,tree_depth_idx))
                    # if self.skip_connections[nr_nodes_all_previous_levels+ts_module_idx-1]!=None:
                    #     latent_x_subset=latent_x_subset+node_outputs[self.skip_connections[nr_nodes_all_previous_levels+ts_module_idx-1]]
                    

                else: # depth being 0
                    x_subset = x[:, :, selected_features]
                    # latent representation
                    latent_x_subset=ts_module(x_subset)


                if tree_depth_idx<self.tree_depth-1:#then prepare the softmax node splits for the next level:
                    #Compute left and right probabilities: the softmax weights can be computed using either the merged latent x subset or the sampled x_subset:
                    #you would need to check which way leads to higher performance
                    softmax_proba=self.path_proba_modules[tree_depth_idx][ts_module_idx](latent_x_subset)
                    left_proba,right_proba=softmax_proba[:,:,0].unsqueeze(-1),softmax_proba[:,:,1].unsqueeze(-1)

                    left_latent_x_subset=left_proba*latent_x_subset
                    previous_level_outputs[tree_depth_idx].append(left_latent_x_subset)
                    
                    right_latent_x_subset=right_proba*latent_x_subset
                    previous_level_outputs[tree_depth_idx].append(right_latent_x_subset)
                    # print('Length of Outputs for next level: ',len(previous_level_outputs[tree_depth_idx]),'\n')

                else:
                    tree_outputs.append(latent_x_subset)
                
                node_outputs.append(latent_x_subset) 


            # # ---------------------------- KEEPING THIS SECTION FOR SKIP CONNECTION REMINDER ---------------------
            # if node_idx==0:
            #     current_node_output=self.node_components[node_idx](x)
            #     #check if forecast_sequence_difference==True:
            #     #if self.forecast_sequence_difference==True:
            #     #    #save difference to future view from forward pass:
            #     #    forecast_difference_output=current_node_output
            #     #    
            #     #    #add current_node_output to x, and return current_node_output as
            #     #    current_node_output=x+forecast_difference_output
            # else:
            #     current_node_output=self.node_components[node_idx](node_outputs[-1])

            #     if self.skip_connections[node_idx-1]!=None:
            #         current_node_output=current_node_output+node_outputs[self.skip_connections[node_idx-1]]
            # node_outputs.append(current_node_output)     
            # #-------------------------------------------------------------------------------------------------------



        # Merge tree leaves with specified leaves operation:

        # print('\nNumber of Leaf Nodes: ',len(tree_outputs))
        if self.merge_mode == 'bnn':    # if bnn is merging mode then all other leaves operations become obsolete., they are bypassed.
             tree_outputs=torch.concat(tree_outputs,axis=-1) 
             tree_outputs=self.output_leaf_projection(tree_outputs)
            
        else:   # if merge mode is not bnn, then we apply the specified leaves operation:
            if self.leaves_operation == 'concat':
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
    


#---------------------------------------------
#           CUSTOM TREE END
#---------------------------------------------




def get_NAS_model_params(NAS_model,regularization):
    nas_model_params=[]
    for node in NAS_model.node_components:
        nas_model_params.extend(list(node.parameters()))
    
    if regularization==False:
        try:
            nas_model_params.extend(list(NAS_model.output_channel_projection.parameters()))
            nas_model_params.extend(list(NAS_model.pred_len_projection.parameters()))
        except:
            "no_projections_exception"
    return nas_model_params

#Training & Testing Functions:
import time
def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

@ignore_warnings
def environment_step(sampled_architecture,**kwargs):
    
    keep_running_sampled_model=True
    criterion=nn.MSELoss()

    # kl_loss  = bnn.BKLLoss(reduction='mean', last_layer_only=True)
    
    #while keep_running_sampled_model==True:
    #    try:
    nr_inputs=kwargs['special_parameters']['nr_inputs']
    nr_targets=kwargs['special_parameters']['nr_targets']

    #batch_size=kwargs['special_parameters']['batch_size']
    epochs=kwargs['special_parameters']['epochs']
    #patience=kwargs['special_parameters']['patience']
    
    use_amp=kwargs['special_parameters']['use_amp']
    #walk_forward_validation=kwargs['special_parameters']['walk_forward_validation']
    checkpoints=kwargs['special_parameters']['checkpoints']
    reverse_rescale=kwargs['special_parameters']['reverse_rescale']
    use_gpu=kwargs['special_parameters']['use_gpu']
    #device=kwargs['special_parameters']['device']
    device = kwargs['special_parameters']['device']#_acquire_device(use_gpu=use_gpu)#True)#torch.device('cpu')
    
    rewards_first_subsets=[]
    rewards_second_subsets=[]
    training_testing_times=[]

    for seq_len in [48,96,192]:
        #seq_len=kwargs['special_parameters']['seq_len']
        pred_len=seq_len#kwargs['special_parameters']['pred_len']

        with open('Energy_Production_Data/EU_Time_Series/'+kwargs['special_parameters']['dataloader_name']+str(seq_len)+'_Train.pkl','rb') as f:
            current_dataloader=pickle.load(f)
            train_data=current_dataloader[kwargs['special_parameters']['train_test_split']]['Train_Set_Loader']
            validation_data=current_dataloader[kwargs['special_parameters']['train_test_split']]['Test_Set_Loader']
            rescaler=current_dataloader[kwargs['special_parameters']['train_test_split']]['Scaler']
        
        NAS_model=Custom_Torch_Model(input_channels=nr_inputs,
                                        output_channels=nr_targets,
                                        seq_len=seq_len,pred_len=pred_len,
                                        nr_nodes=sampled_architecture['nr_nodes'],
                                        ts_module_nodes=sampled_architecture['ts_module_nodes'],
                                        revin_nodes=sampled_architecture['revin_nodes'],
                                        dropout_nodes=sampled_architecture['dropout_nodes'],
                                        activations_nodes=sampled_architecture['activations_nodes'],
                                        skip_connections=sampled_architecture['skip_connections'],
                                        #forecast_sequence_difference=sampled_architecture['forecast_sequence_difference'],
                                        prediction_projections=sampled_architecture['prediction_projections'],
                                        d_model_nodes=sampled_architecture['d_model_nodes'],
                                        n_heads_nodes=sampled_architecture['n_heads_nodes'],
                                        self_attention_type_nodes=sampled_architecture['self_attention_type_nodes'],
                                        patch_len_nodes=sampled_architecture['patch_len_nodes'],
                                        stride_nodes=sampled_architecture['stride_nodes'],
                                        train_pe_nodes=sampled_architecture['train_pe_nodes'],
                                        kernel_size_nodes=sampled_architecture['kernel_size_nodes'],
                                        activations_second_nodes=sampled_architecture['activations_second_nodes'],
                                        dff_nodes=sampled_architecture['dff_nodes'],
                                        sampling_nodes=sampled_architecture['sampling_nodes'])
        
        
        
        NAS_model.to(device)
        
        nas_model_params=[]
        for node in NAS_model.node_components:
            nas_model_params.extend(list(node.parameters()))
        try:
            nas_model_params.extend(list(NAS_model.output_channel_projection.parameters()))
            nas_model_params.extend(list(NAS_model.pred_len_projection.parameters()))
        except:
            "no_projections_exception"
        optimizer_NAS_model = torch.optim.Adam(nas_model_params, lr=0.001)
        
        #nas_model_params=get_NAS_model_params(NAS_model=NAS_model,regularization=False)
        #optimizer_NAS_model=get_optimizer(NAS_model_params=nas_model_params,opt_name=sampled_architecture['optimizer'],lr=sampled_architecture['lr'])
        
        
        #1. Train the model: ####################################################################################
        start_tr_ts_time_first_subset=time.time()
        NAS_model=train_NAS_model(NAS_model=NAS_model,optimizer_NAS_model=optimizer_NAS_model,checkpoints=checkpoints,
                                device=device,patience=1,epochs=epochs,
                                train_data=train_data,criterion=criterion,use_amp=use_amp,
                                walk_forward_validation=True)#,sampled_architecture=sampled_architecture)#callback on train set
        #Done with training the model: ##############################################################################
        #2. Validate the model: ############################################################################################
        #Compute the reward:
        reward=validate_NAS_model(NAS_model=NAS_model,validation_data=validation_data,
                        compute_reward=True,device=device,
                        mse_array=[],smape_array=[],reverse_rescale=reverse_rescale,rescaler=(rescaler if reverse_rescale==True else None))
        end_tr_ts_time_first_subset=time.time()
        tr_ts_time_first_subset=end_tr_ts_time_first_subset-start_tr_ts_time_first_subset
        rewards_first_subsets.append(reward)
        
        del train_data
        del validation_data
        #Done with testing the model: ####################################################################################         

        
        train_data=current_dataloader['Train_Test2']['Train_Set_Loader']
        validation_data=current_dataloader['Train_Test2']['Test_Set_Loader']
        start_tr_ts_time_second_subset=time.time()
        NAS_model=train_NAS_model(NAS_model=NAS_model,optimizer_NAS_model=optimizer_NAS_model,checkpoints=checkpoints,
                                device=device,patience=1,epochs=epochs,
                                train_data=train_data,criterion=criterion,use_amp=use_amp,
                                walk_forward_validation=True)#,sampled_architecture=sampled_architecture)
        
        reward_second_subset=validate_NAS_model(NAS_model=NAS_model,validation_data=validation_data,
                        compute_reward=True,device=device,
                        mse_array=[],smape_array=[],reverse_rescale=reverse_rescale,rescaler=(rescaler if reverse_rescale==True else None))
        end_tr_ts_time_second_subset=time.time()
        
        tr_ts_time_second_subset=end_tr_ts_time_second_subset-start_tr_ts_time_second_subset
        ts_ts_time_rew=(1.0/(tr_ts_time_first_subset))+(1.0/(tr_ts_time_second_subset))
        training_testing_times.append(ts_ts_time_rew)
        
        rewards_second_subsets.append(reward_second_subset)

        
    torch.cuda.empty_cache()
    
    return [np.sum(rewards_first_subsets),np.sum(rewards_second_subsets),sampled_architecture,np.sum(training_testing_times)]


def train_NAS_model(NAS_model,optimizer_NAS_model,checkpoints,
                   device,patience,epochs,
                   train_data,criterion,use_amp,
                   walk_forward_validation,kl_criterion=None,validation_data=None):
    
    #1. Train the model: ####################################################################################
    iter_count = 0
    
    path = os.path.join(checkpoints, 'TS_Benchmark_training')
    #if not os.path.exists(path):
    #    os.makedirs(path)

    early_stopping = EarlyStopping(patience=patience,delta=0.01,verbose=False)

    #scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer_NAS_model,
    #                                        steps_per_epoch = len(train_data),
    #                                        pct_start = 0.3,
    #                                        epochs = epochs,
    #                                        max_lr = 0.001)
    
    for epoch in range(0,epochs):
        loss_denom=0
        train_loss=0

        NAS_model.train()
        for i, train_items in enumerate(train_data):    
            #Get train item: #############################################################################
            batch_x, batch_y=train_items
            #Make sure input and output tensors are contiguous:

            iter_count += 1
            optimizer_NAS_model.zero_grad()
            #Make sure both the data and the model are on the same device: (otherwise potentially a runtime error)
            #try:
            batch_x = batch_x.to(torch.float32).to(device)
            batch_y = batch_y.to(torch.float32).to(device)

            #if NAS_model.forecast_sequence_difference==True:
            #    difference_y_x=batch_y-batch_x
            #    difference_y_x=difference_y_x.float().to(device)
            #if not batch_x.is_contiguous():
            #    batch_x = batch_x.contiguous()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = NAS_model(batch_x)
            else:
                outputs = NAS_model(batch_x)

            #Make sure predicted tensors are contiguous:
            if not outputs.is_contiguous():
                outputs = outputs.contiguous()
            if not batch_y.is_contiguous():
                batch_y = batch_y.contiguous()

            if not batch_x.is_contiguous():
                batch_x = batch_x.contiguous()

            # adjust kl weight based on number of batches, so that the kl loss has a similar magnitude as the mse loss, and does not get overshadowed by it or vice versa. 
            # This is important to ensure that the kl loss can effectively regularize the model and prevent overfitting, while still allowing the mse loss to drive the model towards accurate predictions.
            # kl_weight = 1/len(train_data) 
            # loss = criterion(outputs, batch_y) + kl_weight * (kl_criterion(NAS_model) if kl_criterion is not None else 0.0)


            loss = criterion(outputs, batch_y)  
            #Save the loss relevant for early stopping before adding the L1 and L2 penalties:
            #in the case of the latter, the loss would always keep on decreasing due to lower the magnitude
            #of the trainable variables, thus exclude them from early stopping!
            train_loss=train_loss+loss
            loss_denom=loss_denom+1

            #Check for regularization type:
            #if sampled_architecture['reg_type'] is not None:
            #    #Get NAS model parameters trained in previous iteration:
            #    nas_model_params=get_NAS_model_params(NAS_model=NAS_model,regularization=True)

            #    if 'L1' in sampled_architecture['reg_type']:
            #        l1_reg_loss = sum(p.abs().sum() for p in nas_model_params)
            #        loss=loss+torch.tensor(sampled_architecture['reg_coeff'],dtype=torch.float32).to(device)*l1_reg_loss.to(torch.float32).to(device)

            #    if 'L2' in sampled_architecture['reg_type']:
            #        l2_reg_loss=sum((p**2).sum() for p in nas_model_params)
            #        loss=loss+torch.tensor(sampled_architecture['reg_coeff'],dtype=torch.float32).to(device)*l2_reg_loss.to(torch.float32).to(device)

            
            
            if use_amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer_NAS_model)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer_NAS_model.step()
            #End backward Pass: ################################################################################
            

        #Apply Early Stopping Callback: ########################################################################
        #if validation_data!=None:
        #    val_loss=validate_NAS_model(NAS_model=NAS_model,validation_data=validation_data,
        #                   compute_reward=False,device=device,
        #                   validation_loss=[],criterion=criterion)
        #    val_loss=torch.tensor(val_loss).float()
        #    early_stopping(torch.mean(val_loss), NAS_model, path)
        #else:
        #train_loss=torch.tensor(train_loss).float()
        early_stopping(train_loss/loss_denom, NAS_model, path)
        if early_stopping.early_stop:
            #print("Early stopping")
            break
        #End Train Items: #############################################################################
        
        #Potentially adjust the learning rate: ########################################################
        #adjust_learning_rate(optimizer=optimizer_NAS_model, scheduler=scheduler, epoch=epoch + 1,lradj="type3",
        #learning_rate=0.001, printout=False)
        #End adjusting the learning rate: #############################################################
        
        #When applying walk-forward validation, then load model weights from best path
        #if walk_forward_validation==True:
        #    best_model_path = path+'/'+'checkpoint.pth'
        #    NAS_model.load_state_dict(torch.load(best_model_path))
    #Done with training the model: ##############################################################################
    return NAS_model


def validate_NAS_model(NAS_model,validation_data,
                       compute_reward,device,
                       mse_array=None,smape_array=None,reverse_rescale=None,rescaler=None,
                       validation_loss=None,criterion=None):#,
                       #model_names_array=None,subset_names_array=None,batch_sample_indices=None,
                       #country_names=None,EU_countries_dict=None):
    
    NAS_model.eval()
    
    with torch.no_grad():
        for i, validation_items in enumerate(validation_data):
            batch_x, batch_y=validation_items
            batch_x = batch_x.to(torch.float32).to(device)
            batch_y = batch_y.to(torch.float32).to(device)
            
            outputs=NAS_model(batch_x).to(device)
            
            if compute_reward==True:
                #if NAS_model.forecast_sequence_difference==True:
                #    outputs=outputs[0]
                #Reverse rescale the predictions: ##################################################################
                true_batch_values=batch_y.detach().cpu().numpy()
                batch_predictions=outputs.detach().cpu().numpy()
                    
                #Reverse rescale predictions & true values:
                for batch_idx in range(0,len(true_batch_values)):
                    #print(true_batch_values[batch_idx,:,:].shape)
                    if reverse_rescale==True:
                        true_batch=rescaler.inverse_transform(true_batch_values[batch_idx,:,:])
                        predicted_batch=rescaler.inverse_transform(batch_predictions[batch_idx,:,:])
                    else:
                        true_batch=true_batch_values[batch_idx,:,:]
                        predicted_batch=batch_predictions[batch_idx,:,:]

                    for country_idx in range(0,true_batch.shape[1]):
                        true_country=true_batch[:,country_idx]
                        predicted_country=predicted_batch[:,country_idx]

                        results=RegressionMetric(y_true=true_country,y_pred=predicted_country).get_metrics_by_list_names(["RMSE","SMAPE"])
                        mse_array.append(results['RMSE'])
                        #smape_array.append(results['SMAPE'])
                        
            else:
                if NAS_model.forecast_sequence_difference==True:
                    outputs=outputs[0]
                
                validation_loss.append(criterion(outputs, batch_y))
          

            

    if compute_reward==True:
        if reverse_rescale==True:
            #reward=(np.array(mse_array)+(np.array(smape_array)*100))/2
            #reward=100/np.mean(reward) 
            #reward=np.array(mse_array)
            reward=100/np.mean(mse_array) 
        else:
            #reward=(np.array(mse_array)+(np.array(smape_array)))/2
            reward=np.array(mse_array)
            reward=1/np.mean(reward) 
        
        #reward=1.0/(np.array(smape_array).mean())
        
        return reward
    else:
        return validation_loss


def random_sampling_model(search_space,embedding_matrix,embedding_none_matrix):

    sampled_architecture={'nr_nodes':None,
                          'kernel_size_nodes':{},
                          'd_model_nodes':{},
                          'patch_len_nodes':{},
                          'train_pe_nodes':{},
                          'stride_nodes':{},
                          'dff_nodes':{},
                          'sampling_nodes':{},
                          'ts_module_nodes':[],
                          'revin_nodes':[],
                          'dropout_nodes':[],
                          'activations_nodes':[],
                          'activations_second_nodes':{},
                          'skip_connections':[],
                          'prediction_projections':None}
    
    
    
    embedding_sampled_architecture=[]
    keys=list(search_space.keys()).copy()
    
    for ss_key in keys:
        sampled_cell_component=np.random.choice(search_space[ss_key],size=1)[0]
        index_sampled_component=search_space[ss_key].index(sampled_cell_component)

        if "nr_" in ss_key:
            sampled_architecture['nr_nodes']=sampled_cell_component
            embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
            
        elif ss_key=='prediction_projections':
            sampled_architecture['prediction_projections']=sampled_cell_component
            embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])

            
        else:
            current_node=int(ss_key.split('_')[1])
            if current_node<sampled_architecture['nr_nodes']:
    
                if 'ts_module' in ss_key:
                    sampled_architecture['ts_module_nodes'].append(sampled_cell_component)
                    embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])

                    
                elif 'dropout' in ss_key:
                    sampled_architecture['dropout_nodes'].append(sampled_cell_component)
                    embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                
                elif 'skip' in ss_key:
                    sampled_architecture['skip_connections'].append(sampled_cell_component)    
                    embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    
                elif 'kernel_size' in ss_key:
                    if 'Dnonlinear' in sampled_architecture['ts_module_nodes'][current_node]:
                        sampled_architecture['kernel_size_nodes']['node_'+str(current_node)]=sampled_cell_component
                        embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    else:
                        embedding_sampled_architecture.append(embedding_none_matrix[ss_key])
                
                elif 'd_model' in ss_key:
                    if sampled_architecture['ts_module_nodes'][current_node] in ['TSMixer','MTSMixer','Attention','Patching']:
                        sampled_architecture['d_model_nodes']['node_'+str(current_node)]=sampled_cell_component
                        embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    else:
                        embedding_sampled_architecture.append(embedding_none_matrix[ss_key])
                
                    

                elif 'patch_len' in ss_key:
                    if "Patching" in sampled_architecture['ts_module_nodes'][current_node]:
                        sampled_architecture['patch_len_nodes']['node_'+str(current_node)]=sampled_cell_component
                        embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    else:
                        embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

                
                elif 'stride' in ss_key:
                    if "Patching" in sampled_architecture['ts_module_nodes'][current_node]:
                        sampled_architecture['stride_nodes']['node_'+str(current_node)]=sampled_cell_component
                        embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    else:
                        embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

                elif 'dff' in ss_key:
                    if "MTSMixer" in sampled_architecture['ts_module_nodes'][current_node]:
                        sampled_architecture['dff_nodes']['node_'+str(current_node)]=sampled_cell_component
                        embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    else:
                        embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

                elif 'sampling' in ss_key:
                    if "MTSMixer" in sampled_architecture['ts_module_nodes'][current_node]:
                        sampled_architecture['sampling_nodes']['node_'+str(current_node)]=sampled_cell_component
                        embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                    else:
                        embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

                
                else:# 'activation' in ss_key:# and 'cell_inject' not in ss_key:
                    
                    #Second activation Function:
                    if 'second' in ss_key:
                        if "Attention" not in sampled_architecture['ts_module_nodes'][current_node] and "Patching" not in sampled_architecture['ts_module_nodes'][current_node]:
                            if '1.component' in ss_key:
                                sampled_architecture['activations_second_nodes']['node_'+str(current_node)]=sampled_cell_component
                                embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                            
                            elif 'action' in ss_key:
                                if sampled_cell_component!=None:
                                    sampled_architecture['activations_second_nodes']['node_'+str(current_node)]=sampled_architecture['activations_second_nodes']['node_'+str(current_node)]+'_'+sampled_cell_component
                                    embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                                else:
                                    embedding_sampled_architecture.append(embedding_none_matrix[ss_key])
                            
                            else:#2.component
                                if '_' in sampled_architecture['activations_second_nodes']['node_'+str(current_node)]:#i.e., sampled an action
                                    #Overwrite sample component and index sampled component:
                                    possible_options_second_component=search_space[ss_key].copy()
                                    possible_options_second_component.remove(sampled_architecture['activations_second_nodes']['node_'+str(current_node)].split('_')[0])
                                    
                                    sampled_cell_component=np.random.choice(possible_options_second_component,size=1)[0]
                                    index_sampled_component=possible_options_second_component.index(sampled_cell_component)

                                    sampled_architecture['activations_second_nodes']['node_'+str(current_node)]=sampled_architecture['activations_second_nodes']['node_'+str(current_node)]+'_'+sampled_cell_component
                                    embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                                else:
                                    embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

                        else:
                            embedding_sampled_architecture.append(embedding_none_matrix[ss_key])
                    
                    #First activation Function:
                    else:
                        if '1.component' in ss_key:
                            sampled_architecture['activations_nodes'].append(sampled_cell_component)
                            embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])

                        elif 'action' in ss_key:
                            if sampled_cell_component!=None:
                                sampled_architecture['activations_nodes'][-1]=sampled_architecture['activations_nodes'][-1]+'_'+sampled_cell_component
                                embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                            else:
                                embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

                        else:#2.component
                            if '_' in sampled_architecture['activations_nodes'][-1]:#i.e., sampled an action
                                possible_options_second_component=search_space[ss_key].copy()
                                possible_options_second_component.remove(sampled_architecture['activations_nodes'][-1].split('_')[0])
                                
                                sampled_cell_component=np.random.choice(possible_options_second_component,size=1)[0]
                                index_sampled_component=possible_options_second_component.index(sampled_cell_component)

                                sampled_architecture['activations_nodes'][-1]=sampled_architecture['activations_nodes'][-1]+'_'+sampled_cell_component
                                embedding_sampled_architecture.append(embedding_matrix[ss_key][:,index_sampled_component,:])
                            else:
                                embedding_sampled_architecture.append(embedding_none_matrix[ss_key])

            else:
                embedding_sampled_architecture.append(embedding_none_matrix[ss_key])
    
    if sampled_architecture["prediction_projections"]==None:
        sampled_architecture["prediction_projections"]=False

    
    if sampled_architecture["revin_nodes"]==[]:
        sampled_architecture["revin_nodes"]=[True]*sampled_architecture["nr_nodes"]



    return sampled_architecture,torch.stack(embedding_sampled_architecture,axis=1)


class Custom_Linear(nn.Module):
    """
    A class implementing custom linear prediction layer:
    
    
    Parameters:
    ----------
    previous_component_alternatives : int
        Number of values to sample from for the cell component, predicted by the preceeding actor head.
    current_component_alternatives : int
        Number of potential values to sample from for the current cell component.
    latent_shape_common : int
        The dimnesionality of the latent feature space produced by the shared LSTM layer.
    
    
    Attributes: 
    ----------  
    current_component_kernel : tensor
        The trainable weights applied to the LSTM latent representation of the current state of the environment.
    current_component_bias : tensor
        The bias weight tensor.
    previous_component_kernel : tensor
        The trainable weights applied to the probabilities of the previously sampled cell component.

    
    Methods
    -------
    forward(x_current,x_previous):
        Performs the forward pass using the LSTM latent representation & the predicted softmax squashed
        probabilities produced by the preceeding actor head (x_previous).
    
    """
    def __init__(self,
                previous_component_alternatives,
                current_component_alternatives,
                latent_shape_common):
        
        super().__init__()
        self.current_component_kernel = torch.nn.Linear(latent_shape_common,
                                                        current_component_alternatives)
        
        self.previous_component_kernel = torch.nn.Linear(previous_component_alternatives,
                                                         current_component_alternatives)
        
    def forward(self, x_current,x_previous):
        """
        Performs a forward pass of custom linear layer.
        
        """
        return self.current_component_kernel(x_current)+self.previous_component_kernel(x_previous)


#Actor-Critic:
class Policy(nn.Module):
    """
    A class implementing both the actor and the critic in one neural network model:
    
    
    Parameters:
    ----------
    embedding_dimension : int
        Dimensionality of continuous vectors representing the components of the search space
    actor_head_shapes : list of int values
        List of integer values corresponding to the number of possible values each actor head samples
    shared_units : int
        The number of neurons in the shared LSTM-cell(s) part of the actor-critic model.
    search_space_values : dict
        The mapping between the components in the search space (the keys) and the possible values to
        sample from (the values of the dictionary).
    
    
    Attributes:
    (in addition to all Parameters)
    ----------
    shared_actor_critic_layer : torch object
        The time series-based LSTM layer processing the embedded representation of the previously sampled cell.
    cell_components_to_sample: list of str
        The names of all cell components. 
    action_heads: list 
        A torch Parameterlist of the linear actor layers responsible for the sampling of all cell components.
    value_head: torch layer object
        The linear critic prediction layer
    saved_actions: list
        A mapping between the sum of the log-probability of the sampled cells during an episode,
        and the corresponding values produced by the critic layer
    rewards: list
        The list of rewards achieved during an episode
    

    Methods
    -------
    select_action(predicted_probs):
        Creates a categorical distribution from the continuous probabilities, and sample a single value.
    
    forward(current_state_environment,previous_memory_state=None):
        Performs the forward pass of the actor and the critic
    
    """
    def __init__(self,
                embedding_dimension,
                actor_head_shapes,
                shared_units,
                search_space_values,
                embedding_matrix_values,
                embedding_none_matrix_values):#,
                #disconnected_nodes_idx):

        super(Policy, self).__init__()
        self.embedding_dimension=embedding_dimension
        self.actor_head_shapes=actor_head_shapes
        self.shared_units=shared_units
        
        self.search_space_values=search_space_values
        self.embedding_matrix_values=embedding_matrix_values
        self.embedding_none_matrix_values=embedding_none_matrix_values
        
        #self.disconnected_nodes_idx=disconnected_nodes_idx
        
        self.shared_actor_critic_layer = nn.LSTM(input_size=self.embedding_dimension, 
                                                 hidden_size=self.shared_units,
                                                num_layers=2)
        self.cell_components_to_sample=list(self.search_space_values.keys())
        

        #Create as many actor's head as number of cell components:
        #Create as many actor's head as number of cell components:
        self.action_heads=[]
        for actor_head_shapes_idx in range(0,len(self.actor_head_shapes)):
            if actor_head_shapes_idx==0:
                self.action_heads.append(nn.Linear(self.shared_units,self.actor_head_shapes[0]))
            else:
                self.action_heads.append(Custom_Linear(latent_shape_common=self.shared_units,
                                                      current_component_alternatives=self.actor_head_shapes[actor_head_shapes_idx],
                                                      previous_component_alternatives=self.embedding_dimension))
                                                      #self.actor_head_shapes[actor_head_shapes_idx-1]))
                                                      
        self.action_heads=nn.ParameterList(self.action_heads)      
        
        # Critic's layer
        self.value_head = nn.Linear(self.shared_units, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        

    def select_action(self,predicted_probs):
        """Select an action from the probabilities produced by a single actor head.

        Parameters
        ----------
        predicted_probs : list of torch values
            The predicted probabilities for selecting a single cell component
            
        Returns
        -------
        action.item() : int
            Returns the index of the sampled cell component.
        log_prob_action : float
            Returns the log-probability of the action selected from the categorical distribution.    
        """
    
        # create a categorical distribution over the list of probabilities of actions
        m= Categorical(predicted_probs)

        # and sample an action using the distribution
        action = m.sample()
        log_prob_action=m.log_prob(action)
        
        return [action.item(),log_prob_action]
    
    def train_action_heads(self,action_head_idx,trainable):
        for p in self.action_heads[action_head_idx].parameters():
            p.requires_grad=trainable

    def forward_pass_single_action_head(self,cell_component,action_head_idx,hidden_state_environment,
                                                    embedding_sampled_architecture,log_prob_sum,action_probs,cell_entropy,
                                                    embeddings_cell_component=None):
                
        cell_component_logits=(self.action_heads[action_head_idx](hidden_state_environment) if action_head_idx==0 else
                                    self.action_heads[action_head_idx](x_current=hidden_state_environment,
                                                                    x_previous=embeddings_cell_component))[0]
                                                                    #embeddings_cell_component))[0]
                                                                    
        cell_component_probs=F.softmax(cell_component_logits,dim=-1)

        #Compute normalized entropy of cell component:
        cell_component_probs_np=cell_component_probs.detach().numpy().flatten()
        cell_component_entropy=-1*np.sum([np_pr*np.log(np_pr) for np_pr in cell_component_probs_np])/np.log(len(cell_component_probs_np))
        cell_entropy.append(cell_component_entropy)
        

        selected_cell_component,log_prob_cell_component=self.select_action(predicted_probs=cell_component_probs)
        #get embeddings of selected cell component:
        embeddings_cell_component=self.embedding_matrix_values[cell_component][:,selected_cell_component,:]
        embedding_sampled_architecture.append(embeddings_cell_component)
        log_prob_sum=log_prob_sum+log_prob_cell_component
        action_probs.append(cell_component_probs[selected_cell_component])

        return selected_cell_component,embeddings_cell_component,log_prob_sum


    def forward(self, current_state_environment,
                previous_memory_state=None):
        """Performs a forward pass of both the actor and the critic.

        Parameters
        ----------
        current_state_environment : tensor
            The embedded representation of the cell sampled in the previous timestep of the controller.
        previous_memory_state: torch tensor
            The latent representation produced by the shared LSTM-layer from the embedded representation of 
            previously predicted cells. In the first timestep of the first episode previous_memory_state=None.
            
        Returns
        -------
        action_probs : tensor
            The softmax squashed probabilities produced by actor heads.
        selected_actions: list
            A nested list, the elements of which represent lists consisting of three items: the name of the
            cell component, the index of the sampled value of the cell component and the actual value of the sampled
            component.
        memory_state_environment : tensor
            The LSTM latent representation of the current state of the environment including past accumulated 
            memory.
        """
        
        latent_state_environment = self.shared_actor_critic_layer(input=current_state_environment,
                                                                 hx=previous_memory_state)
        
        hidden_state_environment,memory_state_environment=latent_state_environment
        hidden_state_environment=hidden_state_environment[:,-1,:]
        

        #Sampling Actions from Actor Model: ########################################################################
        # actor heads: choses action to take from state s_t
        # by returning probability of each action: consider conditional hyperparameters!
        sampled_architecture={'nr_nodes':None,
                          'kernel_size_nodes':{},
                          'd_model_nodes':{},
                          'patch_len_nodes':{},
                          'train_pe_nodes':{},
                          'stride_nodes':{},
                          'dff_nodes':{},
                          'sampling_nodes':{},
                          'ts_module_nodes':[],
                          'revin_nodes':[],
                          'dropout_nodes':[],
                          'activations_nodes':[],
                          'activations_second_nodes':{},
                          'skip_connections':[],
                          'prediction_projections':None}
        
        
        
        embedding_sampled_architecture=[]
        action_probs=[]
        cell_entropy=[]
        log_prob_sum=0

        for action_head_idx in range(0,len(self.action_heads)):
            cell_component=self.cell_components_to_sample[action_head_idx]
                    
            #Save selected action:
            if "nr_" in cell_component:
                #Perform forward pass with single action head:
                selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                            action_head_idx=action_head_idx,
                                                                            hidden_state_environment=hidden_state_environment,
                                                                            embedding_sampled_architecture=embedding_sampled_architecture,
                                                                            log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy)
                sampled_architecture['nr_nodes']=self.search_space_values[cell_component][selected_cell_component]
                
            
            elif cell_component=="prediction_projections":
                selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                sampled_architecture['prediction_projections']=self.search_space_values[cell_component][selected_cell_component]
            
            

            else:
                nr_nodes=sampled_architecture['nr_nodes']
                current_node=int(cell_component.split('_')[1])

                if current_node<nr_nodes:
                    #Perform forward pass with single action head:
                    
                    if 'ts_module' in cell_component:
                        selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                        
                        sampled_architecture['ts_module_nodes'].append(self.search_space_values[cell_component][selected_cell_component])
                    
                    
                    elif "dropout" in cell_component:
                        selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                        
                        sampled_architecture['dropout_nodes'].append(self.search_space_values[cell_component][selected_cell_component])


                    elif "skip" in cell_component:
                        
                        selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                        sampled_architecture['skip_connections'].append(self.search_space_values[cell_component][selected_cell_component])
                    
                    
                    elif 'd_model' in cell_component:
                        if sampled_architecture['ts_module_nodes'][current_node] in ['TSMixer','MTSMixer','Attention','Patching']:
                            selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                           
                            sampled_architecture['d_model_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                            
                        else:
                            
                            embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])

                    
                    #Decomposition:
                    elif 'kernel_size' in cell_component:
                        if 'Dnonlinear' in sampled_architecture['ts_module_nodes'][current_node]:
                            selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                            
                            sampled_architecture['kernel_size_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                            
                        else:
                            
                            embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])

                    
                    #Patching:
                    elif 'patch_len' in cell_component:
                        if sampled_architecture['ts_module_nodes'][current_node]=='Patching':
                            selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                            
                            sampled_architecture['patch_len_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                            
                        else:
                            
                            embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])
                    
                    elif 'stride' in cell_component:
                        if sampled_architecture['ts_module_nodes'][current_node]=='Patching':
                            selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                            
                            sampled_architecture['stride_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                            
                        else:
                            
                            embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])
                    
                    elif 'dff' in cell_component:
                        if sampled_architecture['ts_module_nodes'][current_node]=='MTSMixer':
                            selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                            
                            sampled_architecture['dff_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                            
                        else:
                            
                            embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])
                    
                    elif 'sampling' in cell_component:
                        if sampled_architecture['ts_module_nodes'][current_node]=='MTSMixer':
                            selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                action_head_idx=action_head_idx,
                                                                                hidden_state_environment=hidden_state_environment,
                                                                                embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                embeddings_cell_component=embeddings_cell_component)
                            
                            sampled_architecture['sampling_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                            
                        else:
                            
                            embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])
                    
                          

                    else:# 'activation' in cell_component:# and 'cell_inject' not in cell_component:
                        #Second Activation Function:
                        if 'second' in cell_component:
                            
                            if sampled_architecture['ts_module_nodes'][current_node] not in ['Attention','Patching']:
                                if '1.component' in cell_component:
                                    selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                        action_head_idx=action_head_idx,
                                                                                        hidden_state_environment=hidden_state_environment,
                                                                                        embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                        log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                        embeddings_cell_component=embeddings_cell_component)
                                    
                                    sampled_architecture['activations_second_nodes']['node_'+str(current_node)]=self.search_space_values[cell_component][selected_cell_component]
                                
                                elif 'action' in cell_component:
                                    selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                        action_head_idx=action_head_idx,
                                                                                        hidden_state_environment=hidden_state_environment,
                                                                                        embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                        log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                        embeddings_cell_component=embeddings_cell_component)
                                    if self.search_space_values[cell_component][selected_cell_component]!=None:
                                        sampled_architecture['activations_second_nodes']['node_'+str(current_node)]=sampled_architecture['activations_second_nodes']['node_'+str(current_node)]+'_'+self.search_space_values[cell_component][selected_cell_component]
                                else:#2.component
                                    if '_' in sampled_architecture['activations_second_nodes']['node_'+str(current_node)]:
                                        possible_options_second_component=self.search_space_values[cell_component].copy()
                                        possible_options_second_component.remove(sampled_architecture['activations_second_nodes']['node_'+str(current_node)].split('_')[0])

                                        selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                        action_head_idx=action_head_idx,
                                                                                        hidden_state_environment=hidden_state_environment,
                                                                                        embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                        log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                        embeddings_cell_component=embeddings_cell_component)
                                        sampled_architecture['activations_second_nodes']['node_'+str(current_node)]=sampled_architecture['activations_second_nodes']['node_'+str(current_node)]+'_'+possible_options_second_component[selected_cell_component]

                                    else:
                                        embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])


                            else:
                                embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])
                                

                        #First Activation Function:  
                        else:
                            if '1.component' in cell_component:
                                selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                    action_head_idx=action_head_idx,
                                                                                    hidden_state_environment=hidden_state_environment,
                                                                                    embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                    log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                    embeddings_cell_component=embeddings_cell_component)
                                
                                sampled_architecture['activations_nodes'].append(self.search_space_values[cell_component][selected_cell_component])
                            
                            elif 'action' in cell_component:
                                selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                    action_head_idx=action_head_idx,
                                                                                    hidden_state_environment=hidden_state_environment,
                                                                                    embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                    log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                    embeddings_cell_component=embeddings_cell_component)
                                if self.search_space_values[cell_component][selected_cell_component]!=None:
                                    sampled_architecture['activations_nodes'][-1]=sampled_architecture['activations_nodes'][-1]+'_'+self.search_space_values[cell_component][selected_cell_component]
                            
                            else:#2.component
                                if '_' in sampled_architecture['activations_nodes'][-1]:
                                    possible_options_second_component=self.search_space_values[cell_component].copy()
                                    possible_options_second_component.remove(sampled_architecture['activations_nodes'][-1].split('_')[0])

                                    selected_cell_component,embeddings_cell_component,log_prob_sum=self.forward_pass_single_action_head(cell_component=cell_component,
                                                                                    action_head_idx=action_head_idx,
                                                                                    hidden_state_environment=hidden_state_environment,
                                                                                    embedding_sampled_architecture=embedding_sampled_architecture,
                                                                                    log_prob_sum=log_prob_sum,action_probs=action_probs,cell_entropy=cell_entropy,
                                                                                    embeddings_cell_component=embeddings_cell_component)
                                    sampled_architecture['activations_nodes'][-1]=sampled_architecture['activations_nodes'][-1]+'_'+possible_options_second_component[selected_cell_component]
                            
                                else:
                                    embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])

                else:
                    #Skip forward pass with single action head:
                    embedding_sampled_architecture.append(self.embedding_none_matrix_values[cell_component])
                    

        if sampled_architecture["prediction_projections"]==None:
            sampled_architecture["prediction_projections"]=False
        
        
        if sampled_architecture["revin_nodes"]==[]:
            sampled_architecture["revin_nodes"]=[True]*sampled_architecture["nr_nodes"]
        
        # critic: evaluates being in the state s_t
        critic_value_function = self.value_head(hidden_state_environment)
        cell_entropy=np.mean(cell_entropy)
        return cell_entropy,action_probs, sampled_architecture,memory_state_environment,torch.stack(embedding_sampled_architecture,axis=1),log_prob_sum,critic_value_function
    

def finish_episode(actor_critic_model,optimizer_actor_critic,gamma,rewards_history):
    R=0
    actor_critic_model.rewards.extend(rewards_history)
    saved_actions =actor_critic_model.saved_actions
    
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in actor_critic_model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]])))

    # reset gradients
    optimizer_actor_critic.zero_grad()

    # sum up all the values of policy_losses and value_losses
    actor_critic_loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    print(black('Traianble Variables before gradient updates: ',['bold']))
    print(list(actor_critic_model.parameters())[0][0][:10])
    
    # perform backprop
    actor_critic_loss.backward(retain_graph=True)
    optimizer_actor_critic.step()
    
    print(black('Traianble Variables after gradient updates: ',['bold']))
    print(list(actor_critic_model.parameters())[0][0][:10])
    print(black('----------------------------------------------------------------------------',['bold']))
    print(black('----------------------------------------------------------------------------',['bold']),'\n\n')


    # reset rewards and action buffer
    del actor_critic_model.rewards[:]
    del actor_critic_model.saved_actions[:]


#GED:
def compute_nas_edit_distance(first_sampled_cell,second_sampled_cell,
                              input_seq_len,
                              input_channels,
                              min_max_deltas):
    
    
    #Computation of NAS Edit Distance between two architectures:
    nas_edit_distance_matrix=np.zeros(shape=(first_sampled_cell['nr_nodes'],
                                             second_sampled_cell['nr_nodes']))
    
    for first_idx_cell in range(0,nas_edit_distance_matrix.shape[0]):
        for second_idx_cell in range(0, nas_edit_distance_matrix.shape[1]):
            
            distance_values_array=[]
            #print(first_sampled_cell['ts_module_nodes'][first_idx_cell])
            #print(second_sampled_cell['ts_module_nodes'][second_idx_cell])
            #1. Masses distance:
            if 'Dnonlinear' not in first_sampled_cell['ts_module_nodes'][first_idx_cell] and 'Dnonlinear' not in second_sampled_cell['ts_module_nodes'][second_idx_cell]:
                percentual_distance_mass=np.abs(first_sampled_cell['d_model_nodes']['node_'+str(first_idx_cell)]-second_sampled_cell['d_model_nodes']['node_'+str(second_idx_cell)])
               #print(percentual_distance_mass)
                #print(min_max_deltas['d_model_nodes']['max_value']-min_max_deltas['d_model_nodes']['min_value'])
                percentual_distance_mass=percentual_distance_mass/(min_max_deltas['d_model_nodes']['max_value']-min_max_deltas['d_model_nodes']['min_value'])
            else:
                if 'Dnonlinear' in first_sampled_cell['ts_module_nodes'][first_idx_cell]:
                    if first_sampled_cell['ts_module_nodes'][first_idx_cell]=='Dnonlinear_conv1d':
                        mass_first_cell=input_seq_len*input_channels 
                    else:
                        mass_first_cell=input_seq_len*input_seq_len
                else:
                    mass_first_cell=first_sampled_cell['d_model_nodes']['node_'+str(first_idx_cell)]
                
                if 'Dnonlinear' in second_sampled_cell['ts_module_nodes'][second_idx_cell]:
                    if second_sampled_cell['ts_module_nodes'][second_idx_cell]=='Dnonlinear_conv1d':
                        mass_second_cell=input_seq_len*input_channels 
                    else:
                        mass_second_cell=input_seq_len*input_seq_len
                        #print("Mass Second Cell:",mass_second_cell)
                else:
                    mass_second_cell=second_sampled_cell['d_model_nodes']['node_'+str(second_idx_cell)]
                
                max_denom=np.max([mass_first_cell,mass_second_cell,min_max_deltas['d_model_nodes']['max_value']])
                min_denom=np.min([mass_first_cell,mass_second_cell,min_max_deltas['d_model_nodes']['min_value']])
                #print("Max Units in search space: ",min_max_deltas['d_model_nodes']['max_value'])
                
                #print("Min Units in search space: ",min_max_deltas['d_model_nodes']['min_value'])
                #print("Mass first cell:",mass_first_cell)
                #print("Normalizing denominator:",max_denom-min_denom)

                percentual_distance_mass=np.abs(mass_first_cell-mass_second_cell)/(max_denom-min_denom)
                
            distance_values_array.append(percentual_distance_mass)

            
            #2. Distance between activations:
            if '_' not in first_sampled_cell['activations_nodes'][first_idx_cell] and '_' not in second_sampled_cell['activations_nodes'][second_idx_cell]:
                activations_distance=1.0-int(first_sampled_cell['activations_nodes'][first_idx_cell]==second_sampled_cell['activations_nodes'][second_idx_cell])
                distance_values_array.append(activations_distance)
            
            elif '_' in first_sampled_cell['activations_nodes'][first_idx_cell] and '_' in second_sampled_cell['activations_nodes'][second_idx_cell]:
                
                first_cell_act_first_cmp,first_cell_act_action,first_cell_act_second_cmp=first_sampled_cell['activations_nodes'][first_idx_cell].split('_')
                second_cell_act_first_cmp,second_cell_act_action,second_cell_act_second_cmp=second_sampled_cell['activations_nodes'][second_idx_cell].split('_')
                
                first_cmp_dist=1.0-int(first_cell_act_first_cmp==second_cell_act_first_cmp)
                act_action_dist=1.0-int(first_cell_act_action==second_cell_act_action)
                second_cmp_dist=1.0-int(first_cell_act_second_cmp==second_cell_act_second_cmp)
                
                activations_distance=np.mean([first_cmp_dist,act_action_dist,second_cmp_dist])
                distance_values_array.append(activations_distance)
            
            else:
                if '_' in first_sampled_cell['activations_nodes'][first_idx_cell]:
                    first_cmp_dist=1.0-int(first_sampled_cell['activations_nodes'][first_idx_cell].split('_')[0]==second_sampled_cell['activations_nodes'][second_idx_cell])
                else:
                    first_cmp_dist=1.0-int(first_sampled_cell['activations_nodes'][first_idx_cell]==second_sampled_cell['activations_nodes'][second_idx_cell].split('_')[0])
                act_action_dist=1.0
                second_cmp_dist=1.0
                
                activations_distance=np.mean([first_cmp_dist,act_action_dist,second_cmp_dist])
                distance_values_array.append(activations_distance)

            #3. Distance between dropout rates:
            dropout_distance=np.abs(first_sampled_cell['dropout_nodes'][first_idx_cell]-second_sampled_cell['dropout_nodes'][second_idx_cell])
            dropout_distance=dropout_distance/np.abs(min_max_deltas['dropout']['max_value']-min_max_deltas['dropout']['min_value'])
            distance_values_array.append(dropout_distance)

            
            #5. Distance between ts module types:
            ts_modules_distance=1.0-int(first_sampled_cell['ts_module_nodes'][first_idx_cell]==second_sampled_cell['ts_module_nodes'][second_idx_cell])
            distance_values_array.append(ts_modules_distance)

            #Potentially move at the end:
            #6. Distance between skip connections: relevant only from second layer onwards:
            if (first_idx_cell>0.0 and second_idx_cell>0.0) and (first_idx_cell==second_idx_cell):#meaning add skip dist only for pairs
                #1-1 and 2-2:
                skip_connections_distance=1.0-int(first_sampled_cell['skip_connections'][first_idx_cell-1]==second_sampled_cell['skip_connections'][second_idx_cell-1])
                distance_values_array.append(skip_connections_distance) 
                
                

            #7.Distance between second Activations:
            second_act_first_cell=(first_sampled_cell['activations_second_nodes']['node_'+str(first_idx_cell)] if 'node_'+str(first_idx_cell) in first_sampled_cell['activations_second_nodes'].keys() else 'None')
            second_act_second_cell=(second_sampled_cell['activations_second_nodes']['node_'+str(second_idx_cell)] if 'node_'+str(second_idx_cell) in second_sampled_cell['activations_second_nodes'].keys() else 'None')
            if [second_act_first_cell,second_act_second_cell]!=['None','None']:
                
                if '_' not in second_act_first_cell and '_' not in second_act_second_cell:  
                    second_activations_distance=1.0-int(second_act_first_cell==second_act_second_cell) 
                    distance_values_array.append(second_activations_distance)

                elif '_' in second_act_first_cell and '_' in second_act_second_cell:
                    first_cell_second_act_first_cmp,first_cell_second_act_action,first_cell_second_act_second_cmp=second_act_first_cell.split('_')
                    second_cell_second_act_first_cmp,second_cell_second_act_action,second_cell_second_act_second_cmp=second_act_second_cell.split('_')
                    
                    second_act_first_cmp_dist=1.0-int(first_cell_second_act_first_cmp==second_cell_second_act_first_cmp)
                    second_act_action_dist=1.0-int(first_cell_second_act_action==second_cell_second_act_action)
                    second_act_second_cmp_dist=1.0-int(first_cell_second_act_second_cmp==second_cell_second_act_second_cmp)
                    
                    second_activations_distance=np.mean([second_act_first_cmp_dist,second_act_action_dist,second_act_second_cmp_dist])
                    distance_values_array.append(second_activations_distance)
            
                else:
                    if '_' in second_act_first_cell:
                        second_act_first_cmp_dist=1.0-int(second_act_first_cell.split('_')[0]==second_act_second_cell)
                    else:
                        second_act_first_cmp_dist=1.0-int(second_act_first_cell==second_act_second_cell.split('_')[0])
                    second_act_action_dist=1.0
                    second_act_second_cmp_dist=1.0
                    
                    second_activations_distance=np.mean([second_act_first_cmp_dist,second_act_action_dist,second_act_second_cmp_dist])
                    distance_values_array.append(second_activations_distance)


                

            
            #Kernel size in Dnonlinear-AvgPool and Dnonlinear-Conv1d:
            if  'Dnonlinear' in first_sampled_cell['ts_module_nodes'][first_idx_cell] or 'Dnonlinear' in second_sampled_cell['ts_module_nodes'][second_idx_cell]:
                if 'Dnonlinear' not in first_sampled_cell['ts_module_nodes'][first_idx_cell]:
                    avg_pool_ks_first_cell=0.0
                    avg_pool_ks_second_cell=second_sampled_cell['kernel_size_nodes']['node_'+str(second_idx_cell)]
                elif 'Dnonlinear' not in second_sampled_cell['ts_module_nodes'][second_idx_cell]:
                    avg_pool_ks_first_cell=first_sampled_cell['kernel_size_nodes']['node_'+str(first_idx_cell)]
                    avg_pool_ks_second_cell=0.0
                else:
                    avg_pool_ks_first_cell=first_sampled_cell['kernel_size_nodes']['node_'+str(first_idx_cell)]
                    avg_pool_ks_second_cell=second_sampled_cell['kernel_size_nodes']['node_'+str(second_idx_cell)]

                distance_avgpool_ks=np.abs(avg_pool_ks_first_cell-avg_pool_ks_second_cell)
                if avg_pool_ks_first_cell==0.0 or avg_pool_ks_second_cell==0.0:
                    distance_avgpool_ks=distance_avgpool_ks/min_max_deltas['kernel_size']['max_value']#same as subtracting 0.0 from max value
                else:
                    distance_avgpool_ks=distance_avgpool_ks/np.abs(min_max_deltas['kernel_size']['max_value']-min_max_deltas['kernel_size']['min_value'])
                distance_values_array.append(distance_avgpool_ks)

            #8. Stride and Patch Length in Patching:
            if  first_sampled_cell['ts_module_nodes'][first_idx_cell]=='Patching' or second_sampled_cell['ts_module_nodes'][second_idx_cell]=='Patching':
                if first_sampled_cell['ts_module_nodes'][first_idx_cell]!='Patching':
                    stride_first_cell=0.0
                    patch_len_first_cell=0.0
                    stride_second_cell=second_sampled_cell['stride_nodes']['node_'+str(second_idx_cell)]
                    patch_len_second_cell=second_sampled_cell['patch_len_nodes']['node_'+str(second_idx_cell)]
                elif second_sampled_cell['ts_module_nodes'][second_idx_cell]!='Patching':
                    stride_first_cell=first_sampled_cell['stride_nodes']['node_'+str(first_idx_cell)]
                    patch_len_first_cell=first_sampled_cell['patch_len_nodes']['node_'+str(first_idx_cell)]
                    stride_second_cell=0.0
                    patch_len_second_cell=0.0
                else:
                    stride_first_cell=first_sampled_cell['stride_nodes']['node_'+str(first_idx_cell)]
                    patch_len_first_cell=first_sampled_cell['patch_len_nodes']['node_'+str(first_idx_cell)]
                    stride_second_cell=second_sampled_cell['stride_nodes']['node_'+str(second_idx_cell)]
                    patch_len_second_cell=second_sampled_cell['patch_len_nodes']['node_'+str(second_idx_cell)]

                distance_stride=np.abs(stride_first_cell-stride_second_cell)
                distance_patch_len=np.abs(patch_len_first_cell-patch_len_second_cell)
                if stride_first_cell==0.0 or stride_second_cell==0.0:#then the same goes for patch len
                    distance_stride=distance_stride/min_max_deltas['stride']['max_value']#same as subtracting 0.0 from max value
                    distance_patch_len=distance_patch_len/min_max_deltas['patch_len']['max_value']#same as subtracting 0.0 from max value
                    
                else:
                    distance_stride=distance_stride/np.abs(min_max_deltas['stride']['max_value']-min_max_deltas['stride']['min_value'])
                    distance_patch_len=distance_patch_len/np.abs(min_max_deltas['patch_len']['max_value']-min_max_deltas['patch_len']['min_value'])
                    
                    
                distance_values_array.append(distance_stride)
                distance_values_array.append(distance_patch_len)

            #9. Sampling and Dff in MTSMixer:
            if  first_sampled_cell['ts_module_nodes'][first_idx_cell]=='MTSMixer' or second_sampled_cell['ts_module_nodes'][second_idx_cell]=='MTSMixer':
                if first_sampled_cell['ts_module_nodes'][first_idx_cell]!='MTSMixer':
                    dff_first_cell=0.0
                    sampling_first_cell=0.0
                    dff_second_cell=second_sampled_cell['dff_nodes']['node_'+str(second_idx_cell)]
                    sampling_second_cell=second_sampled_cell['sampling_nodes']['node_'+str(second_idx_cell)]
                elif second_sampled_cell['ts_module_nodes'][second_idx_cell]!='MTSMixer':
                    dff_first_cell=first_sampled_cell['dff_nodes']['node_'+str(first_idx_cell)]
                    sampling_first_cell=first_sampled_cell['sampling_nodes']['node_'+str(first_idx_cell)]
                    dff_second_cell=0.0
                    sampling_second_cell=0.0
                else:
                    dff_first_cell=first_sampled_cell['dff_nodes']['node_'+str(first_idx_cell)]
                    sampling_first_cell=first_sampled_cell['sampling_nodes']['node_'+str(first_idx_cell)]
                    dff_second_cell=second_sampled_cell['dff_nodes']['node_'+str(second_idx_cell)]
                    sampling_second_cell=second_sampled_cell['sampling_nodes']['node_'+str(second_idx_cell)]

                distance_dff=np.abs(dff_first_cell-dff_second_cell)
                distance_sampling=np.abs(sampling_first_cell-sampling_second_cell)
                if dff_first_cell==0.0 or dff_second_cell==0.0:#then the same goes for sampling
                    distance_dff=distance_dff/min_max_deltas['dff']['max_value']#same as subtracting 0.0 from max value
                    distance_sampling=distance_sampling/min_max_deltas['sampling']['max_value']#same as subtracting 0.0 from max value
                    
                else:
                    distance_dff=distance_dff/np.abs(min_max_deltas['dff']['max_value']-min_max_deltas['dff']['min_value'])
                    distance_sampling=distance_sampling/np.abs(min_max_deltas['sampling']['max_value']-min_max_deltas['sampling']['min_value'])
                    
                    
                distance_values_array.append(distance_dff)
                distance_values_array.append(distance_sampling)



            #Average all distance values:
            #print(distance_values_array)
            nas_edit_distance_matrix[first_idx_cell,second_idx_cell]=np.mean(distance_values_array)


    #Match the layers based on minimal distance:
    nas_edit_distance_array=[]

    #print('Matrix before matching: \n',nas_edit_distance_matrix)

    for i in range(np.min(nas_edit_distance_matrix.shape[0], nas_edit_distance_matrix[1])):
        min_row,min_col=np.where(nas_edit_distance_matrix==np.min(nas_edit_distance_matrix))
        match_nas_edit_distance=nas_edit_distance_matrix[min_row[0],min_col[0]]
        nas_edit_distance_matrix[min_row[0],:]=+np.inf
        nas_edit_distance_matrix[:,min_col[0]]=+np.inf
        nas_edit_distance_array.append(match_nas_edit_distance)

    #Account for not matched layers if one of the two nas models had more layers:    
    if nas_edit_distance_matrix.shape[0] != nas_edit_distance_matrix.shape[1]:
         nas_edit_distance_array.append(1.0)

    #Account for differences in prediction projection layers:
    nas_edit_distance_array.append(1.0-int(first_sampled_cell['prediction_projections']==second_sampled_cell['prediction_projections']))
    
    #Account for different number of hidden layers:
    nas_edit_distance_array.append(1.0-int(first_sampled_cell['nr_nodes']==second_sampled_cell['nr_nodes']))
    
    nas_edit_distance_array=np.array(nas_edit_distance_array)
    nas_edit_distance=nas_edit_distance_array.mean()  
    return nas_edit_distance




#Other functions:
def _acquire_device(use_gpu=True,use_multi_gpu=False,devices='0, 1',gpu=0):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            gpu) if not use_multi_gpu else devices
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
    return device



    

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #if self.verbose:
            #print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss



