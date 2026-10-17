import sys
sys.path.append("..")
from pathlib import Path

# Parent of notebook current working directory
parent_dir = Path.cwd().parent

from Benchmark_Models.utils import  (_acquire_device, _select_optimizer, _select_criterion,create_args, EarlyStopping, 
                                     adjust_learning_rate, get_peak_lr, make_param_groups)
from typing import Callable, Any
import pickle
import time
import copy
import warnings
from functools import wraps
from permetrics import RegressionMetric

import pandas as pd
import os 
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import numpy as np
from numpy import array
from sklearn.preprocessing import StandardScaler

import argparse
from dask.distributed import Client
import matplotlib.pyplot as plt
from datetime import datetime,timedelta

from Benchmark_Models.Build_NAS_models import default_build_model_fn


class NAS_Forecasting():
    """A class for training and evaluating selected NAS models.
    
    
    Methods:
    --------
    perform_walk_forward_validation():
        Generates the predictions using the walk-forward validation approach for all models
        included in models_dataloaders_dict.
    train_model(model,args,current_train_dataloader,criterion,device,
        model_optim,current_model,EarlyStopping):
        Trains each model on the specified subset saved in the current_train_dataloader.
    test_model(model,args,current_test_dataloader,device,
        current_model,reverse_rescale,current_scaler)
        Tests the trained model on the current out-of-time subset.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 dataloader: dict,
                 model_configs: dict,
                 build_NAS_model=default_build_model_fn,
                 nr_inputs=19,
                 nr_outputs=19,
                 reverse_rescale=False,
                 use_gpu=True):
        """
        Initializes the NAS_Models class with the provided parameters.
        
        df: pd.DataFrame
            The input energy dataframe containing the data.
        dataloader: dict
            A dictionary containing the dataloaders for training and testing.
        model_configs: dict
            A dictionary containing the configurations for the NAS model.
        build_NAS_model: Callable
            A function to build the NAS tree model based on the provided configurations."""

        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs

        self.df = df
        self.dataloader = dataloader
        self.build_NAS_model = build_NAS_model
        self.model_configs = model_configs
        self.reverse_rescale = reverse_rescale
        self.use_gpu = use_gpu

        # indices for Dask client to parallelize the walk-forward validation for each model configuration
        self.config_indices = list(range(len(model_configs)))


    def train_model(self, model, args, current_train_dataloader, criterion, device,
                    model_optim, EarlyStopping):
        """
        Trains the provided model on the specified training dataloader.
        
        Parameters:
        -----------
        model: torch.nn.Module
            The model to be trained.
        args: argparse.Namespace
            The command-line arguments containing training configurations.
        current_train_dataloader: torch.utils.data.DataLoader
            The dataloader for the current training subset.
        criterion: torch.nn.Module
            The loss function used for training.
        device: torch.device
            The device (CPU or GPU) to perform computations on.
        model_optim: torch.optim.Optimizer
            The optimizer used for updating model parameters.
        current_model: str
            The name of the current model being trained.
        EarlyStopping: EarlyStopping
            An instance of the EarlyStopping class to monitor validation loss and stop training early if necessary.
        
        Returns:
        --------
        mean_train_loss: float
            The mean training loss over all epochs.
        model: torch.nn.Module
            The trained model after the training process.
        """
        model.train()
        train_loss = []
        iter_count = 0

        path = os.path.join(args.checkpoints, "NAS_Benchmarking_training")
        if not os.path.exists(path):
            os.makedirs(path)

        # steps_per_epoch = len(current_train_dataloader)
        # warmup_epochs = args.warmup_epochs
        # warmup_steps = max(1, steps_per_epoch * warmup_epochs)

        # warmup_sched = None 
        # if warmup_epochs > 0:
        #     warmup_sched = torch.optim.lr_scheduler.LambdaLR(
        #         model_optim, lr_lambda=lambda step: min(1.0, step / warmup_steps)
        #     )
        # plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     model_optim, mode='min', factor=0.8, patience=1, threshold=1e-4
        # )

        early_stopping = EarlyStopping(patience=args.patience, delta=0.01, verbose=True)
                
        epochs = args.train_epochs
        if args.use_amp:
            grad_scaler = torch.amp.GradScaler(device="cuda")

        global_step = 0
        for epoch in range(0,epochs):
            for i, train_item in enumerate(current_train_dataloader):
                
                # Get train items based on current model
                batch_x, batch_y, batch_x_mark, batch_y_mark,index=train_item
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                # forward pass
                outputs = model(batch_x)

                # Batch x Seq_len x n_features
                loss = criterion(outputs, batch_y)

                # train loss accumulated
                train_loss.append(loss.item())
                
                
                # Backward pass:
                if args.use_amp:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(model_optim)
                    grad_scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                #end

                # if warmup_sched is not None and global_step < warmup_steps:
                #     warmup_sched.step()

                global_step += 1


            # if global_step >= warmup_steps:
            #     plateau_sched.step(np.mean(train_loss))

            # if epoch % 10 == 0:
            #     print(f"Epoch: {epoch}, Training Loss: {np.mean(train_loss)}")

            # Apply Early Stopping Callback:
            early_stopping(np.mean(train_loss), model, path)
            if early_stopping.early_stop:
                # print(f"Early stopping at epoch {epoch}")
                break

        mean_train_loss = np.mean(train_loss)

        return mean_train_loss, model


    def test_model(self, 
                   model: torch.nn.Module, 
                   args: argparse.ArgumentParser, 
                   current_test_dataloader: torch.utils.data.DataLoader, 
                   device: torch.device,
                   reverse_rescale: bool, 
                   current_scaler=Any|None)->list:
        """
        Tests the trained model on the specified test dataloader.
        
        Parameters:
        -----------
        model: torch.nn.Module
            The trained model to be evaluated.
        args: argparse.Namespace
            The command-line arguments containing testing configurations.
        current_test_dataloader: torch.utils.data.DataLoader
            The dataloader for the current test subset.
        device: torch.device
            The device (CPU or GPU) to perform computations on.
        current_model: str
            The name of the current model being tested.
        reverse_rescale: bool
            A flag indicating whether to reverse rescale the predictions.
        current_scaler: sklearn.preprocessing.StandardScaler or similar
            The scaler used for reversing the scaling of predictions if reverse_rescale is True.
        
        Returns:
        --------
        mean_test_loss: float
            The mean test loss over all batches in the test dataloader.
        predictions: np.ndarray
            The predictions made by the model on the test data.
        """

        model.eval()
        test_subset_predictions = []
        test_subset_true_values = []

        with torch.no_grad():
            for i, test_item in enumerate(current_test_dataloader):
                batch_x, batch_y, batch_x_mark, batch_y_mark,index=test_item
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                # forward pass
                outputs = model(batch_x)

                # Store predictions and true values
                true_batch_values = batch_y.detach().cpu().numpy()
                batch_predictions = outputs.detach().cpu().numpy()


                if reverse_rescale:
                    true_batch_values = np.array([current_scaler.inverse_transform(tr) for tr in true_batch_values])
                    batch_predictions = np.array([current_scaler.inverse_transform(pr) for pr in batch_predictions])
 
                test_subset_true_values.append(true_batch_values)
                test_subset_predictions.append(batch_predictions)

        return test_subset_true_values, test_subset_predictions


    def parallel_walk_forward_validation(self, config_idx, **kwargs):

        sys.argv = ['']

        seq_len = kwargs['special_parameters']['seq_len']
        label_len = kwargs['special_parameters']['label_len']
        pred_len = kwargs['special_parameters']['pred_len']

        nr_datapoints_train=kwargs['special_parameters']['nr_datapoints_train']
        nr_datapoints_test=kwargs['special_parameters']['nr_datapoints_test']
        batch_size=kwargs['special_parameters']['batch_size']
        split_indices=kwargs['special_parameters']['split_indices']

        tree_config = self.model_configs[config_idx]
        dataloaders_dict = self.dataloader

        args = create_args(current_model='NAS_Tree',
                           df=self.df,
                           use_gpu=self.use_gpu,
                           seq_len=seq_len,
                           label_len=label_len,
                           pred_len=pred_len)

        args.patience = 1

        device = _acquire_device(args=args)

        model = self.build_NAS_model(tree_config = tree_config, 
                                     nr_inputs = self.nr_inputs, 
                                     nr_outputs = self.nr_outputs,
                                     seq_len=seq_len,
                                     pred_len=pred_len).float()
        model.to(device)

        # select optimizer and criterion based on tree setup
        nr_nodes = tree_config['nr_nodes']
        # peak_lr = get_peak_lr(n_nodes=nr_nodes, base_lr=7e-4)
        peak_lr = 0.001
        # params = make_param_groups(model=model, peak_lr=peak_lr)
        params = make_param_groups(model=model, peak_lr=peak_lr)

        model_optim = torch.optim.Adam(params)
        criterion = _select_criterion()

        # import EarlyStopping
        current_early_stopping = EarlyStopping


        #### Dictionary for Error ####
        error_results = {}

        start_training_time = time.time()

        trained_model = model
        for split_idx in range(len(split_indices)):
            current_subset = 'Train_Test' + str(split_indices[split_idx])
            current_loaders = dataloaders_dict[current_subset]

            current_train_loader = current_loaders['Train_Set_Loader']
            current_test_loader = current_loaders['Test_Set_Loader']
            current_scaler = current_loaders['Scaler']

            avg_train_loss, trained_model = self.train_model(
                model=(model if split_idx == 0 else trained_model),
                args=args,
                current_train_dataloader=current_train_loader,
                criterion=criterion,
                device=device,
                model_optim=model_optim,
                EarlyStopping=current_early_stopping,
            )

            model_batch_test_true, model_batch_test_predictions = self.test_model(
                model=trained_model,
                args=args,
                current_test_dataloader=current_test_loader,
                device=device,
                reverse_rescale=self.reverse_rescale,
                current_scaler=(current_scaler if self.reverse_rescale else None)
            )

            split_key = 'Train_Test' + str(split_idx+1)
            error_results[split_key] = [
                avg_train_loss,
                model_batch_test_true,
                model_batch_test_predictions
            ]

        end_training_time = time.time()

        return [config_idx, error_results, end_training_time-start_training_time]

    
    def execute_parallel_training_testing(self, seq_len, nr_datapoints_train, nr_datapoints_test, split_indices,
                                          batch_size=32, n_workers=10):

        client = Client(n_workers=n_workers, threads_per_worker=1)
        print(f"Client dashboard link: {client.dashboard_link}")

        futures = client.map(
            self.parallel_walk_forward_validation,
            self.config_indices,
            special_parameters={
                'seq_len': seq_len,
                'pred_len': seq_len,
                'label_len': int(seq_len/2),
                'nr_datapoints_train':nr_datapoints_train,
                'nr_datapoints_test':nr_datapoints_test,
                'batch_size': batch_size,
                'split_indices': split_indices,
            },
        )

        model_results = client.gather(futures)

        try:
            client.close()
        except Exception as e:
            print("Opening a new client for next Experiment!")

        all_tree_results = {}

        for result in model_results:
            config_idx, error_results, training_testing_time = result

            all_tree_results[f'tree_{config_idx}'] = {
                'Results': error_results,
                'Training_Testing_Time': training_testing_time,
                'tree_config': self.model_configs[config_idx],
            }

        return all_tree_results






class NAS_TopModels_Runner():
    def __init__(self, 
                 model_configs, 
                 build_NAS_model=default_build_model_fn,
                 nr_inputs = 19,
                 nr_outputs = 19):

        self.nr_timesteps=[48,96,192]
        # self.nr_timesteps=[192]
        self.nr_datapoints=[9984,11500,13956]
        self.nr_splits=[20,19,17]
        # self.nr_datapoints=[13956]
        # self.nr_splits=[17] 

        self.nr_datapoints_train=[1904,2000,2192]
        self.nr_datapoints_test=[404,500,692]
        # self.nr_datapoints_train=[2192]
        # self.nr_datapoints_test=[692]

        self.split_indices={
                    48:np.arange(4,21,1).tolist(),
                    96:np.arange(3,20,1).tolist(),
                    192:np.arange(3,18,1).tolist()}

        self.frame_names=['EU_Energy_Prices_Time_Series_']
                        #   ,
                        #   'EU_Energy_Time_Series_']
                        #   'EU_Total_Load_Time_Series_']

        self.dataloader_names=['EU_Energy_Prices_Dataloader_torch']
                        # 'Energy_Data_Dataloader_torch']
                        # 'EU_Total_Load_Dataloader_torch']

        self.model_configs = model_configs
        self.build_NAS_model = build_NAS_model

        self.nr_inputs = nr_inputs
        self.nr_outputs = nr_outputs
        self.parent_dir = parent_dir
 
    def run_top_nas_forecasters(self):
        for nr_run in np.arange(1, 11):
            for frame_name_idx in range(len(self.frame_names)):
                frame_name = self.frame_names[frame_name_idx]
                print('Current Frame Name: ', frame_name)
 
                for idx in range(len(self.nr_timesteps)):
                    current_nr_timesteps = self.nr_timesteps[idx]
                    current_nr_datapoints_train = self.nr_datapoints_train[idx]
                    current_nr_datapoints_test = self.nr_datapoints_test[idx]
                    print('Current number of Input and Output Timesteps: ', current_nr_timesteps, '\n')
 
                    dt = self.dataloader_names[frame_name_idx]
                    dataloader_name = dt + '_' + str(current_nr_timesteps) + '_' + \
                        ('Transfer_Learn' if dt == 'Energy_Data_Dataloader_torch' else 'Train') + '.pkl'
                    with open(f'{self.parent_dir}/Energy_Production_Data/EU_Time_Series/{dataloader_name}', 'rb') as f:
                        current_dataloader = pickle.load(f)
 
                    with open(f'{self.parent_dir}/Energy_Production_Data/EU_Time_Series/'
                              f'{frame_name}{current_nr_timesteps}_'
                              f'{"Transfer_Learn" if frame_name == "EU_Energy_Time_Series_" else "Train"}.pkl', 'rb') as f:
                        current_eu_energy_frame = pickle.load(f)
 
                    nas_forecasting = NAS_Forecasting(
                        df=current_eu_energy_frame,
                        dataloader=current_dataloader,
                        model_configs=self.model_configs,
                        build_NAS_model=self.build_NAS_model,
                        nr_inputs=self.nr_inputs,
                        nr_outputs=self.nr_outputs,
                        reverse_rescale=False,
                        use_gpu=True,
                    )
 
                    all_models_results = nas_forecasting.execute_parallel_training_testing(
                        seq_len=current_nr_timesteps,
                        nr_datapoints_train=current_nr_datapoints_train,
                        nr_datapoints_test=current_nr_datapoints_test,
                        split_indices=self.split_indices[current_nr_timesteps],
                        batch_size=32,
                    )
 
                    for current_model in all_models_results.keys():
                        with open(f'{self.parent_dir}/Results_Evaluation/'
                                  f'{current_model}_{frame_name}{current_nr_timesteps}_{nr_run}.pkl', 'wb') as f:
                            pickle.dump(all_models_results[current_model], f)
 
        # --- aggregate 10 runs into an average file (point predictions) ---
        for k in range(len(self.model_configs)):
            current_model = f'tree_{k}'
            for current_nr_timesteps in self.nr_timesteps:
                for frame_name in self.frame_names:
                    with open(f'{self.parent_dir}/Results_Evaluation/'
                              f'{current_model}_{frame_name}{current_nr_timesteps}_1.pkl', 'rb') as f:
                        current_model_avg_results = copy.deepcopy(pickle.load(f))
 
                    for current_run in np.arange(2, 11):
                        with open(f'{self.parent_dir}/Results_Evaluation/'
                                  f'{current_model}_{frame_name}{current_nr_timesteps}_{current_run}.pkl', 'rb') as f:
                            current_model_current_run = pickle.load(f)
 
                        current_model_avg_results['Training_Testing_Time'] += current_model_current_run['Training_Testing_Time']
                        if current_run == 10:
                            current_model_avg_results['Training_Testing_Time'] /= 10.0
 
                        for tr_test_subset in current_model_current_run['Results'].keys():
                            current_run_pred_batches = current_model_current_run['Results'][tr_test_subset][2]
                            avg_pred_batches = current_model_avg_results['Results'][tr_test_subset][2]
                            for batch_idx in range(len(avg_pred_batches)):
                                avg_pred_batches[batch_idx] = avg_pred_batches[batch_idx] + current_run_pred_batches[batch_idx]
                                if current_run == 10:
                                    avg_pred_batches[batch_idx] = avg_pred_batches[batch_idx] / 10.0
                            current_model_avg_results['Results'][tr_test_subset][2] = avg_pred_batches
 
                    with open(f'{self.parent_dir}/Results_Evaluation/'
                              f'{current_model}_{frame_name}{current_nr_timesteps}_Avg_10_Runs.pkl', 'wb') as f:
                        pickle.dump(current_model_avg_results, f)