import sys
sys.path.append("..")
from pathlib import Path

# Parent of notebook current working directory
parent_dir = Path.cwd().parent


from typing import Callable, Any
import pandas as pd
import os 
import torch
import torch.nn as nn
import numpy as np
from numpy import array
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
import argparse
from dask.distributed import Client
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import time
import pickle
import time
import copy
import warnings
from functools import wraps
from permetrics import RegressionMetric

from Benchmark_Models.Multivariate_Models import Multivariate_Forecasting
from Benchmark_Models.Univariate_Models import Univariate_Forecasting
from Benchmark_Models.utils import _acquire_device,create_args, EarlyStopping
from Benchmark_Models.time_series_models.LTSF_Linear.LTSF_Models.DLinear import Model as DLinear
from Benchmark_Models.time_series_models.PatchTST.PatchTST_Supervised.PatchTST_Model.PatchTST import Model as PatchTST
from Benchmark_Models.time_series_models.Basisformer.Basisformer_Model.Basisformer import Basisformer
from Benchmark_Models.time_series_models.TSMixer.TSMixer_Model.TSMixer_Torch import TSMixerModel
from Benchmark_Models.time_series_models.MTSMixer.MTSMixer_Model.MTSMixer_Torch import MTSMixerModel

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import autogluon

#Define the search space for dragon:
from dragon.search_space.bricks_variables import mlp_var,recurrence_2d,recurrence_1d, dropout, identity_var, mlp_const_var, conv_1d, conv_2d, pooling_1d, pooling_2d, attention_1d, attention_2d, norm_1d, norm_2d
from dragon.search_space.bricks_variables import operations_var, dag_var, node_var
from dragon.search_space.base_variables import ArrayVar
from dragon.search_operators.base_neighborhoods import ArrayInterval
from dragon.search_algorithm.ssea import SteadyStateEA
from dragon.utils.plot_functions import load_archi
from dragon.search_space.dag_variables import Node
from dragon.search_space.bricks import MLP




class NAS_Benchmarking():
    """
    A class implementing the benchmarks of our novel NAS framework.

    Parameters:
    ----------
    
    Attributes:
    ----------
    nr_timesteps: list
        Number of time steps in target window setting.

    nr_datapoints: list
        Number of total time steps with indices corresponding to each target window setting.

    nr_splits: list
        Number of train-test splits with indices corresponding to each target window setting.

    nr_datapoints_train: list
        Number of time steps in each train set with indices corresponding to each target window setting.

    nr_datapoints_test: list
        Number of time steps in each test set with indices corresponding to each target window setting. 

    split_indices: dict
        Indices of train-test splits according to number of time steps to predict in the future.

    frame_names: list
        Names of both dataframe types (total supplier load and RE+Fossil Fuel)

    dataloader_names: list
        Names of both dataloader types (total supplier load and RE+Fossil Fuel)

    time_series_models_dict: dict 
        Dictionary containing a callable for each baseline that requires data-specific training.


    Methods
    -------
    run_multivariate_forecasters():
        Implements  Basisformer, PatchTST, DLinear, TSMixer, MTSMixer.
    run_pretrained_forecasters():
        Implements TimesFM, Time-MoE, GPT4TS, Sundial.
    run_autogluon_nas()
        Implements Autogluon-NAS.
    run_dragon_nas()
        Implements Dragon-NAS.

    """
    def __init__(self):


        self.nr_timesteps=[48,96,192]
        self.nr_datapoints=[9984,11500,13264]
        self.nr_splits=[20,19,16]

        self.nr_datapoints_train=[1904,2000,2192]
        self.nr_datapoints_test=[404,500,692]

        self.split_indices={48:np.arange(4,21,1).tolist(),
                    96:np.arange(3,20,1).tolist(),
                    192:np.arange(1,17,1).tolist()}

        self.frame_names=['EU_Energy_Prices_Time_Series_',
                          'EU_Energy_Time_Series_']
                        #   'EU_Total_Load_Time_Series_']

        self.dataloader_names=['EU_Energy_Prices_Dataloader_torch',
                        'Energy_Data_Dataloader_torch']
                        # 'EU_Total_Load_Dataloader_torch']

        self.time_series_models_dict={ 'DLinear':DLinear,
                                        'PatchTST':PatchTST,
                                        'Basisformer':Basisformer,
                                        'TSMixer':TSMixerModel,
                                        'MTSMixer':MTSMixerModel}
        
        super(NAS_Benchmarking,self).__init__()

    def run_multivariate_forecasters(self):

        
        #Perform multivariate forecasting with 5 baseline models
        current_multivariate_models=['Basisformer','PatchTST',
                                    'DLinear','TSMixer','MTSMixer']
        
        for nr_run in np.arange(1,11):
            for frame_name_idx in range(0,len(self.frame_names)):
                frame_name=self.frame_names[frame_name_idx]

                print('Current Frame Name: ',frame_name)

                for idx in range(0,len(self.nr_timesteps)):
                    
                    #Create train and test splits based on number of timesteps to predict:
                    current_nr_timesteps=self.nr_timesteps[idx]
                    current_nr_datapoints_train=self.nr_datapoints_train[idx]
                    current_nr_datapoints_test=self.nr_datapoints_test[idx]

                    print('Current number of Input and Output Timesteps: ',current_nr_timesteps,'\n')

                    #Import the datasets based on the number of timesteps to predict:
                    dt=self.dataloader_names[frame_name_idx]
                    dataloader_name=dt+'_'+str(current_nr_timesteps)+'_'+('Transfer_Learn' if dt=='Energy_Data_Dataloader_torch' else 'Train')+'.pkl'
                    with open(str(parent_dir)+'/Energy_Production_Data/EU_Time_Series/'+dataloader_name,'rb') as f:
                        current_dataloader=pickle.load(f)

                    with open(str(parent_dir)+'/Energy_Production_Data/EU_Time_Series/'+frame_name+str(current_nr_timesteps)+'_'+('Transfer_Learn' if frame_name=='EU_Energy_Time_Series_' else 'Train')+'.pkl','rb') as f:
                        current_eu_energy_frame=pickle.load(f) 

                    
                    #Parallel model training and testing of multivariate forecasting techniques:
                    #Data Preprocessing takes place in the function executing the parallelization
                    #as otherwise, when the datasets become large, dask has difficulties with serializing the data:
                    multivariate_forecasting=Multivariate_Forecasting(df=current_eu_energy_frame,
                                                                    dataloader=current_dataloader,
                                                                    model_names=current_multivariate_models,
                                                                    time_series_models_dict=self.time_series_models_dict,
                                                                    reverse_rescale=False,
                                                                    use_gpu=True)

                    #Execute parallelization:
                    all_models_results=multivariate_forecasting.execute_parallel_training_testing(seq_len=current_nr_timesteps,
                                                                                                nr_datapoints_train=current_nr_datapoints_train,
                                                                                                nr_datapoints_test=current_nr_datapoints_test,
                                                                                                batch_size=32,
                                                                                                split_indices=self.split_indices[current_nr_timesteps])
                    
                    #Save results per model per number of timesteps:
                    for current_model in all_models_results.keys():
                        with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_'+str(nr_run)+'.pkl','wb') as f:
                                pickle.dump(all_models_results[current_model],f)

        #Create an aggregated file from all 10 prediction files:
        for current_model in current_multivariate_models:
            for current_nr_timesteps in self.nr_timesteps:
                for frame_name in self.frame_names:
                    
                    nr_run=1
                    #Collect average results from 10 runs:
                    with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_'+str(nr_run)+'.pkl','rb') as f:
                            current_model_first_run=pickle.load(f)

                    current_model_avg_results=copy.deepcopy(current_model_first_run)
                    #print('First predicted sequence from 1.run: \n',current_model_avg_results['Results']['Train_Test1'][2][0])
                    #print('Current model 1. run training and testing time: ',current_model_avg_results['Training_Testing_Time'])

                    for current_run in np.arange(2,11):
                        with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_'+str(current_run)+'.pkl','rb') as f:
                            current_model_current_run=pickle.load(f)

                        #Compute average training testing time:
                        current_model_avg_results['Training_Testing_Time']=current_model_avg_results['Training_Testing_Time']+current_model_current_run['Training_Testing_Time']
                        if current_run==10:
                            current_model_avg_results['Training_Testing_Time']=current_model_avg_results['Training_Testing_Time']/10.0
                        
                        #Compute average batch predictions:
                        for tr_test_subset in current_model_current_run['Results'].keys():
                            current_run_pred_batches=current_model_current_run['Results'][tr_test_subset][2]
                            avg_pred_batches=current_model_avg_results['Results'][tr_test_subset][2]

                            for batch_idx in range(0,len(avg_pred_batches)):
                                avg_pred_batches[batch_idx]=avg_pred_batches[batch_idx]+current_run_pred_batches[batch_idx]
                                if current_run==10:
                                    avg_pred_batches[batch_idx]=avg_pred_batches[batch_idx]/10.0
                            
                            current_model_avg_results['Results'][tr_test_subset][2]=avg_pred_batches

                    #Save current results:
                    with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_Avg_10_Runs.pkl','wb') as f:
                            pickle.dump(current_model_avg_results,f)


    def run_pretrained_forecasters(self):
        #Make 10 runs with pretrained baselines:
        gpt4ts_args=create_args(current_model='GPT4TS')

        device=_acquire_device(use_gpu=True)
        pretrained_models=["Sundial",
                        'TimeMoe',"TimesFM",
                        "GPT4TS"]

        for nr_run in np.arange(1,11,1):
            
            for pretrained_model in pretrained_models:
                
                if nr_run>1 and pretrained_model in ['TimeMoe',"TimesFM"]:
                    continue
                else:
                    print("Current pretrained Model: ",pretrained_model)
                    
                    for frame_name_idx in range(0,len(self.frame_names)):
                        frame_name=self.frame_names[frame_name_idx]
                        
                        print('- Current Frame Name: ',frame_name)
                        dataset_dict={}

                        for idx in range(0,len(self.nr_timesteps)):
                            
                            #Create train and test splits based on number of timesteps to predict:
                            current_nr_timesteps=self.nr_timesteps[idx]
                            current_nr_datapoints_train=self.nr_datapoints_train[idx]
                            current_nr_datapoints_test=self.nr_datapoints_test[idx]

                            #Import the datasets based on the number of timesteps to predict:
                            dt=self.dataloader_names[frame_name_idx]
                            dataloader_name=dt+'_'+str(current_nr_timesteps)+'_'+('Transfer_Learn' if dt=='Energy_Data_Dataloader_torch' else 'Train')+'.pkl'
                            with open(str(parent_dir)+'/Energy_Production_Data/EU_Time_Series/'+dataloader_name,'rb') as f:
                                current_dataloader=pickle.load(f)

                            #Import the datasets based on the number of timesteps to predict:
                            with open(str(parent_dir)+'/Energy_Production_Data/EU_Time_Series/'+frame_name+str(current_nr_timesteps)+'_'+('Transfer_Learn' if frame_name=='EU_Energy_Time_Series_' else 'Train')+'.pkl','rb') as f:
                                current_eu_energy_frame=pickle.load(f)

                            dataset_dict[idx]={'df':current_eu_energy_frame,
                                            'current_dataloader':current_dataloader,
                                                    'nr_timesteps':current_nr_timesteps,
                                                    'nr_datapoints_train':current_nr_datapoints_train,
                                                    'nr_datapoints_test':current_nr_datapoints_test,
                                                    'split_indices':self.split_indices[current_nr_timesteps]}
                            
                        #TimesFM: univariate forecasting
                        univariate_forecasting=Univariate_Forecasting(dataset_dict=dataset_dict,
                                                                    #dataloader=current_dataloader,
                                                                    device=device,
                                                                    pretrained_model=pretrained_model,
                                                                    args=(gpt4ts_args if pretrained_model=="GPT4TS" else None),
                                                                    num_samples=(10 if pretrained_model in ["GPT4TS","Sundial"] else None))
                            
                        all_univariate_forecasting_results=univariate_forecasting.execute_parallel_univariate_forecasting(batch_size=32)

                        for univariate_results in all_univariate_forecasting_results:
                            #previous save directory: 'evaluation/Results'
                            with open(str(parent_dir)+'/Results_Evaluation/'+pretrained_model+'_'+frame_name+str(dataset_dict[univariate_results[0]]['nr_timesteps'])+'_'+str(nr_run)+'.pkl','wb') as f:
                                pickle.dump(univariate_results[1],f)

        #Aggregate the results from the probabilistic baselines:
        current_multivariate_models=["Sundial","GPT4TS"]
        for current_model in current_multivariate_models:
            for current_nr_timesteps in self.nr_timesteps:
                for frame_name in self.frame_names:
                    
                    nr_run=1
                    #Collect average results from 10 runs:
                    with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_'+str(nr_run)+'.pkl','rb') as f:
                            current_model_first_run=pickle.load(f)

                    current_model_avg_results=copy.deepcopy(current_model_first_run)
                    

                    for current_run in np.arange(2,11):
                        with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_'+str(current_run)+'.pkl','rb') as f:
                            current_model_current_run=pickle.load(f)

                        #Compute average training testing time:
                        current_model_avg_results['Training_Testing_Time']=current_model_avg_results['Training_Testing_Time']+current_model_current_run['Training_Testing_Time']
                        if current_run==10:
                            current_model_avg_results['Training_Testing_Time']=current_model_avg_results['Training_Testing_Time']/10.0
                        
                        #Compute average batch predictions:
                        for tr_test_subset in current_model_current_run['Results'].keys():
                            current_run_pred_batches=current_model_current_run['Results'][tr_test_subset][1]
                            avg_pred_batches=current_model_avg_results['Results'][tr_test_subset][1]

                            #Here batch_idx=seq_idx
                            for batch_idx in range(0,len(avg_pred_batches)):
                                avg_pred_batches[batch_idx]=avg_pred_batches[batch_idx]+current_run_pred_batches[batch_idx]
                                if current_run==10:
                                    avg_pred_batches[batch_idx]=avg_pred_batches[batch_idx]/10.0
                            
                            current_model_avg_results['Results'][tr_test_subset][1]=avg_pred_batches

                    #Save current results:
                    with open(str(parent_dir)+'/Results_Evaluation/'+current_model+'_'+frame_name+str(current_nr_timesteps)+'_Avg_10_Runs.pkl','wb') as f:
                            pickle.dump(current_model_avg_results,f)

    def run_autogluon_nas(self):

        nr_datapoints_train={48:1904,
                            96:2000,
                            192:2192}

        nr_datapoints_test={48:404,
                            96:500,
                            192:692}

        for nr_run in np.arange(1,11):

            nas_results_dict={}
            nas_results_dict['NASTrain_Energy_Data']={48:{'Results':{},
                                                    'Training_Testing_Time':None},
                                                    96:{'Results':{},
                                                    'Training_Testing_Time':None},
                                                    192:{'Results':{},
                                                    'Training_Testing_Time':None}}

            nas_results_dict['TL_Total_Load']={48:{'Results':{},
                                                'Training_Testing_Time':None},
                                                96:{'Results':{},
                                                'Training_Testing_Time':None},
                                                192:{'Results':{},
                                                'Training_Testing_Time':None}}


            for seq_len in self.split_indices.keys():

                #Loop through both time series types:
                ts_frames=['EU_Energy_Prices_Time_Series_', 'EU_Energy_Time_Series_'] #,'EU_Total_Load_Time_Series_']
                for ts_frame in ts_frames:
                
                    #Import relevant sequence-time series_
                    with open(str(parent_dir)+"/"+ts_frame+str(seq_len)+'_'+('Transfer_Learn' if ts_frame=='EU_Energy_Time_Series_' else 'Train')+'.pkl','rb') as f:
                        eu_df=pickle.load(f)

                    print('Current Time Series Frame: ',ts_frame+str(seq_len)+'_'+('Transfer_Learn' if ts_frame=='EU_Energy_Time_Series_' else 'Train')+'.pkl')
                    
                    #Get all indices:
                    all_date_indices=eu_df.index.tolist()
                    print('Current Sequence Length: ',seq_len)
                    
                    total_training_testing_time_current_seq_len=0.0
                    reduced_total_training_testing_time_current_seq_len=0.0

                    for split_idx in range(0,len(self.split_indices[seq_len][1:-1])):
                        test_idx=self.split_indices[seq_len][1:-1][split_idx]
                        print('Current Test Subset: ','Test_Set'+str(test_idx+1))
                        current_test_subset=eu_df[eu_df['Subset']=='Test_Set'+str(test_idx+1)]
                        
                        train_end=all_date_indices.index(current_test_subset.index[0])
                        print('Train End: ',train_end)
                        train_start=np.max([train_end-nr_datapoints_train[seq_len]-2*nr_datapoints_test[seq_len],0])
                        print('Train Start: ',train_start)
                        current_train_set=eu_df.iloc[train_start:train_end-2*nr_datapoints_test[seq_len]]
                        current_validation_set=eu_df.iloc[train_start+nr_datapoints_train[seq_len]:train_end]
                        print('Shape of Train Data: ',current_train_set.shape[0])
                        print('Shape of Validation Data: ',current_validation_set.shape[0]/2)
                        
                        #Perform the rescaling of validation and test sets based on train sets:
                        means=current_train_set.iloc[:,:-2].mean(axis=0).copy()
                        stds=current_train_set.iloc[:,:-2].std(axis=0).copy()

                        rescaled_train_frame=(current_train_set.iloc[:,:-2].copy()-means)/stds
                        rescaled_validation_frame=(current_validation_set.iloc[:,:-2].copy()-means)/stds
                        rescaled_test_frame=(current_test_subset.iloc[:,:-2].copy()-means)/stds

                        rescaled_train_validation_frame=pd.concat([rescaled_train_frame,rescaled_validation_frame],axis=0)

                        #While training AutogluonTS: reserve certain number of timesteps for validation purposes.
                        #not possible to have seperate validation set as with other models,
                        #as eventually ATS model has to be refit on the original data,
                        #as we measure efficiency, we cannot increase the number of time points to retrain on:
                        current_nr_train_points=rescaled_train_frame.shape[0]

                        #Determine number of validation windows based on pred_len size:
                        num_val_windows=int((rescaled_train_validation_frame.shape[0]-current_nr_train_points)/seq_len)
                        print('Number of Validation Windows: ',num_val_windows)
                        
                        #Prepare Train, Validation and Test Data in AutoGluonTS Format:
                        train_validation_data=prepare_autogluon_format(current_frame=rescaled_train_validation_frame)
                        test_data=sequence_splitting(current_frame=rescaled_test_frame,seq_len=seq_len)
                        
                        #Perform NAS with AutoGluonTS:
                        #Create Autogluon predictor:
                        predictor = TimeSeriesPredictor(prediction_length=seq_len, eval_metric="RMSE",target='target',freq='H')
                        #Fit the predictor for a maximum number of time:
                        predictor.fit(
                            train_validation_data,
                            time_limit=int((60*60*24*3)/(15*2*3)),#(60*60*24*4)/15,#a maximum of four days for all 15 subsets (
                            #(keep time similar to NAS using efficient search space as well as to DRAGON NAS)
                            presets='best_quality',#"fast_training",
                            excluded_model_types=["RecursiveTabular","DirectTabular",
                                                "TemporalFusionTransformer","PatchTST",
                                                "DeepAR","Chronos","TiDE"],
                            hyperparameter_tune_kwargs=None,
                            num_val_windows=num_val_windows#walk-forward validation
                        )
                        

                        #Start tracking training time of current train test subset:
                        training_testing_time=0.0
                        
                        start_time=time.time()
                        predictor.refit_full(model='best')
                        refitting_time=time.time()-start_time
                        training_testing_time=training_testing_time+refitting_time

                        #Predict the test data:
                        true_test_outputs=[]
                        prediction_outputs=[]
                        prediction_time=0.0

                        client=Client(n_workers=5)   
                        print(client.dashboard_link)
                        futures = client.map(predict_parallel,test_data,
                                                special_parameters={
                                                        'predictor':predictor})

                        preds_trues_runtimes_values = client.gather(futures)

                        for dask_item in preds_trues_runtimes_values:
                            prediction_outputs.append(dask_item[0])
                            true_test_outputs.append(dask_item[1])
                            prediction_time=prediction_time+dask_item[2]

                        prediction_time=prediction_time/len(test_data)

                        training_testing_time=training_testing_time+prediction_time
                        total_training_testing_time_current_seq_len=total_training_testing_time_current_seq_len+training_testing_time

                        nr_models_ensemble=len(list(predictor._trainer.load_model(predictor.model_best).model_to_weight.keys()))
                        reduced_total_training_testing_time_current_seq_len=reduced_total_training_testing_time_current_seq_len+(training_testing_time/nr_models_ensemble)
                        #act as if we could parallelize the models in the ensemble
                        print('Total Training and Fitting Time: ',training_testing_time)
                        
                        dask_client_cluster_open=True
                        while dask_client_cluster_open==True:
                            try:
                                #client.shutdown()
                                client.close()
                                #cluster.close()
                                dask_client_cluster_open=False
                            except:
                                print('Try again to shutdown and close Dask Client and Cluster!')
                        
                        #Once Dask Client completely shotdown and closed:
                        # also delete client and clister objects from current episode of AC 
                        del client

                        if 'EU_Energy_Time_Series_' in ts_frame:
                            nas_results_dict['NASTrain_Energy_Data'][seq_len]['Results']['Train_Test'+str(test_idx+1)]={'Avg_Predictions':prediction_outputs,
                                                                    'True_Values':true_test_outputs}
                            
                        else:
                            nas_results_dict['TL_Total_Load'][seq_len]['Results']['Train_Test'+str(test_idx+1)]={'Avg_Predictions':prediction_outputs,
                                                                    'True_Values':true_test_outputs}
                            
                    
                    
                    #Save the sum for the entire training(refitting, not NAS) and testing time:
                    if 'EU_Energy_Time_Series_' in ts_frame:
                        nas_results_dict['NASTrain_Energy_Data'][seq_len]['Training_Testing_Time']=total_training_testing_time_current_seq_len
                        nas_results_dict['NASTrain_Energy_Data'][seq_len]['Reduced_Training_Testing_Time']=reduced_total_training_testing_time_current_seq_len
                    else:
                        nas_results_dict['TL_Total_Load'][seq_len]['Training_Testing_Time']=total_training_testing_time_current_seq_len
                        nas_results_dict['TL_Total_Load'][seq_len]['Reduced_Training_Testing_Time']=reduced_total_training_testing_time_current_seq_len
                    print('\n\n')

            with open(str(parent_dir)+'/Results_Evaluation/AutoGluonTS_NAS_Res_RMSE_'+str(nr_run)+'.pkl','wb') as f:
                pickle.dump(nas_results_dict,f)

        #Aggregate results from 10 runs:
        nas_avg_results_dict={'NASTrain_Energy_Data':{48:{'Results':{},
                                                                        'Training_Testing_Time':None},
                                                                    96:{'Results':{},
                                                                        'Training_Testing_Time':None},
                                                                    192:{'Results':{},
                                                                        'Training_Testing_Time':None}},
                                            'TL_Total_Load':{48:{'Results':{},
                                                                'Training_Testing_Time':None},
                                                            96:{'Results':{},
                                                                'Training_Testing_Time':None},
                                                            192:{'Results':{},
                                                                'Training_Testing_Time':None}}} 

        #Load first run:
        with open(str(parent_dir)+'/Results_Evaluation/AutoGluonTS_NAS_Res_RMSE_1.pkl','rb') as f:
                nas_first_run=pickle.load(f)

        #Get dict results from first run:
        for nas_dataset in nas_first_run.keys():
            for forecast_horizon in nas_first_run[nas_dataset].keys():
                #Get training and testing time:
                nas_avg_results_dict[nas_dataset][forecast_horizon]['Training_Testing_Time']=nas_first_run[nas_dataset][forecast_horizon]['Training_Testing_Time']

                #Get results per subset:
                nas_avg_results_dict[nas_dataset][forecast_horizon]['Results']=nas_first_run[nas_dataset][forecast_horizon]['Results']

        for current_run in np.arange(2,11,1):
            #Average results from all remaining runs:
            with open(str(parent_dir)+'/Results_Evaluation/AutoGluonTS_NAS_Res_RMSE_'+str(current_run)+'.pkl','rb') as f:
                    nas_current_run=pickle.load(f)

            for nas_dataset in nas_avg_results_dict.keys():
                for forecast_horizon in nas_avg_results_dict[nas_dataset].keys():
                    #Compute average training and testing time:
                    nas_avg_results_dict[nas_dataset][forecast_horizon]['Training_Testing_Time']=nas_avg_results_dict[nas_dataset][forecast_horizon]['Training_Testing_Time']+nas_current_run[nas_dataset][forecast_horizon]['Training_Testing_Time']
                    if current_run==10:
                        nas_avg_results_dict[nas_dataset][forecast_horizon]['Training_Testing_Time']=nas_avg_results_dict[nas_dataset][forecast_horizon]['Training_Testing_Time']/10.0

                    for test_subset in nas_avg_results_dict[nas_dataset][forecast_horizon]['Results'].keys():
                        #Get list of average predictions:
                        avg_pred_list=nas_avg_results_dict[nas_dataset][forecast_horizon]['Results'][test_subset]['Avg_Predictions']
                        
                        #Get list of current run predictions:
                        current_pred_list=nas_current_run[nas_dataset][forecast_horizon]['Results'][test_subset]['Avg_Predictions']
                        
                        #Loop through each predicted sequence:
                        for pred_seq_idx in range(0,len(current_pred_list)):
                            avg_pred_list[pred_seq_idx]=avg_pred_list[pred_seq_idx]+current_pred_list[pred_seq_idx]

                            #Divide by 10 only if current_run==10:
                            if current_run==10:
                                avg_pred_list[pred_seq_idx]=avg_pred_list[pred_seq_idx]/10.0
                        
                        #Afterward overwrite average predictions in avg nas results dict:
                        nas_avg_results_dict[nas_dataset][forecast_horizon]['Results'][test_subset]['Avg_Predictions']=avg_pred_list

        with open(str(parent_dir)+'/Results_Evaluation/AutoGluonTS_NAS_Res_RMSE_Avg_10_Runs.pkl','wb') as f:
            pickle.dump(nas_avg_results_dict,f)
    
    def run_dragon_nas(self):
        #Create the search space for dragon:
        nr_inputs=19
        candidate_operations_2d = operations_var("2D Candidate operations", size=4, 
        candidates=[identity_var("Unitary"), attention_2d("Attention"), mlp_var("MLP"), conv_2d("Convolution", nr_inputs), 
        pooling_2d('Pooling'), norm_2d("Norm"), dropout('Dropout'),recurrence_2d('RNN')
        ])
        dag_2d = dag_var("2D Dag", candidate_operations_2d)

        candidate_operations_1d = operations_var("2D Candidate operations", size=4,
        candidates=[identity_var("Unitary"), attention_1d("Attention"), mlp_var("MLP"), conv_1d("Convolution", nr_inputs), 
        pooling_1d('Pooling'), norm_1d("Norm"), dropout('Dropout'),recurrence_1d('RNN')
        ])
        dag_1d = dag_var("1D Dag", candidate_operations_1d)
        out = node_var("Out", operation=mlp_const_var('Operation', nr_inputs), activation_function=nn.Identity())

        search_space = ArrayVar(dag_2d, dag_1d, out, label="Search Space", neighbor=ArrayInterval())
        labels = [e.label for e in search_space]

        #Run the dragon search:
        batch_size=32
        nr_inputs=19
        for seq_len in self.nr_timesteps:
            
            #################################################################################################
            def loss_function(args, idx, verbose=False):
                args = dict(zip(labels, args))
                seq_len=48
                model = LoadModel(args, torch.Size([nr_inputs,seq_len,1]))#train_set.__getitem__(0)[0].shape)
                
                model.to(device)
                model = train_model(model, train_loader, verbose=verbose)
                #aggregated_loss,loss, preds, trues = test_model(model, val_loader[0])[0]
                aggregated_loss_frist = test_model(model, val_loader[0])[0]
                aggregated_loss_second = test_model(model, val_loader[1])[0]
                aggregated_loss=np.mean([aggregated_loss_frist,aggregated_loss_second])
                
                print(f"Idx: {idx}, on validation set: RMSE Loss = {aggregated_loss}")#, RMSE = {rmse}")
                
                #return loss, model
                return aggregated_loss, model
            #################################################################################################

            
            dataloader_name='EU_Total_Load_Dataloader_torch_'
            with open(str(parent_dir)+dataloader_name+str(seq_len)+'_Train.pkl','rb') as f:
                current_dataloader=pickle.load(f)

            train_loader=current_dataloader['Train_Test1']['Train_Set_Loader']
            validation_data_1=current_dataloader['Train_Test1']['Test_Set_Loader']
            validation_data_2=current_dataloader['Train_Test2']['Test_Set_Loader']
            val_loader=[validation_data_1,validation_data_2]
            device=torch.device('cuda:0')

            search_algorithm = SteadyStateEA(search_space, n_iterations=10000, 
                                            population_size=100,
                                            selection_size=2, evaluation = loss_function, save_dir="save_dragon"+str(seq_len)+"/test_load_"+str(seq_len))
            min_loss = search_algorithm.run()


        #Then make 10 runs with best performing models:
        for nr_run in np.arange(1,11,1):

            res=[]

            current_dragon_run='dragon_nr_'+str(nr_run)
            nas_results_dict={}
            nas_results_dict['NASTrain_Energy_Data']={48:{'Results':{},
                                                    'Training_Testing_Time':None},
                                                    96:{'Results':{},
                                                    'Training_Testing_Time':None},
                                                    192:{'Results':{},
                                                    'Training_Testing_Time':None}}
            
            nas_results_dict['TL_Total_Load']={48:{'Results':{},
                                                    'Training_Testing_Time':None},
                                                    96:{'Results':{},
                                                    'Training_Testing_Time':None},
                                                    192:{'Results':{},
                                                    'Training_Testing_Time':None}}

            reverse_rescale=False
            
            for dt in self.dataloader_names:
                for nr_timesteps in [48,96,192]:
                    client=Client()
                    print(black('Dask Client Dashboard Link: ',['bold']),client.dashboard_link)
                    dataloader_name=dt+'_'+str(nr_timesteps)+'_'+('Transfer_Learn' if dt=='Energy_Data_Dataloader_torch' else 'Train')+'.pkl'
                    print('Current Dataloader: ',dataloader_name,'\n')

                    nr_inputs=(15 if 'Energy_Data_Dataloader_torch' in dataloader_name else 19)
                    nr_targets=(15 if 'Energy_Data_Dataloader_torch' in dataloader_name else 19)

                    futures = client.map(environment_step,current_dragon_run,
                                        special_parameters={'seq_len':nr_timesteps,
                                                'nr_inputs':nr_inputs,
                                                'nr_targets':nr_targets,
                                                'pred_len':nr_timesteps,
                                                'epochs':50,
                                                'dataloader_name':dataloader_name,
                                                'patience':1,
                                                'use_amp':False,
                                                'checkpoints':'./checkpoints/',
                                                'train_test_split':'Train_Test'+str(1),
                                                'use_gpu':True,
                                                'subset_indices':self.split_indices[nr_timesteps],
                                                'search_space':search_space})

                    current_subset_predictions= client.gather(futures)[0]
                    
                    avg_tr_test_time=current_subset_predictions[2]
                    
                    if 'Energy_Data_Dataloader_torch' in dataloader_name:
                        nas_results_dict['NASTrain_Energy_Data'][nr_timesteps]['Training_Testing_Time']=avg_tr_test_time
                    else:
                        nas_results_dict['TL_Total_Load'][nr_timesteps]['Training_Testing_Time']=avg_tr_test_time

                    for subset_idx in range(0,len(self.split_indices[nr_timesteps])):
                        subset_true_values=current_subset_predictions[1][subset_idx]
                        subset_predictions=current_subset_predictions[0][subset_idx]
                        
                        if 'Energy_Data_Dataloader_torch' in dataloader_name:
                            nas_results_dict['NASTrain_Energy_Data'][nr_timesteps]['Results']['Train_Test'+str(subset_idx+1)]={'Avg_Predictions':subset_predictions,
                                                                    'True_Values':subset_true_values}
                        else:
                            nas_results_dict['TL_Total_Load'][nr_timesteps]['Results']['Train_Test'+str(subset_idx+1)]={'Avg_Predictions':subset_predictions,
                                                                    'True_Values':subset_true_values}


                    #cluster.close()
                    try:
                        client.close()
                    except:
                        print('Opening a new client!')

            res.append(copy.deepcopy(nas_results_dict))

            models_results_dict={'Dragon_NAS':res[0]}

            with open(str(parent_dir)+'/Results_Evaluation/Dragon_NAS_Rewards_Res_'+str(nr_run)+'.pkl','wb') as f:
                pickle.dump(models_results_dict,f)
            
        #Aggregate the results from 10 runs with best dragon models:
        nas_avg_results_dict={'Dragon_NAS':{'NASTrain_Energy_Data':{48:{'Results':{},
                                                                        'Training_Testing_Time':None},
                                                                    96:{'Results':{},
                                                                        'Training_Testing_Time':None},
                                                                    192:{'Results':{},
                                                                        'Training_Testing_Time':None}},
                                            'TL_Total_Load':{48:{'Results':{},
                                                                'Training_Testing_Time':None},
                                                            96:{'Results':{},
                                                                'Training_Testing_Time':None},
                                                            192:{'Results':{},
                                                                'Training_Testing_Time':None}}} }

        #Load first run:
        with open(str(parent_dir)+'/Results_Evaluation/Dragon_NAS_Rewards_Res_1.pkl','rb') as f:
                nas_first_run=pickle.load(f)

        #Get dict results from first run:
        for nas_model in nas_first_run.keys():
            for nas_dataset in nas_first_run[nas_model].keys():
                for forecast_horizon in nas_first_run[nas_model][nas_dataset].keys():
                    #Get training and testing time:
                    nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']=nas_first_run[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']

                    #Get results per subset:
                    nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Results']=nas_first_run[nas_model][nas_dataset][forecast_horizon]['Results']

        for current_run in np.arange(2,11,1):
            #Average results from all remaining runs:
            with open(str(parent_dir)+'/Results_Evaluation/Dragon_NAS_Rewards_Res_'+str(current_run)+'.pkl','rb') as f:
                    nas_current_run=pickle.load(f)

            for nas_model in nas_avg_results_dict.keys():
                for nas_dataset in nas_avg_results_dict[nas_model].keys():
                    for forecast_horizon in nas_avg_results_dict[nas_model][nas_dataset].keys():
                        #Compute average training and testing time:
                        nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']=nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']+nas_current_run[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']
                        if current_run==10:
                            nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']=nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Training_Testing_Time']/10.0

                        for test_subset in nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Results'].keys():
                            #Get list of average predictions:
                            avg_pred_list=nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Results'][test_subset]['Avg_Predictions']
                            
                            #Get list of current run predictions:
                            current_pred_list=nas_current_run[nas_model][nas_dataset][forecast_horizon]['Results'][test_subset]['Avg_Predictions']
                            
                            #Loop through each predicted sequence:
                            for pred_seq_idx in range(0,len(current_pred_list)):
                                avg_pred_list[pred_seq_idx]=avg_pred_list[pred_seq_idx]+current_pred_list[pred_seq_idx]

                                #Divide by 10 only if current_run==10:
                                if current_run==10:
                                    avg_pred_list[pred_seq_idx]=avg_pred_list[pred_seq_idx]/10.0
                            
                            #Afterward overwrite average predictions in avg nas results dict:
                            nas_avg_results_dict[nas_model][nas_dataset][forecast_horizon]['Results'][test_subset]['Avg_Predictions']=avg_pred_list
                            


        with open(str(parent_dir)+'/Results_Evaluation/Dragon_NAS_Rewards_Res_RMSE_Avg_10_Runs.pkl','wb') as f:
            pickle.dump(nas_avg_results_dict,f)

        

#Dragon-related functions:
class LoadModel(nn.Module):
    def __init__(self, args, ts_shape):
        super().__init__()
        self.args = args
        self.input_shape = ts_shape
        seq_len=ts_shape[1]

        # 2-dimensional DAG
        self.cell_2d = args['2D Dag']
        self.cell_2d.set(self.input_shape)

        # Compute after flatten
        #self.flat_shape = (self.cell_2d.output_shape[0],np.prod(self.cell_2d.output_shape[1:]))
        self.flat_shape = (seq_len,np.prod(self.cell_2d.output_shape[1:]))
        
        #print('self.cell_2d.output_shape[0]:',self.cell_2d.output_shape[0])
        #print('self.cell_2d.output_shape[1:]:',self.cell_2d.output_shape[1:])
        #print('self.flat_shape:',self.flat_shape)
        # 1-dimensional DAG
        self.cell_1d = args['1D Dag']
        self.cell_1d.set(self.flat_shape)

        # Output layer
        self.output = args["Out"]
        #print('self.cell_1d.output_shape:',self.cell_1d.output_shape)
        args['Out'].set(self.cell_1d.output_shape)

    def forward(self, X):
        out_2d = self.cell_2d(X)
        #print(out_2d.shape)
        flat = nn.Flatten(start_dim=2)(out_2d)
        out_1d = self.cell_1d(flat)
        #print(out_1d.shape)
        out = self.output(out_1d)
        #print('Shape of output: ',out.shape)
        return out.squeeze(-1)
    
    def set_prediction_to_save(self, name, df):
        self.prediction = {name: df}
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, "best_model.pth")
        torch.save(self.state_dict(), full_path)
        if hasattr(self, "prediction"):
            for k in self.prediction.keys():
                self.prediction[k].to_csv(os.path.join(path, f"best_model_{k}_outputs.csv"))

def MSE(actual, pred):
    return np.mean((actual - pred) ** 2)


def MAPE(actual, pred):
    return np.mean(np.abs((actual - pred) / actual))


def RMSE(actual, pred):
    return np.sqrt(MSE(actual, pred))

def train_model(model, data_loader, verbose=False):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.train()
    for ep in range(20):
        ep_loss = 0
        for X,y in data_loader:
            optimizer.zero_grad()
            #print('Shape of x inputs: ',X.shape)
            X, y = X.float().to(device), y.float().to(device)
            if len(X.shape)!=4:
              X=X.unsqueeze(-1)
            pred = model(X)
            #print('Shape of y outputs: ',y.shape)
            #print('Shape of pred outputs: ',pred.shape)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            ep_loss += loss
        if verbose:
            print(f"Train | Epoch: {ep}, loss = {ep_loss / len(data_loader)}")
    return model

def test_model(model, data_loader):
    loss_fn = nn.MSELoss()
    model.eval()
    test_loss = 0
    preds = []
    trues = []

    rmse_array=[]
    
    with torch.no_grad():
      for X, y in data_loader:
          X, y = X.float().to(device), y.float().to(device)
          
          if len(X.shape)!=4:
              X=X.unsqueeze(-1)
          
          pred = model(X)  

          #Compute dragon losses:
          loss = loss_fn(pred, y).item()
          test_loss += loss
          preds.append(pred.detach().cpu().numpy())
          trues.append(y.detach().cpu().numpy())

          #Compute aggregated score:
          trues_values=y.cpu().numpy()
          predicted_values=pred.cpu().numpy()

          for batch_idx in range(0,len(trues_values)):        
            for country_idx in range(0,trues_values[batch_idx,:,:].shape[-1]):
                true_country=trues_values[batch_idx,:,country_idx]
                predicted_country=predicted_values[batch_idx,:,country_idx]
                results=RegressionMetric(y_true=true_country,y_pred=predicted_country).get_metrics_by_list_names(["RMSE"])
                rmse_array.append(results['RMSE'])
                
    loss = test_loss / len(data_loader)
    aggregated_loss=np.mean(np.array(rmse_array))

    return aggregated_loss,loss, preds, trues



#reset the trainable parameters to random:
def reset_random(model: nn.Module) -> None:
    """
    Set all trainable parameters in the model to random values.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.copy_(torch.randn_like(param))

def environment_step(nr_dragon_run,**kwargs):
    seq_len=kwargs['special_parameters']['seq_len']
    nr_inputs=kwargs['special_parameters']['nr_inputs']
    nr_targets=kwargs['special_parameters']['nr_targets']
    pred_len=kwargs['special_parameters']['pred_len']
    #batch_size=kwargs['special_parameters']['batch_size']
    epochs=kwargs['special_parameters']['epochs']
    #patience=kwargs['special_parameters']['patience']
    
    use_amp=kwargs['special_parameters']['use_amp']
    #walk_forward_validation=kwargs['special_parameters']['walk_forward_validation']
    checkpoints=kwargs['special_parameters']['checkpoints']
    use_gpu=kwargs['special_parameters']['use_gpu']
    subset_indices=kwargs['special_parameters']['subset_indices']
    search_space=kwargs['special_parameters']['search_space']
    

    best_model = load_archi("save_dragon"+str(seq_len)+"/test_load_"+str(seq_len)+"/best_model/x.pkl")
    labels = [e.label for e in search_space]
    best_model = dict(zip(labels, best_model))
    NAS_model = LoadModel(best_model, torch.Size([nr_inputs,seq_len,1]))    
    NAS_model.cell_1d.set((seq_len,np.prod(NAS_model.cell_2d.output_shape[1:])))
    NAS_model.output=Node(combiner="add", operation=MLP, hp={"out_channels": nr_inputs}, activation=nn.Identity())#out
    #Overwrite input shape of output module:
    NAS_model.output.set(NAS_model.cell_1d.output_shape)
    #Reset all trainable variables of the Dragon NAS model:
    NAS_model.apply(reset_random)
    
    #reverse_rescale=kwargs['special_parameters']['reverse_rescale']
    
    with open(kwargs['special_parameters']['dataloader_name'],'rb') as f:
        current_dataloader=pickle.load(f)
        train_data=current_dataloader['Train_Test'+str(subset_indices[0])]['Train_Set_Loader']
        validation_data=current_dataloader['Train_Test'+str(subset_indices[0])]['Test_Set_Loader']
        rescaler=current_dataloader['Train_Test'+str(subset_indices[0])]['Scaler']
    
    criterion=nn.MSELoss()
    
    
    optimizer_NAS_model = torch.optim.Adam(NAS_model.parameters(), lr=0.001)

    device = _acquire_device(use_gpu=use_gpu)
    NAS_model.to(device)
    
    subset_predictions=[]
    true_values=[]
    
    #Start counting time:
    start_training_time = time.time()

    #1. Train the model: ####################################################################################
    NAS_model=train_NAS_model(NAS_model=NAS_model,optimizer_NAS_model=optimizer_NAS_model,checkpoints=checkpoints,
                              device=device,patience=1,epochs=epochs,
                              train_data=train_data,criterion=criterion,use_amp=use_amp)
    #Done with training the model: ##############################################################################
    
    
    #2. Validate the model: ############################################################################################
    #Compute the reward:
    NAS_outputs_first_subset=validate_NAS_model(NAS_model=NAS_model,validation_data=validation_data,device=device,
                       rescaler=rescaler)
    subset_predictions.append(NAS_outputs_first_subset[1])
    true_values.append(NAS_outputs_first_subset[0])
    #Done with testing the model: ####################################################################################         

    for subset_idx in range(1,len(subset_indices)):
        train_data=current_dataloader['Train_Test'+str(subset_indices[subset_idx])]['Train_Set_Loader']
        validation_data=current_dataloader['Train_Test'+str(subset_indices[subset_idx])]['Test_Set_Loader']
        
        NAS_model=train_NAS_model(NAS_model=NAS_model,optimizer_NAS_model=optimizer_NAS_model,checkpoints=checkpoints,
                              device=device,patience=1,epochs=epochs,
                              train_data=train_data,criterion=criterion,use_amp=use_amp)
        NAS_outputs_current_subset=validate_NAS_model(NAS_model=NAS_model,validation_data=validation_data,device=device,
                       rescaler=rescaler)
        
        subset_predictions.append(NAS_outputs_current_subset[1])
        true_values.append(NAS_outputs_current_subset[0])
    #end coutning time:
    end_training_time = time.time()

    return [subset_predictions,true_values,end_training_time-start_training_time]

def train_NAS_model(NAS_model,optimizer_NAS_model,checkpoints,
                   device,patience,epochs,
                   train_data,criterion,use_amp,validation_data=None):
    
    #1. Train the model: ####################################################################################
    
    iter_count = 0
    
    path = os.path.join(checkpoints, 'TS_Benchmark_Testing')
    #if not os.path.exists(path):
    #    os.makedirs(path)

    early_stopping = EarlyStopping(patience=patience,delta=0.01,verbose=False)

    #scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer_NAS_model,
    #                                        steps_per_epoch = len(train_data),
    #                                        pct_start = 0.3,
    #                                        epochs = epochs,
    #                                        max_lr = 0.001)

    val_error_array=[]
    for epoch in range(0,epochs):
        train_loss=[]
        
        NAS_model.train()

        for i, train_items in enumerate(train_data):    
            #Get train item: #############################################################################
            batch_x, batch_y,_,_,_=train_items
            batch_x=batch_x.unsqueeze(-1)
            iter_count += 1
            optimizer_NAS_model.zero_grad()
            #Make sure both the data and the model are on the same device: (otherwise potentially a runtime error)
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = NAS_model(batch_x)
            else:
                outputs = NAS_model(batch_x)
                
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            if use_amp:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer_NAS_model)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer_NAS_model.step()
            #End backward Pass: ################################################################################

        #val_error=validate_NAS_model(NAS_model=NAS_model,validation_data=validation_data,device=device,
        #               mse_array=[],smape_array=[],reverse_rescale=False)
        #val_error_array.append(val_error)

        #Apply Early Stopping Callback: ########################################################################
        early_stopping(np.mean(train_loss), NAS_model, path)
        if early_stopping.early_stop:
            #print("Early stopping")
            break
        #End Train Items: #############################################################################
        
        #Potentially adjust the learning rate: ########################################################
        #adjust_learning_rate(optimizer=optimizer_NAS_model, scheduler=scheduler, epoch=epoch + 1,lradj="type3",
        #learning_rate=0.001, printout=False)
        #End adjusting the learning rate: #############################################################
        
    #Done with training the model: ##############################################################################
    return NAS_model#,np.array(val_error_array)

def validate_NAS_model(NAS_model,validation_data,
                       device,
                       mse_array=None,smape_array=None,reverse_rescale=None,rescaler=None,
                       validation_loss=None,criterion=None):#,
                       #model_names_array=None,subset_names_array=None,batch_sample_indices=None,
                       #country_names=None,EU_countries_dict=None):
    
    NAS_model.eval()
    true_values=[]
    predictions=[]
    with torch.no_grad():
        for i, validation_items in enumerate(validation_data):
            batch_x, batch_y,_,_,_=validation_items
            batch_x=batch_x.unsqueeze(-1)
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            outputs=NAS_model(batch_x)
            
            #Reverse rescale the predictions: ##################################################################
            true_batch_values=batch_y.detach().cpu().numpy()
            batch_predictions=outputs.detach().cpu().numpy()
            for batch_idx in range(0,len(true_batch_values)):
                
                if mse_array!=None and smape_array!=None:
                    for country_idx in range(0,true_batch_values[batch_idx,:,:].shape[-1]):
                        true_country=true_batch_values[batch_idx,:,country_idx]
                        predicted_country=batch_predictions[batch_idx,:,country_idx]
                        results=RegressionMetric(y_true=true_country,y_pred=predicted_country).get_metrics_by_list_names(["RMSE","SMAPE"])
                        mse_array.append(results['RMSE'])
                        smape_array.append(results['SMAPE'])
                else:
            
                    true_values.append(true_batch_values[batch_idx,:,:])
                    predictions.append(batch_predictions[batch_idx,:,:])  
    if mse_array!=None and smape_array!=None:
        return np.mean((np.array(mse_array)+np.array(smape_array))/2)
    else:
        return np.array(true_values),np.array(predictions) 




#Autogluon-NAS related functions:
def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

def prepare_autogluon_format(current_frame):
    cols=current_frame.columns
    autogluon_ids=[]
    autogluon_ts=[]
    autogluon_timestamp=[]

    for col in cols:
        current_ts=list(current_frame[col].values)
        autogluon_ts.extend(current_ts)
        autogluon_ids.extend([col]*len(current_ts))
        autogluon_timestamp.extend(list(current_frame.index))

    autogluon_format_df=pd.DataFrame({'timestamp':np.array(autogluon_timestamp),
                                    'item_id':np.array(autogluon_ids),
                                    'target':np.array(autogluon_ts)})


    autogluon_timeseries_data= TimeSeriesDataFrame.from_data_frame(
        autogluon_format_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )
    return autogluon_timeseries_data

def sequence_splitting(current_frame,seq_len):
    rescaled_test_sequences=[]

    for timestamp_idx in range(0,current_frame.shape[0]-seq_len*2):
        s_begin = timestamp_idx
        s_end = s_begin + seq_len
        #used for building a sequence!
        seq_x=current_frame.iloc[s_begin:s_end]
        #Prepare inputs for AutoGluonTS format:
        seq_x_autogluonts=prepare_autogluon_format(current_frame=seq_x)
        
        #used for building targets!
        r_begin = s_end
        r_end = r_begin + seq_len
        seq_y=current_frame.iloc[r_begin:r_end,:]
        #If rescale:
        #rescale based on input sequence until start token:
        rescaled_test_sequences.append([seq_x,seq_x_autogluonts, seq_y])
    return rescaled_test_sequences

def predict_parallel(seq_pair,**kwargs):
    start_time=time.time()
    predictor=kwargs['special_parameters']['predictor']

    seq_x_autogluonts=seq_pair[1]

    #Create new autogluon ts data:
    predictor.make_future_data_frame(seq_x_autogluonts)

    #Generate predictions on this new data:
    predicted_y_autogluon_ts=predictor.predict(model=predictor.model_best,data=seq_x_autogluonts)
    
    #convert predicted_y_autogluon_ts back to sequence format for evaluation purposes:
    col_ts_index=np.array(predicted_y_autogluon_ts.index.tolist())
    predicted_y_autogluon_ts['ID']=col_ts_index[:,0]
    predicted_y=[]
    
    for col in seq_pair[-1].columns:#follows the order of the time series in the true y sequence:
        predicted_y.append(predicted_y_autogluon_ts[predicted_y_autogluon_ts['ID']==col]['mean'].values)
    predicted_y=np.array(predicted_y).T
    end_time=time.time()
    return [predicted_y,seq_pair[-1].values,end_time-start_time]#seq_pair[-1].values=true values
    

