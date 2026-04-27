from pmdarima import auto_arima
import timesfm
from chronos import ChronosPipeline
import pandas as pd
import torch
import numpy as np
import time
from dask.distributed import Client
#from dask_cuda import LocalCUDACluster

import time
from transformers import AutoModelForCausalLM
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
from huggingface_hub import snapshot_download
from os import path

import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math



class Univariate_Forecasting():
    """
    A class implementing the univariate forecasting techniques Chronos, TimesFM and ARIMA:

    Parameters:
    ----------
    df : Pandas dataframe
        The dataframe containing the (electricity price-related) time series
    train_size : int
        The number of training time steps.
    test_size : list of int values
        The number of testing time steps.
    seq_len : int
        The size of the input sequence, i.e., the number of input time steps.
    pred_len : int
        The size of the output sequence, i.e., the number of time steps to be predicted.
    n_splits : int
        The number of walk-forward validation splits.
    univariate_forecasters:list
        The univariate forecasting techniques to compute predictions with.
    
    Attributes:
    (the same as the parameters)
    ----------

    Methods
    -------
    compute_predictions(current_univariate_forecaster):
        Generates the predictions using the supported univariate models.
    """

    def __init__(self,
                 dataset_dict:dict,
                 #dataloader:dict,
                 device:torch.device,
                 pretrained_model:str,
                 args:None,
                 num_samples:None):
        
        self.dataset_dict=dataset_dict
        #self.dataloader=dataloader
        self.device=device
        self.pretrained_model=pretrained_model
        self.args=args
        self.num_samples=num_samples
            

        super(Univariate_Forecasting, self).__init__()

    def compute_predictions(self,data_idx,**kwargs)->list:
        """Computes predictions from univariate time series.
        
        Parameters:
        ----------
        current_univariate_forecaster: str
        The name of the univariate forecasting technique.
        
        Returns:
        ----------
        dict_results: dict
        The dictionary with the generated predictions.

        Raises:
        ----------
        ValueError: If specified current_univariate_forecaster is not in the list [ARIMA, Chronos, TimesFM].
        """
        #Load the models:
        if self.pretrained_model=="GPT4TS":
            self.args.seq_len=self.dataset_dict[data_idx]['nr_timesteps']
            self.args.pred_len=self.dataset_dict[data_idx]['nr_timesteps']
            gpt4ts_model = GPT4TS(self.args, self.device)
            gpt4ts_model.to(self.device)
            for pr in gpt4ts_model.parameters():
                pr.requires_grad=False

        elif self.pretrained_model=="Timer":
            # Timer Module:
            timer_model = AutoModelForCausalLM.from_pretrained('thuml/timer-base-84m', trust_remote_code=True)
            timer_model.to(self.device)

        elif self.pretrained_model=="Sundial":
            #Sundial Module:
            sundial_model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True) 
            sundial_model.to(self.device)

        elif self.pretrained_model=='TimeMoe':
            timemoe_model = AutoModelForCausalLM.from_pretrained("Maple728/TimeMoE-200M",
                device_map=self.device,  trust_remote_code=True)

        else:
            #TimesFM Module:
            freq = torch.tensor([0], dtype=torch.long).to(self.device)

            if self.dataset_dict[data_idx]['nr_timesteps']==12:
                timesfm_context_len=8
            elif self.dataset_dict[data_idx]['nr_timesteps']==24:
                timesfm_context_len=16
            elif self.dataset_dict[data_idx]['nr_timesteps']==48:
                timesfm_context_len=32
            
            elif self.dataset_dict[data_idx]['nr_timesteps'] in [96,192]:
                timesfm_context_len=self.dataset_dict[data_idx]['nr_timesteps']
            
            else:
                if self.dataset_dict[data_idx]['nr_timesteps']==336:
                    timesfm_context_len=320
                else:
                    timesfm_context_len=512
            timesfm_model = timesfm.TimesFm(hparams=timesfm.TimesFmHparams(context_len=timesfm_context_len,
                                horizon_len=self.dataset_dict[data_idx]['nr_timesteps'],
                                per_core_batch_size=32,  
                                input_patch_len=32,output_patch_len=128,num_layers=20,model_dims=1280,backend=self.device),
                                checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="google/timesfm-1.0-200m-pytorch"))

            timesfm_model= PatchedTimeSeriesDecoder(timesfm_model._model_config)
            checkpoint_path = path.join(snapshot_download("google/timesfm-1.0-200m-pytorch"), "torch_model.ckpt")
            loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
            timesfm_model.load_state_dict(loaded_checkpoint)
            timesfm_model.to(self.device)
            for pr in timesfm_model.parameters():
                pr.requires_grad=False

        """
        nr_timestep_inputs=np.min([self.dataset_dict[data_idx]['nr_timesteps'],512])
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=(nr_timestep_inputs if nr_timestep_inputs!=336 else 320), 
                horizon_len=self.dataset_dict[data_idx]['nr_timesteps'],  
                input_patch_len=32,#self.dataset_dict[data_idx]['input_patch_len'],
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=self.device),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"))
        """

        current_df=self.dataset_dict[data_idx]['df']
        nr_datapoints_train=self.dataset_dict[data_idx]['nr_datapoints_train']
        nr_datapoints_test=self.dataset_dict[data_idx]['nr_datapoints_test']
        batch_size=kwargs['special_parameters']['batch_size']

        #timeseries_preprocessor=Multivaraite_TimeSeries_Preprocesser(df=current_df,
        #                                                            train_size=nr_datapoints_train,
        #                                                            test_size=nr_datapoints_test,
        #                                                            seq_len=self.dataset_dict[data_idx]['nr_timesteps'],
        #                                                            pred_len=self.dataset_dict[data_idx]['nr_timesteps'],
        #                                                            batch_size=batch_size)

        #Create a dictionary with all dataloaders per model name:
        dataloaders_dict=self.dataset_dict[data_idx]['current_dataloader']#timeseries_preprocessor.preprocess_data(current_multivariate_forecasters=['TSMixer'])['TSMixer']#will do the same as for TimesFM

        start_time = time.time()
        pretrained_results_dict={}

        subset_indices=self.dataset_dict[data_idx]['split_indices']
        #print(data_idx)
        #print(len(subset_indices))

        for split_idx in range(0,len(subset_indices)):
            current_subset='Train_Test'+str(subset_indices[split_idx])
            #for current_subset in dataloaders_dict.keys():
            
            current_loaders=dataloaders_dict[current_subset]
            current_test_loader=current_loaders['Test_Set_Loader']
            
            true_sequences_outputs=[]
            pretrained_squences_predictions=[]

            
            for i, test_items in enumerate(current_test_loader):    
                #Get train item: #############################################################################
                batch_x, batch_y,batch_x_mark,batch_y_mark,index=test_items
                batch_y = batch_y[:, -self.dataset_dict[data_idx]['nr_timesteps']:,:]
                
                #generate the forecast with each model:
                batch_x=batch_x.to(self.device)
                batch_x_mark=batch_x_mark.to(self.device)
                
                if self.pretrained_model=="GPT4TS":
                    batch_x_predictions=torch.mean(torch.stack([gpt4ts_model(batch_x,batch_x_mark) for nr_sample in range(0,self.num_samples)],axis=0),axis=0)
                elif self.pretrained_model=="Timer":
                    batch_x_predictions =torch.stack([timer_model.generate(batch_x[:,:,var], max_new_tokens=batch_x.shape[1]) for var in range(0,batch_x.shape[-1])],axis=-1)
                
                elif self.pretrained_model=="TimeMoe":
                    batch_x_predictions =torch.stack([timemoe_model.generate(batch_x[:,:,var], max_new_tokens=batch_x.shape[1])[:,-batch_x.shape[1]:] for var in range(0,batch_x.shape[-1])],axis=-1)
                
                elif self.pretrained_model=="Sundial":
                    batch_x_predictions=torch.stack([torch.mean(sundial_model.generate(batch_x[:,:,var], max_new_tokens=batch_x.shape[1], num_samples=self.num_samples),dim=1) for var in range(0,batch_x.shape[-1])],axis=-1)
                else:
                    padding_second_dimension=timesfm_context_len+batch_x.shape[1]
                    input_padding = torch.zeros((batch_x.shape[0],padding_second_dimension)).to(self.device)
                    batch_x_predictions=torch.stack([timesfm_model.decode(input_ts=batch_x[:,-timesfm_context_len:,var],
                            paddings=input_padding,freq=freq,horizon_len=batch_x.shape[1])[0] for var in range(0,batch_x.shape[-1])],axis=-1)

                for batch_idx in range(0,batch_x.shape[0]):
                    test_output_seq=batch_y[batch_idx,:,:].detach().cpu().numpy()
                    true_sequences_outputs.append(test_output_seq)
                    
                    sequence_predictions=batch_x_predictions[batch_idx,:,:].detach().cpu().numpy()
                    pretrained_squences_predictions.append(sequence_predictions)
        
                
                """
                for batch_idx in range(0,batch_x.shape[0]):
                    test_input_seq=batch_x[batch_idx,:,:]#.detach().cpu().numpy()
                    
                    if test_input_seq.shape[0] in [96,192]:
                        forecast_input=[test_input_seq[:,var] for var in range(0,test_input_seq.shape[1])]
                    else:
                        if test_input_seq.shape[0]==336:
                            forecast_input=[test_input_seq[-320:,var] for var in range(0,test_input_seq.shape[1])]
                        else:
                            forecast_input=[test_input_seq[-512:,var] for var in range(0,test_input_seq.shape[1])]
                    
                    frequency_input=[0]*len(forecast_input)

                    test_output_seq=batch_y[batch_idx,:,:].detach().cpu().numpy()
                    true_sequences_outputs.append(test_output_seq)

                    global_point_forecast = tfm.forecast(
                    forecast_input,
                    freq=frequency_input)[0].T
                    timesfm_squences_predictions.append(global_point_forecast)
                """

            true_sequences_outputs=np.array(true_sequences_outputs)
            pretrained_squences_predictions=np.array(pretrained_squences_predictions)
            
            pretrained_results_dict['Train_Test'+str(split_idx+1)]=[true_sequences_outputs,pretrained_squences_predictions]

            #pretrained_results_dict[current_subset]=[true_sequences_outputs,pretrained_squences_predictions]

        end_time = time.time()

        return [data_idx,{'Results':pretrained_results_dict,'Training_Testing_Time':end_time-start_time}]

            

    def execute_parallel_univariate_forecasting(self,batch_size)->dict:
        
        #cluster = LocalCUDACluster()
        client=Client()#cluster)

        print(client.dashboard_link)
        futures = client.map(self.compute_predictions,
                            list(self.dataset_dict.keys()),
                            special_parameters={'batch_size':batch_size})
                
        univariate_results = client.gather(futures)
        try:
            #cluster.close()
            client.close()
        except:
            print('Opening a new Client for the next Experiment!')
            

        return univariate_results

#####
#GPT4TS-related functions:


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print(x)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
        
        
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs