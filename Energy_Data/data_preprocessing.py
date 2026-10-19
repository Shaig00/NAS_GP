import sys
sys.path.append("..")

import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import torch
import warnings
from functools import wraps



def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

class Data_Preprocessing():
    """
    A class facilitating the import and the preprocessing of energy time series. The latter are split into sequences, rescaled and 
    saved in pytorch dataloaders in a format suitable for all 13 forecasting techniques implemented in the NAS study.

    Parameters:
    ----------
    
    Attributes:
    ----------
    nr_timesteps : list
        A list containing the prediction horizon settings included in the empirical research.
    nr_datapoints : list
        A list containing the number of all timesteps for the 17 train-test/validation subsets corresponding to each forecasting setting.
    nr_splits : list
        The corresponding number of splits calculated in such a way to avoid an overlap with the first two subsets of the 
        longest forecasting setting, i.e., 192 hours. The need to avoid overlap results from testing NAS on the first two train-validation subset pairs for all
        forecasting settings.
    nr_datapoints_train : list
        The number of train timesteps per forecasting setting.
    nr_datapoints_test : list
        The number of test timesteps per forecasting setting.
    eu_energy_frame : pd.DataFrame
        The dataframe object, in which the NAS transfer learning time series are stored after importing the data.
    eu_total_load_frame : pd.DataFrame
        The dataframe object, in which the NAS train time series are stored after importing the data.
    

    Methods
    -------
    set_transfer_learn_data(transfer_learn_frame):
        Save the merged time series into the self.eu_energy_frame dataframe object.
    set_train_data(train_frame):
        Save the merged time series into the self.eu_total_load_frame dataframe object.
    merge_transfer_learn_data():
        Import and merge the EU energy time series containing, among others, also renewable data types.
    merge_train_data():
        Import and merge the EU energy time series containing the total supplier generation.
    create_timestep_frames():
        Create and saves the pandas dataframe with all train-validation/test subsets per forecasting setting.
    get_timestamp_data(df):
        Extracts time covariates from the original time series dates.
    get_TrainTest_splits(data,train_size,test_set_names):
        Generates all possible train and test subsets based on the test subset names provided
        as an argument to the function. 
    create_dataloader(current_subset,current_timestamps_subset,seq_len,pred_len,batch_size):
        Creates a pytorch dataloader for a single dataset after splitting it into input and output sequences, among others.
    create_timesteps_dataloaders():
        For all forecasting settings, this function splits the original time series into multiple chunks of train and test subsets suitable for walk-forward
        validation, rescales the subsets, stores them in a dictionary and saves them in the EU_Time_Series folder. To facilitate that, the function
        calls the functions get_TrainTest_splits and create_dataloader.

    """

    def __init__(self):
        self.nr_timesteps=[48,96,192]
        self.nr_datapoints=[9984,11500,13956]
        self.nr_splits=[20,19,17]

        self.nr_datapoints_train=[1904,2000,2192]
        self.nr_datapoints_test=[404,500,692]

        #Placeholder for energy (transfer learn) and total load (train) time series datasets: 
        self.eu_energy_frame=None
        self.eu_total_load_frame=None
        self.eu_energy_prices_frame=None

        self.train_data_frame=None
        self.transfer_learn_data_frame=None
        self.exg_data_frame=None

    def set_transfer_learn_data(self,transfer_learn_frame):
        self.eu_energy_frame=transfer_learn_frame
        self.transfer_learn_data_frame=transfer_learn_frame

    # def set_train_data(self,train_frame):
    #     self.eu_total_load_frame=train_frame
    #     self.train_data_frame=train_frame

    def set_exg_data(self, exg_frame):
        self.eu_total_load_frame = exg_frame
        self.exg_data_frame = exg_frame

    def set_train_data_energy_prices(self, train_frame):
        self.eu_energy_prices_frame=train_frame
        self.train_data_frame=train_frame

    def merge_transfer_learn_data(self):
        #Set the countries to include:
        EU_countries=['Germany',
                'Spain','Romania']

        #Specify the time series types to include per country:
        energy_types_keep=['MTU',
                        'Biomass - Actual Aggregated [MW]',
                        'Fossil Gas - Actual Aggregated [MW]',
                        'Hydro Run-of-river and poundage - Actual Aggregated [MW]',
                        'Hydro Water Reservoir - Actual Aggregated [MW]',
                        'Wind Onshore - Actual Aggregated [MW]']

        #Create a pandas dataframe from all time series:
        eu_dfs=[]
        for current_country in EU_countries:
            
            #Load different enegry time series:
            current_country_df_2023=pd.read_csv('../Energy_Data/EU_Time_Series/'+current_country+'_Generation_per_Production_Type_2023.csv',sep=',')
            current_country_df_2023.columns=[col.replace('  - ',' - ') for col in current_country_df_2023.columns]
            current_country_df_2024=pd.read_csv('../Energy_Data/EU_Time_Series/'+current_country+'_Generation_per_Production_Type_2024.csv',sep=',')
            current_country_df_2025=pd.read_csv('../Energy_Data/EU_Time_Series/'+current_country+'_Generation_per_Production_Type_2025.csv',sep=',')
            
            current_country_df=pd.concat([current_country_df_2023,
                                        current_country_df_2024,
                                        current_country_df_2025],axis=0)
            
            
            #Select a set of time series for the current country:
            current_country_df=current_country_df[energy_types_keep]
            current_dates=current_country_df[["MTU"]]
            current_country_df=current_country_df.drop(["MTU"],axis=1)
            
            #Rename columns:
            renamed_columns=[current_country+" "+col.split(" - ")[0]+" [MW]" for col in current_country_df.columns]
            current_country_df.columns=renamed_columns
            #Change format of dates:
            current_dates=current_dates["MTU"].str.replace(" (UTC)","",regex=False)
            current_dates=[datetime.strptime(date.split(" - ")[-1],'%d.%m.%Y %H:%M') for date in current_dates.values]
            current_country_df.index=np.array(current_dates)
            eu_dfs.append(current_country_df)

        #Concat Eu dataframes as well as drop  missing values: 
        eu_energy_frame=pd.concat(eu_dfs,axis=1).dropna()

        #Convert columns to dtype float32: 
        for col in eu_energy_frame.columns:
            eu_energy_frame[col]=eu_energy_frame[col].astype(np.float32)
        eu_energy_frame=eu_energy_frame.resample('h').sum()
        
        #Set the start of the transfer learn dataset:
        # datetime_index=pd.DatetimeIndex(['2023-10-25 00:00:00'])
        datetime_index=pd.DatetimeIndex(['2024-01-10 00:00:00'])
        eu_energy_frame=eu_energy_frame[(eu_energy_frame.index>=datetime_index[0])]
        self.set_transfer_learn_data(transfer_learn_frame=eu_energy_frame)


    def merge_exg_data(self):
 
        EU_countries=['Austria','Belgium','Bulgaria',
              'Croatia','Czech_Republic', 'Estonia',
              'Finland','France', 'Germany',
              'Greece','Hungary','Latvia','Lithuania',
              'Netherlands', 'Poland','Portugal',
              'Romania', 'Spain']
        #NOTE: Slovakia is excluded. Its Day-ahead Total Load Forecast series ends
        #on 2024-06-30 while its actual load continues, and keeping it truncates
        #the intersection to 4151 hours. Only its load covariates are missing;
        #its price channel is unaffected.
 
        energy_types_keep=["MTU","Day-ahead Total Load Forecast"]
 
        total_load_eu_dfs=[]
        for current_country in EU_countries:
            #Load electricity total load time series:
            total_load_df=pd.concat([pd.read_csv('../Energy_Data/EU_Time_Series/'+current_country+'_Total_Load_2023.csv',sep=','),
                                    pd.read_csv('../Energy_Data/EU_Time_Series/'+current_country+'_Total_Load_2024.csv',sep=','),
                                    pd.read_csv('../Energy_Data/EU_Time_Series/'+current_country+'_Total_Load_2025.csv',sep=',')],axis=0)
 
            #Format columns and
            #Select a set of time series for the current country:
            total_load_df.columns=['MTU','Day-ahead Total Load Forecast','Actual Total Load']
            total_load_df=total_load_df[energy_types_keep]
            #NOTE: the actual load series is dropped on purpose. It is not known at
            #forecast time, so using it as a covariate would be look-ahead. The
            #day-ahead forecast is published the previous day and is available.
 
            current_dates=total_load_df[["MTU"]]
            total_load_df=total_load_df.drop(["MTU"],axis=1)
 
            #Rename columns:
            total_load_df.columns=[current_country+' Day-ahead Total Load [MW]']
            total_load_df=total_load_df.apply(pd.to_numeric,errors='coerce')
 
            #Change format of dates:
            current_dates=current_dates["MTU"].str.replace(" (UTC)","",regex=False)
            current_dates=[datetime.strptime(date.split(" - ")[-1],'%d.%m.%Y %H:%M') for date in current_dates.values]
            total_load_df.index=np.array(current_dates)
 
            #Resample to hourly BEFORE the concat: ENTSO-E publishes load at 15, 30
            #or 60 minute MTU depending on the country and the year, so concatenating
            #raw indices leaves mostly NaN. Mean, not sum, because load is
            #instantaneous MW rather than a flow.
            total_load_df=total_load_df.resample('h').mean()
 
            total_load_eu_dfs.append(total_load_df)
 
        #Concat Eu dataframes as well as drop missing values:
        total_load_eu_frame=pd.concat(total_load_eu_dfs,axis=1).dropna()
 
        #Convert columns to dtype float32:
        for col in total_load_eu_frame.columns:
            total_load_eu_frame[col]=total_load_eu_frame[col].astype(np.float32)
 
        #Set the start of the train dataset:
        datetime_indices=pd.DatetimeIndex(['2024-01-10 00:00:00'])
        total_load_eu_frame=total_load_eu_frame[(total_load_eu_frame.index >= datetime_indices[0])]
 
        #Restore a gapless hourly index. The dropna above removes any hour missing
        #for even one country, which leaves holes around the DST changeover. Both
        #this frame and the price frame are later sliced by position, so a hole here
        #shifts every subsequent row relative to the prices and the two stop
        #referring to the same hours.
        full_index=pd.date_range(start=total_load_eu_frame.index.min(),
                                 end=total_load_eu_frame.index.max(),freq='h')
        n_missing=len(full_index)-total_load_eu_frame.shape[0]

        total_load_eu_frame=total_load_eu_frame.reindex(full_index)
        #Load is strongly weekly-periodic, so for a multi-hour outage the same hour
        #one week earlier is a far better estimate than a straight line, which would
        #flatten the diurnal cycle -- the very signal these covariates carry. The
        #thirteen hours missing on 2024-10-27 are the DST changeover, absent for all
        #eighteen countries at once.
        total_load_eu_frame=total_load_eu_frame.fillna(total_load_eu_frame.shift(168))
        total_load_eu_frame=total_load_eu_frame.fillna(total_load_eu_frame.shift(-168))
        #Short residual gaps only:
        total_load_eu_frame=total_load_eu_frame.interpolate(limit=3,limit_direction='both')
        total_load_eu_frame=total_load_eu_frame.astype(np.float32)

 
        print('Exogenous frame: ',total_load_eu_frame.shape[0],' hours x ',
              total_load_eu_frame.shape[1],' countries (',
              total_load_eu_frame.index.min(),' -> ',total_load_eu_frame.index.max(),')')
        print('Hours filled by interpolation: ',n_missing)
        if total_load_eu_frame.isna().any().any():
            print('WARNING: gaps remain after interpolation: ',
                  int(total_load_eu_frame.isna().sum().sum()))
 
        self.set_exg_data(exg_frame=total_load_eu_frame)


    def merge_train_data_electricity_prices(self):

        EU_countries=['AUSTRIA','BELGIUM','BULGARIA',
              'CROATIA','CZECH_REPUBLIC', 'ESTONIA',
              'FINLAND','FRANCE', 'GERMANY',
              'GREECE','HUNGARY','LATVIA','LITHUANIA',
              'NETHERLANDS', 'POLAND','PORTUGAL',
              'ROMANIA', 'SLOVAKIA', 'SPAIN']

        energy_types_keep=["MTU (UTC)","Sequence","Day-ahead Price (EUR/MWh)"]

        energy_prices_eu_dfs=[]
        for current_country in EU_countries:
            #Load electricity total load time series:
            energy_prices_df=pd.concat([pd.read_csv('../Energy_Data/EU_Energy_Prices/'+current_country+'_ENERGY_PRICES_2023.csv',sep=','),
                                    pd.read_csv('../Energy_Data/EU_Energy_Prices/'+current_country+'_ENERGY_PRICES_2024.csv',sep=','),
                                    pd.read_csv('../Energy_Data/EU_Energy_Prices/'+current_country+'_ENERGY_PRICES_2025.csv',sep=',')],axis=0)

            #Format columns and
            #Select a set of time series for the current country:
            energy_prices_df=energy_prices_df[energy_types_keep]
            energy_prices_df.columns=['MTU', 'Sequence', current_country[0]+str.lower(current_country[1:])+' Day-ahead Price (EUR/MWh)']
            
            energy_prices_df=energy_prices_df.loc[(energy_prices_df['Sequence'] == 'Sequence 1') | 
                                                  (energy_prices_df['Sequence'] == 'Without Sequence')]
            energy_prices_df=energy_prices_df.drop(["Sequence"],axis=1)
            
            current_dates=energy_prices_df[["MTU"]]
            energy_prices_df=energy_prices_df.drop(["MTU"],axis=1)
            
            #Change format of dates:
            current_dates=current_dates["MTU"].str.replace(" (UTC)","",regex=False)
            current_dates=[datetime.strptime(date.split(" - ")[-1],'%d/%m/%Y %H:%M:%S') for date in current_dates.values]
            energy_prices_df.index=np.array(current_dates)

            energy_prices_eu_dfs.append(energy_prices_df)

        #Concat Eu dataframes as well as drop  missing values: 
        energy_prices_eu_frame=pd.concat(energy_prices_eu_dfs,axis=1).dropna()

        #Convert columns to dtype float32: 
        for col in energy_prices_eu_frame.columns:
            energy_prices_eu_frame[col]=energy_prices_eu_frame[col].astype(np.float32)
        energy_prices_eu_frame=energy_prices_eu_frame.resample('h').mean()

        #Set the start of the train dataset:
        datetime_indices=pd.DatetimeIndex(['2024-01-10 00:00:00'])
        energy_prices_eu_frame=energy_prices_eu_frame[(energy_prices_eu_frame.index >= datetime_indices[0])]
        self.set_train_data_energy_prices(train_frame=energy_prices_eu_frame)


    def create_timestep_frames(self):

        #Drop time steps with values equal to 0 across multiple time series:
        drop_indices_transfer_frame=pd.DatetimeIndex(['2025-01-13 23:00:00', '2025-01-14 00:00:00',
                                '2025-01-14 01:00:00', '2025-01-14 02:00:00',
                                '2025-03-30 03:00:00', '2025-04-02 03:00:00'])
        
        drop_indices_train_frame=['2023-09-27 05:00:00', '2023-10-31 20:00:00','2023-11-21 20:00:00', '2023-12-08 11:00:00',
               '2023-12-09 08:00:00', '2023-12-10 14:00:00','2023-12-16 13:00:00', '2024-02-18 23:00:00','2024-04-22 20:00:00', '2024-05-02 18:00:00',
               '2024-05-03 23:00:00', '2024-05-04 06:00:00','2024-05-05 01:00:00', '2024-05-05 03:00:00','2024-05-05 04:00:00', '2024-05-05 19:00:00',
               '2024-05-06 10:00:00', '2024-05-06 21:00:00','2024-05-07 01:00:00', '2024-05-07 17:00:00','2024-05-08 05:00:00', '2024-05-08 19:00:00',
               '2024-05-09 19:00:00', '2024-06-27 00:00:00','2024-06-27 01:00:00', '2024-09-03 16:00:00','2024-09-03 17:00:00', '2024-09-03 18:00:00',
               '2024-09-03 19:00:00', '2024-10-27 22:00:00','2025-01-01 00:00:00', '2025-01-13 23:00:00','2025-01-20 08:00:00', '2025-03-30 01:00:00',
               '2025-03-30 02:00:00', '2025-03-30 03:00:00','2025-03-30 04:00:00', '2025-04-01 16:00:00','2025-04-02 01:00:00', '2025-04-02 03:00:00',
               '2025-04-02 04:00:00', '2025-04-14 23:00:00','2025-04-16 02:00:00', '2025-04-23 06:00:00','2025-04-24 05:00:00']


        frames=[self.train_data_frame]
        # frames = [self.exg_data_frame]
        
        for frame_idx in range(0,len(frames)):

            current_frame_allPredictionHorizons=frames[frame_idx].copy()

            for idx in range(0,len(self.nr_timesteps)):
                #Create train and test splits based on number of timesteps to predict:
                current_nr_timesteps=self.nr_timesteps[idx]
                
                current_nr_datapoints=self.nr_datapoints[idx]
                current_nr_datapoints_train=self.nr_datapoints_train[idx]
                current_nr_datapoints_test=self.nr_datapoints_test[idx]
                current_nr_splits=self.nr_splits[idx]
                
                #Filter out unneccessary number of rows:
                current_frame=current_frame_allPredictionHorizons.iloc[:current_nr_datapoints,:].copy()
                
                subsets=['Train_Set']*current_nr_datapoints_train
                for nr_subset in range(0,current_nr_splits):
                    subsets.extend(['Test_Set'+str(nr_subset+1)]*current_nr_datapoints_test)
                    
                #Append Subsets column:
                current_frame['Subset']=np.array(subsets)
                
                # Filter out rows with 0s:
                # if frame_idx==1:
                #     current_frame=current_frame[(current_frame.index!=drop_indices_transfer_frame[0])&(current_frame.index!=drop_indices_transfer_frame[1])&
                #                                                 (current_frame.index!=drop_indices_transfer_frame[2])&(current_frame.index!=drop_indices_transfer_frame[3])&
                #                                                 (current_frame.index!=drop_indices_transfer_frame[4])&(current_frame.index!=drop_indices_transfer_frame[5])]
                # elif frame_idx==0 and self.eu_total_load_frame is not None:
                #     for drop_idx in drop_indices_train_frame:
                #         current_frame=current_frame[current_frame.index!=drop_idx]
                # else:
                #     pass
                
                #Add a date column:
                current_frame['date']=current_frame.index
                
                #Save the timestep dataframes:
                if frame_idx==0:
                    with open('../Energy_Data/EU_Time_Series/EU_Total_Load_Time_Series_'+str(current_nr_timesteps)+'_Train.pkl','wb') as f:
                        pickle.dump(current_frame,f)
                    
                # elif frame_idx==0 and self.eu_total_load_frame is not None:
                #     with open('../Energy_Data/EU_Time_Series/EU_Energy_Time_Series_'+str(current_nr_timesteps)+'_Transfer_Learn.pkl','wb') as f:
                #         pickle.dump(current_frame,f)
                elif frame_idx==0:# and self.eu_energy_prices_frame is not None:
                    with open('../Energy_Data/EU_Time_Series/EU_Energy_Prices_Time_Series_'+str(current_nr_timesteps)+'_Train.pkl','wb') as f:
                        pickle.dump(current_frame,f)
    
    @ignore_warnings
    def get_timestamp_data(self,df)->pd.DataFrame:
        """Create a dataframe of the time features using the original time series dataset.


        Returns:
        ----------
        df_stamp: pd.DataFrame
        The dataframe containing the extracted time fgeatures, e.g., month, day, week etc.

        """
        df_stamp=df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)+1
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp = df_stamp.drop(['date'], axis=1)
        df_stamp['Subset']=np.array(df['Subset'])

        return df_stamp

    def get_TrainTest_splits(self,data,train_size,
                         test_set_names):
    
        timestamp_data=self.get_timestamp_data(df=data)
        
        train_test_splits=[]
        
        for current_test_set_name in test_set_names:
            last_test_set_index=data[data['Subset']==current_test_set_name].index[-1]
            test_size=data[data['Subset']==current_test_set_name].shape[0]

            first_index_train_set=last_test_set_index-timedelta(hours=train_size+test_size-1)

            current_train_test_split=data.loc[first_index_train_set:last_test_set_index,:]
            current_train_subset=current_train_test_split[current_train_test_split['Subset']!=current_test_set_name]
            current_test_subset=current_train_test_split[current_train_test_split['Subset']==current_test_set_name]

            current_timestamp_split=timestamp_data.loc[first_index_train_set:last_test_set_index,:]
            current_train_timestamps=current_timestamp_split[current_timestamp_split['Subset']!=current_test_set_name]
            current_test_timestamps=current_timestamp_split[current_timestamp_split['Subset']==current_test_set_name]

            #Drop subset column from current train test split:
            current_train_subset=current_train_subset.drop(['date','Subset'],axis=1)
            current_test_subset=current_test_subset.drop(['date','Subset'],axis=1)
            
            current_train_timestamps=current_train_timestamps.drop(['Subset'],axis=1)
            current_test_timestamps=current_test_timestamps.drop(['Subset'],axis=1)


            train_test_splits.append([current_train_subset,current_test_subset,current_train_timestamps,current_test_timestamps])

        return train_test_splits

    

    def create_dataloader(self,current_subset,
                      current_timestamps_subset,
                      seq_len,pred_len,
                      batch_size):

        inputs_for_dataloader=[]
        
        for timestamp_idx in range(0,current_subset.shape[0]-seq_len-pred_len):
            s_begin = timestamp_idx
            s_end = s_begin + seq_len
            #used for building a sequence!
            seq_x=current_subset.iloc[s_begin:s_end].values
            seq_x_mark=current_timestamps_subset.iloc[s_begin:s_end,:].values

            r_begin = s_end
            r_end = r_begin + pred_len
            #used for building targets!
            seq_y=current_subset.iloc[r_begin:r_end,:].values
            seq_y_mark=current_timestamps_subset.iloc[r_begin:r_end].values

            #Inputs specifically only for Basisformer:
            index_list = np.arange(timestamp_idx,timestamp_idx+seq_len+pred_len,1)
            ## normalize the index
            len_denominator=len(current_subset) - seq_len - pred_len + 1
            norm_index = index_list / len_denominator
            
            inputs_for_dataloader.append([seq_x, seq_y,seq_x_mark,seq_y_mark,norm_index])
            
        #Save in dataloader:
        current_dataloader=torch.utils.data.DataLoader(inputs_for_dataloader,batch_size=batch_size,shuffle=False)
    
        return current_dataloader
                 

    def create_timesteps_dataloaders(self):

        for idx in range(0,len(self.nr_timesteps)):
            #Create train and test splits based on number of timesteps to predict:
            current_nr_timesteps=self.nr_timesteps[idx]
            current_nr_datapoints_train=self.nr_datapoints_train[idx]
            
            seq_len=current_nr_timesteps
            pred_len=current_nr_timesteps
            batch_size=32

            
            frame_names=[
                        '../Energy_Data/EU_Time_Series/EU_Energy_Prices_Time_Series_'+str(seq_len)+'_Train.pkl'
                        # '../Energy_Data/EU_Time_Series/EU_Energy_Time_Series_'+str(seq_len)+'_Transfer_Learn.pkl',
                        # '../Energy_Data/EU_Time_Series/EU_Total_Load_Time_Series_'+str(seq_len)+'_Train.pkl',
                        ]
            
            dataloaders=[]

            for frame_name in frame_names:
                
                with open(frame_name,'rb') as f:
                    eu_df=pickle.load(f)
                
                
                #Create all train and test split for the walk-forward validation:
                dataloaders_dict={}
                all_TrainTest_splits=self.get_TrainTest_splits(data=eu_df,
                                                        train_size=current_nr_datapoints_train,
                                                        test_set_names=eu_df['Subset'].value_counts().index[1:])
                

                for split_index in range(0,len(all_TrainTest_splits)):
                    
                    nr_TrainTest_split='Train_Test'+str(split_index+1)
                    current_train_subset=all_TrainTest_splits[split_index][0]
                    current_test_subset=all_TrainTest_splits[split_index][1]
                    
                    #Timestamps data:
                    current_train_subset_timestamps=all_TrainTest_splits[split_index][2]
                    current_test_subset_timestamps=all_TrainTest_splits[split_index][3]

                    
                    #Rescale Train Set & based on statistics also Test Set:
                    current_scaler= StandardScaler()
                    current_train_subset=pd.DataFrame(current_scaler.fit_transform(current_train_subset),
                                                    columns=current_train_subset.columns)
                    current_test_subset=pd.DataFrame(current_scaler.transform(current_test_subset),
                                                    columns=current_test_subset.columns)

                    
                    #Create data loaders:
                    current_train_dataloader=self.create_dataloader(current_subset=current_train_subset,
                                                            current_timestamps_subset=current_train_subset_timestamps,
                                                            seq_len=seq_len,pred_len=pred_len,
                                                            batch_size=batch_size)

                    current_test_dataloader=self.create_dataloader(current_subset=current_test_subset,
                                                            current_timestamps_subset=current_test_subset_timestamps,
                                                            seq_len=seq_len,pred_len=pred_len,
                                                            batch_size=batch_size)


                    dataloaders_dict[nr_TrainTest_split]={'Train_Set_Loader':current_train_dataloader,
                                                        'Test_Set_Loader':current_test_dataloader,
                                                        'Scaler':current_scaler}
                dataloaders.append(dataloaders_dict)
                
                
            # #at index 0: 'EU_Total_Load_Dataloader_torch_'+str(seq_len)+'.pkl'
            # with open('../Energy_Data/EU_Time_Series/EU_Total_Load_Dataloader_torch_'+str(seq_len)+'_Train.pkl','wb') as f:
            #     pickle.dump(dataloaders[0],f)

            # #at index 1: 'Energy_Data_Dataloader_torch_'+str(seq_len)+'.pkl'
            # with open('../Energy_Data/EU_Time_Series/Energy_Data_Dataloader_torch_'+str(seq_len)+'_Transfer_Learn.pkl','wb') as f:
            #     pickle.dump(dataloaders[0],f)

            #at index 2: 'EU_Energy_Prices_Dataloader_torch_'+str(seq_len)+'.pkl'
            with open('../Energy_Data/EU_Time_Series/EU_Energy_Prices_Dataloader_torch_'+str(seq_len)+'_Train.pkl','wb') as f:
                pickle.dump(dataloaders[1],f)



