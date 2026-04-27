import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import pickle
import pandas as pd

def visualize_total_load(eu_df_path):

    with open(eu_df_path,'rb') as f:
        eu_df=pickle.load(f)
        
    subsets=list(eu_df['Subset'])

    #Visualize the dataset:
    plt.figure(figsize=(27,7))
    plt.plot(eu_df['Belgium Actual Total Load [MW]'])
    plt.plot(eu_df['Germany Actual Total Load [MW]'])
    plt.plot(eu_df['Poland Actual Total Load [MW]'])

    plt.figtext(0.252,0.8,'Validation\n   Set 1',color= 'darkgray',fontsize=14)
    plt.figtext(0.29,0.8,'Validation\n   Set 2',color= 'darkgray',fontsize=14)
    plt.figtext(0.335,0.8,'Test\n Set 1',color= 'darkgray',fontsize=14)
    plt.figtext(0.37,0.8,'Test\n Set 2',color= 'darkgray',fontsize=14)
    plt.figtext(0.41,0.8,'Test\n Set 3',color= 'darkgray',fontsize=14)
    plt.figtext(0.445,0.8,'Test\n Set 4',color= 'darkgray',fontsize=14)
    plt.figtext(0.485,0.8,'Test\n Set 5',color= 'darkgray',fontsize=14)
    plt.figtext(0.52,0.8,'Test\n Set 6',color= 'darkgray',fontsize=14)
    plt.figtext(0.56,0.8,'Test\n Set 7',color= 'darkgray',fontsize=14)
    plt.figtext(0.60,0.8,'Test\n Set 8',color= 'darkgray',fontsize=14)
    plt.figtext(0.638,0.8,'Test\n Set 9',color= 'darkgray',fontsize=14)
    plt.figtext(0.675,0.8,'Test\n Set 10',color= 'darkgray',fontsize=14)
    plt.figtext(0.71,0.8,'Test\n Set 11',color= 'darkgray',fontsize=14)
    plt.figtext(0.75,0.8,'Test\n Set 12',color= 'darkgray',fontsize=14)
    plt.figtext(0.785,0.8,'Test\n Set 13',color= 'darkgray',fontsize=14)
    plt.figtext(0.825,0.8,'Test\n Set 14',color= 'darkgray',fontsize=14)
    plt.figtext(0.86,0.8,'Test\n Set 15',color= 'darkgray',fontsize=14)


    plt.axvline(x = eu_df[eu_df['Subset']=='Train_Set'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set1'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set2'].index[-1], linestyle='dashed',color = 'black')



    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set3'].index[-1], linestyle='dashed',color = 'black')



    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set4'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set5'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set6'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set7'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set8'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set9'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set10'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set11'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set12'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set13'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set14'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set15'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set16'].index[-1], linestyle='dashed',color = 'black')


    plt.axhline(y=78000,xmin = round(eu_df[eu_df['Subset']=='Train_Set'].shape[0]/eu_df.shape[0],3)+0.01,
                xmax = eu_df[eu_df['Subset'].isin(['Train_Set','Test_Set1','Test_Set2','Test_Set3','Test_Set4','Test_Set5','Test_Set6','Test_Set7','Test_Set8',
                                                'Test_Set9','Test_Set10','Test_Set11','Test_Set12','Test_Set13','Test_Set14','Test_Set15','Test_Set16'])].shape[0]/eu_df.shape[0]-0.015,
                linestyle=':',color = 'darkgray')

    #Set x and y lim limits:
    plt.ylim(1000,90000)
    plt.xlim(eu_df.index[0]-pd.Timedelta(5, unit='d'),eu_df.index[-1]+pd.Timedelta(5, unit='d'))

    #plt.title('Supplier Total Load',fontsize=20)
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Total Energy \n Production in MW',fontsize=20)
    plt.legend(['Belgium','Germany','Poland'],loc=[0.0,-0.2],mode='expand',ncol=3,fontsize=12)
    plt.show()
    #plt.show()
    plt.close()



def visualize_eu_energy(eu_df_path):

    with open(eu_df_path,'rb') as f:
        eu_df=pickle.load(f)
        
    subsets=list(eu_df['Subset'])

    #Visualize the dataset:
    plt.figure(figsize=(27,7))
    plt.plot(eu_df['Germany Fossil Gas [MW]'])
    plt.plot(eu_df['Spain Fossil Gas [MW]'])
    plt.plot(eu_df['Romania Fossil Gas [MW]'])

    plt.figtext(0.252,0.8,'Validation\n   Set 1',color= 'darkgray',fontsize=14)
    plt.figtext(0.29,0.8,'Validation\n   Set 2',color= 'darkgray',fontsize=14)
    plt.figtext(0.335,0.8,'Test\n Set 1',color= 'darkgray',fontsize=14)
    plt.figtext(0.37,0.8,'Test\n Set 2',color= 'darkgray',fontsize=14)
    plt.figtext(0.41,0.8,'Test\n Set 3',color= 'darkgray',fontsize=14)
    plt.figtext(0.445,0.8,'Test\n Set 4',color= 'darkgray',fontsize=14)
    plt.figtext(0.485,0.8,'Test\n Set 5',color= 'darkgray',fontsize=14)
    plt.figtext(0.52,0.8,'Test\n Set 6',color= 'darkgray',fontsize=14)
    plt.figtext(0.56,0.8,'Test\n Set 7',color= 'darkgray',fontsize=14)
    plt.figtext(0.60,0.8,'Test\n Set 8',color= 'darkgray',fontsize=14)
    plt.figtext(0.638,0.8,'Test\n Set 9',color= 'darkgray',fontsize=14)
    plt.figtext(0.675,0.8,'Test\n Set 10',color= 'darkgray',fontsize=14)
    plt.figtext(0.71,0.8,'Test\n Set 11',color= 'darkgray',fontsize=14)
    plt.figtext(0.75,0.8,'Test\n Set 12',color= 'darkgray',fontsize=14)
    plt.figtext(0.785,0.8,'Test\n Set 13',color= 'darkgray',fontsize=14)
    plt.figtext(0.825,0.8,'Test\n Set 14',color= 'darkgray',fontsize=14)
    plt.figtext(0.86,0.8,'Test\n Set 15',color= 'darkgray',fontsize=14)


    plt.axvline(x = eu_df[eu_df['Subset']=='Train_Set'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set1'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set2'].index[-1], linestyle='dashed',color = 'black')



    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set3'].index[-1], linestyle='dashed',color = 'black')



    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set4'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set5'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set6'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set7'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set8'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set9'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set10'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set11'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set12'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set13'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set14'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set15'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set16'].index[-1], linestyle='dashed',color = 'black')


    plt.axhline(y=78000,xmin = round(eu_df[eu_df['Subset']=='Train_Set'].shape[0]/eu_df.shape[0],3)+0.01,
                xmax = eu_df[eu_df['Subset'].isin(['Train_Set','Test_Set1','Test_Set2','Test_Set3','Test_Set4','Test_Set5','Test_Set6','Test_Set7','Test_Set8',
                                                'Test_Set9','Test_Set10','Test_Set11','Test_Set12','Test_Set13','Test_Set14','Test_Set15','Test_Set16'])].shape[0]/eu_df.shape[0]-0.015,
                linestyle=':',color = 'darkgray')

    #Set x and y lim limits:
    plt.ylim(1000,90000)
    plt.xlim(eu_df.index[0]-pd.Timedelta(5, unit='d'),eu_df.index[-1]+pd.Timedelta(5, unit='d'))

    #plt.title('Supplier Total Load',fontsize=20)
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Total Energy \n Production in MW',fontsize=20)
    plt.legend(['Germany','Spain','Romania'],loc=[0.0,-0.2],mode='expand',ncol=3,fontsize=12)
    plt.show()
    #plt.show()
    plt.close()


def visualize_eu_energy_prices(eu_df_path):

    with open(eu_df_path,'rb') as f:
        eu_df=pickle.load(f)
        
    subsets=list(eu_df['Subset'])

    #Visualize the dataset:
    plt.figure(figsize=(27,7))
    plt.plot(eu_df['Germany Day-ahead Price (EUR/MWh)'])
    plt.plot(eu_df['Spain Day-ahead Price (EUR/MWh)'])
    plt.plot(eu_df['Romania Day-ahead Price (EUR/MWh)'])

    plt.figtext(0.252,0.8,'Validation\n   Set 1',color= 'darkgray',fontsize=14)
    plt.figtext(0.29,0.8,'Validation\n   Set 2',color= 'darkgray',fontsize=14)
    plt.figtext(0.335,0.8,'Test\n Set 1',color= 'darkgray',fontsize=14)
    plt.figtext(0.37,0.8,'Test\n Set 2',color= 'darkgray',fontsize=14)
    plt.figtext(0.41,0.8,'Test\n Set 3',color= 'darkgray',fontsize=14)
    plt.figtext(0.445,0.8,'Test\n Set 4',color= 'darkgray',fontsize=14)
    plt.figtext(0.485,0.8,'Test\n Set 5',color= 'darkgray',fontsize=14)
    plt.figtext(0.52,0.8,'Test\n Set 6',color= 'darkgray',fontsize=14)
    plt.figtext(0.56,0.8,'Test\n Set 7',color= 'darkgray',fontsize=14)
    plt.figtext(0.60,0.8,'Test\n Set 8',color= 'darkgray',fontsize=14)
    plt.figtext(0.638,0.8,'Test\n Set 9',color= 'darkgray',fontsize=14)
    plt.figtext(0.675,0.8,'Test\n Set 10',color= 'darkgray',fontsize=14)
    plt.figtext(0.71,0.8,'Test\n Set 11',color= 'darkgray',fontsize=14)
    plt.figtext(0.75,0.8,'Test\n Set 12',color= 'darkgray',fontsize=14)
    plt.figtext(0.785,0.8,'Test\n Set 13',color= 'darkgray',fontsize=14)
    plt.figtext(0.825,0.8,'Test\n Set 14',color= 'darkgray',fontsize=14)
    plt.figtext(0.86,0.8,'Test\n Set 15',color= 'darkgray',fontsize=14)


    plt.axvline(x = eu_df[eu_df['Subset']=='Train_Set'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set1'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set2'].index[-1], linestyle='dashed',color = 'black')



    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set3'].index[-1], linestyle='dashed',color = 'black')



    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set4'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set5'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set6'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set7'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set8'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set9'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set10'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set11'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set12'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set13'].index[-1], linestyle='dashed',color = 'black')


    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set14'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set15'].index[-1], linestyle='dashed',color = 'black')

    plt.axvline(x = eu_df[eu_df['Subset']=='Test_Set16'].index[-1], linestyle='dashed',color = 'black')


    plt.axhline(y=78000,xmin = round(eu_df[eu_df['Subset']=='Train_Set'].shape[0]/eu_df.shape[0],3)+0.01,
                xmax = eu_df[eu_df['Subset'].isin(['Train_Set','Test_Set1','Test_Set2','Test_Set3','Test_Set4','Test_Set5','Test_Set6','Test_Set7','Test_Set8',
                                                'Test_Set9','Test_Set10','Test_Set11','Test_Set12','Test_Set13','Test_Set14','Test_Set15','Test_Set16'])].shape[0]/eu_df.shape[0]-0.015,
                linestyle=':',color = 'darkgray')

    #Set x and y lim limits:
    plt.ylim(-600,1000)
    plt.xlim(eu_df.index[0]-pd.Timedelta(5, unit='d'),eu_df.index[-1]+pd.Timedelta(5, unit='d'))


    #plt.title('Supplier Total Load',fontsize=20)
    plt.xlabel('Date',fontsize=20)
    plt.ylabel('Day-ahead Energy \n Price in EUR per MWh',fontsize=20)
    plt.legend(['Germany','Spain','Romania'],loc=[0.0,-0.2],mode='expand',ncol=3,fontsize=12)
    plt.show()
    #plt.show()
    plt.close()