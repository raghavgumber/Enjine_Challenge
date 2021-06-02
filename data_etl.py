# -*- coding: utf-8 -*-
"""
Created on Sun May 30 20:41:25 2021

@author: Raghav Gumber
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
sns.set()
raw_features=['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM']
class data_etl():
    def __init__(self,dir='Data/data_science_challenge_dataset.csv',raw_features=['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM']):
        self.dir=dir
        self.raw_features=raw_features
    def read_data(self):
        dir=self.dir
        raw_features=self.raw_features
        data=pd.read_csv(dir,parse_dates=['date'])
        data['date']=[x.date() for x in data['date']]
        self.data=data
        ax=(100*data[raw_features].isnull().sum()/len(data)).plot(figsize=(10,5),kind='bar')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title('Missing Data by Column')
    
        plt.savefig('Images/MissingData.png')
        plt.show()
        

    def data_counter(self):
        data=self.data
        data['year']=[d.year for d in data.date]
    
        data['ID_counter']=data.groupby('security_id').cumcount()
        df=data.groupby(['year','security_id']).count()['12M_MOMENTUM'].value_counts().rename('')
        df.plot(kind='bar',title='data distribution by year and security id')
        plt.savefig('Images/DataPerYearAndSecurityID.png')
        plt.show()
        return data
    
    def data_normality(self):
        data=self.data
        most_data_ids=data.security_id.value_counts().head(5).to_dict()
        fig, ax = plt.subplots(5,2,figsize=(18,12))
        i=0
        for sec_id in most_data_ids.keys():
            data.loc[data.security_id==sec_id,'SALES'].reset_index().drop('index',axis=1).rename({'SALES':'Sales through time in years for {}'.format(sec_id)},axis=1).plot(ax=ax[i][0])
            data.loc[data.security_id==sec_id,'SALES'].reset_index().drop('index',axis=1).rename({'SALES':'Sales Distribution for {}'.format(sec_id)},axis=1).plot(kind='density',ax=ax[i][1])
            i+=1
    
        plt.savefig('Images/topPopulatedIDsSalesDistribution.png')
        plt.show()
    
        fig, ax = plt.subplots(5,2,figsize=(15,12))
        i=0
        for sec_id in most_data_ids.keys():
            data.loc[data.security_id==sec_id,'SALES'].pct_change().reset_index().drop('index',axis=1).rename({'SALES':'Sales Change through time in years for {}'.format(sec_id)},axis=1).plot(ax=ax[i][0])
            data.loc[data.security_id==sec_id,'SALES'].pct_change().reset_index().drop('index',axis=1).rename({'SALES':'Sales Change Distribution for {}'.format(sec_id)},axis=1).plot(kind='density',ax=ax[i][1])
            i+=1
    
        plt.savefig('Images/topPopulatedIDsSalesChangeDistribution.png')
        plt.show()
        
        
    ### Transform data such that all next years predictions are in place and for every security such that the 2 dates occur within the same year, its second year is counted as year after
    
    
    
    def imputation_by_id(self,df):
        df['TOTAL_ASSETS'].ffill(inplace=True)

        df['R&D_EXP']=df['R&D_EXP'].fillna(0)

        df['SALES'].ffill(inplace=True)
        ## impute missing sales on the basis of nearest avaiable sales to total asset ratio
        
        df['Sales_to_Total_Assets']=df['SALES']/df['TOTAL_ASSETS']
        df['Sales_to_Total_Assets'].ffill(inplace=True)
        df.loc[df.SALES.isnull(),'SALES']=df.loc[df.SALES.isnull(),'TOTAL_ASSETS']*df.loc[df.SALES.isnull(),'Sales_to_Total_Assets']
        
        return df
    
    def imputation_by_time(self,df):
        ### if missing Sales and recent rales to total_assets for id, grab average over past 20 values to impute missing Sales given assets
        df['Sales_to_Total_Assets_moving_avg']=df['Sales_to_Total_Assets'].ffill().rolling(window=20).mean()
        #df.loc[df.SALES.isnull(),'SALES']=df.loc[df.SALES.isnull(),'TOTAL_ASSETS']*df.loc[df.SALES.isnull(),'Sales_to_Total_Assets_moving_avg']
        #df.loc[df.TOTAL_ASSETS.isnull(),'TOTAL_ASSETS']=df.loc[df.TOTAL_ASSETS.isnull(),'SALES']/df.loc[df.TOTAL_ASSETS.isnull(),'Sales_to_Total_Assets_moving_avg']
        df[['12M_MOMENTUM_running_YTD_avg','WEEKLY_MOMENTUM_running_YTD_avg']]=df.groupby([df.year])['12M_MOMENTUM','WEEKLY_MOMENTUM'].expanding().apply(lambda g: g.mean(skipna=True)).reset_index(level=[0], drop=True)
        df.loc[df['12M_MOMENTUM'].isnull(),'12M_MOMENTUM']=df.loc[df['12M_MOMENTUM'].isnull(),'12M_MOMENTUM_running_YTD_avg']
        df.loc[df['WEEKLY_MOMENTUM'].isnull(),'WEEKLY_MOMENTUM']=df.loc[df['WEEKLY_MOMENTUM'].isnull(),'WEEKLY_MOMENTUM_running_YTD_avg']
        return df
    
    
    def get_data_trans_df(self,data):
        data['ID_counter']=data.groupby('security_id').cumcount()
        data['year']=[d.year for d in data.date]
        dic={ind:ser for ind,ser in data.groupby(['ID_counter','security_id'])}
    
        dic_trans={}
        for k,v in dic.items():
            counter,sec_id=k
            if len(v)>1:
                print(len(v),k)
            try:    
                dic_trans[sec_id][counter]=v.iloc[0].to_dict()
            except:
                dic_trans[sec_id]={counter:v.iloc[0].to_dict()}
        data_trans=[]
        for sec_id in dic_trans.keys():
            df=pd.DataFrame(dic_trans[sec_id]).T
            ### imputing by id
           
            df=self.imputation_by_id(df)
            df['NextYearPredictionSales']=df.SALES.shift(-1)
            df['SalesChange']=df.SALES.apply(np.log).diff().shift(-1)
            #df['R&D_EXP_Change']=df['R&D_EXP'].pct_change().fillna(0)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
            data_trans.append(df)
    
    
        data_trans_df=pd.concat(data_trans,axis=0).sort_values(['date','security_id']).reset_index().drop('index',axis=1)
       
        data_trans_df=self.imputation_by_time(data_trans_df.copy(deep=True))
        return data_trans_df
    
    def plot_missing_features_post_clean(self,data_trans_df):
        data_trans_df.dropna(subset=['SALES'],axis=0)
        data_trans_df.loc[data_trans_df.SALES.isnull()]
        ax=(100*data_trans_df[raw_features].isnull().sum()/len(data_trans_df)).plot(figsize=(10,5),kind='bar')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title('Missing Data by Column Post Imputation')
    
        plt.savefig('Images/MissingDataPostImputation.png')
        plt.show()
    def finalize_data_post_impute(self):
        data_trans_df=self.data_trans_df
        raw_features=self.raw_features
        data_post_impute=data_trans_df.dropna(subset=raw_features+['SalesChange'])
    
        for col in raw_features+['SalesChange']:
            data_post_impute[col]=data_post_impute[col].astype(float)
    
        data_post_impute=data_post_impute[['date','year','security_id']+raw_features+['SalesChange']]
        fig, ax = plt.subplots(figsize=(10,8)) 
        ax = sns.heatmap(data_post_impute[raw_features+['SalesChange']].corr(), cmap="YlGnBu",annot=True)
        plt.savefig('Images/Correlation.png')
        plt.show()
        return data_post_impute
    
    def run_data_etl(self):
        data=self.read_data()
        data=self.data_counter()
        data=self.data
        self.data_normality()
        
        data_trans_df=self.get_data_trans_df(self.data)
        self.plot_missing_features_post_clean(data_trans_df)
        self.data_trans_df=data_trans_df
        data_post_impute=self.finalize_data_post_impute()
        self.data_post_impute=data_post_impute
        
if __name__=='__main__':
    dObj=data_etl()
    dObj.run_data_etl()
    print(dObj.data_post_impute)
    
    