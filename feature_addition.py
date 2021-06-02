# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:13:53 2021

@author: Raghav Gumber
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
sns.set()
raw_features=['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM']
features_for_spread=['12M_MOMENTUM','WEEKLY_MOMENTUM','SALES','TOTAL_ASSETS','R&D_EXP']
class feature_addition():
    '''
    Args:
        data_post_impute: dataframe of features and return cleaned up and imputed
        features_for_spread: features of which we need to calculate spread over their running YTD average and average in the ID group
        raw_features: list of the raw features
        
    '''
    def __init__(self,data_post_impute,features_for_spread=features_for_spread):
        self.data_post_impute=data_post_impute
        self.features_for_spread=features_for_spread
        self.raw_features=raw_features
        
    def add_features(self):
        '''
        adds the features, check the appendix in jupyter notebook for more details on calculation
        '''
        raw_features=self.raw_features
        data_post_impute=self.data_post_impute
        features_for_spread=self.features_for_spread
        df_engineered_features=data_post_impute.copy(deep=True)
        
    
        df_engineered_features[[feat+'_YTD_AVG' for feat in features_for_spread]]=df_engineered_features.groupby([df_engineered_features.year])[features_for_spread].expanding().apply(lambda g: g.mean(skipna=True)).reset_index(level=[0], drop=True)
        df_engineered_features[[feat+'_id_AVG' for feat in features_for_spread]]=df_engineered_features.groupby([df_engineered_features.security_id])[features_for_spread].expanding().apply(lambda g: g.mean(skipna=True)).reset_index(level=[0], drop=True)
    
        df_engineered_features[[feat+'_YTD_AVG_spread' for feat in features_for_spread]]=(np.array(df_engineered_features[features_for_spread])-np.array(df_engineered_features[[feat+'_YTD_AVG' for feat in features_for_spread]]))#/np.array(df_engineered_features[[feat+'_YTD_AVG' for feat in features_for_spread]])
        df_engineered_features[[feat+'_id_AVG_spread' for feat in features_for_spread]]=(np.array(df_engineered_features[features_for_spread])-np.array(df_engineered_features[[feat+'_id_AVG' for feat in features_for_spread]]))#/np.array(df_engineered_features[[feat+'_id_AVG' for feat in features_for_spread]])
        
       
        df_engineered_features['12M_SALES_ratio']=df_engineered_features['12M_MOMENTUM']/df_engineered_features['SALES']
        df_engineered_features['TOTAL_ASSETS_SALES_ratio']=df_engineered_features['TOTAL_ASSETS']/df_engineered_features['SALES']
        df_engineered_features['R&D_EXP_SALES_ratio']=df_engineered_features['R&D_EXP']/df_engineered_features['SALES'] 
        df_engineered_features['12M_to_Weekly_Avg_YTD']=df_engineered_features['12M_MOMENTUM']/df_engineered_features['WEEKLY_MOMENTUM_YTD_AVG']
        df_engineered_features['R&D_12M_ratio']=df_engineered_features['R&D_EXP']/df_engineered_features['12M_MOMENTUM_id_AVG']
    
        
        #df_engineered_features=df_engineered_features.drop([feat+'_YTD_AVG' for feat in features_for_spread]+[feat+'_id_AVG' for feat in features_for_spread],axis=1)
        df_engineered_features=df_engineered_features.fillna(0).replace([np.inf],np.nan)
        # CUTOFF TAILS
        limit=1
        df_engineered_features['SalesChange'].loc[(df_engineered_features['SalesChange']<limit)&(df_engineered_features['SalesChange']>-limit)].plot(kind='kde',title='SalesChangeDistribution')
        df_engineered_features=df_engineered_features.loc[(df_engineered_features['SalesChange']<limit)&(df_engineered_features['SalesChange']>-limit)]
    
        print('data remaining {0} vs original data length post imputation {1}'.format(len(df_engineered_features),len(data_post_impute)))
        
        
        features_for_spread=['12M_MOMENTUM','WEEKLY_MOMENTUM','SALES','TOTAL_ASSETS','R&D_EXP']
        add_features=['12M_SALES_ratio','TOTAL_ASSETS_SALES_ratio','R&D_EXP_SALES_ratio','12M_to_Weekly_Avg_YTD','R&D_12M_ratio']
        final_features=[raw_features+add_features+[feat+'_YTD_AVG_spread' for feat in features_for_spread]\
                        +[feat+'_id_AVG_spread' for feat in features_for_spread]+[feat+'_YTD_AVG' for feat in features_for_spread]+\
                        [feat+'_id_AVG' for feat in features_for_spread]][0]
    
        fig, ax = plt.subplots(figsize=(17,15)) 
        ax = sns.heatmap(df_engineered_features[final_features+['SalesChange']].corr(), cmap="YlGnBu",annot=True)
        ax.set_title('correlation map of added features raw')
        plt.savefig('Images/Correlation_added_features.png')
        plt.show()
        #COUNTER
        cnt_df=pd.DataFrame(df_engineered_features.groupby('security_id').cumcount().rename('cnt'))
        cnt_df['security_id']=df_engineered_features['security_id']
        cnt_df=cnt_df.join(df_engineered_features.groupby('security_id').min()[['year']],on='security_id')
        df_engineered_features['adjusted_year']=np.array(cnt_df['cnt']+cnt_df['year'])
    
        df_engineered_features['adjusted_year']=df_engineered_features.apply(lambda row:max([row['adjusted_year'],row['year']]),axis=1)
    
        self.df_engineered_features=df_engineered_features
        return df_engineered_features
