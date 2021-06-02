# -*- coding: utf-8 -*-
"""
Created on Mon May 31 15:43:30 2021

@author: Raghav Gumber
"""

from data_etl import data_etl
from feature_addition import feature_addition
import utils
import pandas as pd
import numpy as np
from utils import fit_by_id_cols
from utils import customScaler
import copy
from utils import final_features

from Models.Models import LinearRegression,ElasticNet,LinearSVR


scalerFuncDic={'12M_MOMENTUM':'StandardScaler','WEEKLY_MOMENTUM':'StandardScaler','12M_SALES_ratio':'StandardScaler',\
               '12M_to_Weekly_Avg_YTD':'StandardScaler', 'R&D_12M_ratio':'StandardScaler','TOTAL_ASSETS_id_AVG_spread':'StandardScaler',\
              'SALES_id_AVG_spread':'StandardScaler','12M_MOMENTUM_id_AVG_spread':'StandardScaler','WEEKLY_MOMENTUM_id_AVG_spread':'StandardScaler'}
class pipeline():
    def __init__(self,tst_ratio=.3,scalerFuncDic=scalerFuncDic,\
                 dir='Data/data_science_challenge_dataset.csv',raw_features=\
                 ['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM']):
        '''
        Args:
            dir: set directory of data
            raw_features-> list of featues in raw data
            scalerFuncDic-> dictionary of columns to apply which scaler on, default scaler is MinMaxScaler
            
        Initialize pipeline
        '''
        self.dir=dir
        self.raw_features=raw_features
        self.data_etl_Obj=data_etl(dir,raw_features)
        self.tst_ratio=tst_ratio
        self.scalerFuncDic=scalerFuncDic
    def cleaned_data(self):
        '''
        Clean the Data and impute missing values
        '''
        self.data_etl_Obj.run_data_etl()      
        self.data_post_impute=self.data_etl_Obj.data_post_impute
        
    def add_features(self):
        '''
        Add other engineered features
        '''
        feature_addition_obj=feature_addition(self.data_post_impute)
        df_engineered_features=feature_addition_obj.add_features()
        self.df_engineered_features=df_engineered_features

    def show_post_scaled_variance_on_all_data(self):
        '''
        show variance in data and correlation (post scaling)
        '''
        utils.variance_plot_features_post_scaling(self.df_engineered_features,fit_by_id_cols,scalerFuncDic=scalerFuncDic)
        
    def run_scaler(self):
        '''
        Run Scaler in such a way as to scale every year's data on the data available up to that year and store
        it in a dictionary by year.
        '''
        tst_ratio=self.tst_ratio
        df_engineered_features=self.df_engineered_features
        train_df,test_df=utils.get_train_test_dfs(df_engineered_features,1,train_recent=None)
        
        X_train,Y_train,X_test,Y_test=utils.convert_to_passable_features_df(train_df,test_df)
        X=pd.concat([X_train,X_test],axis=0)
        Y=pd.concat([Y_train,Y_test],axis=0)
        scaled_by_year_object=utils.scaled_data_by_year(customScaler(scalerFuncDic=scalerFuncDic,fit_by_id=fit_by_id_cols,id_col='security_id',default_scaler='MinMaxScaler'),customScaler(scalerFuncDic={},fit_by_id=[],default_scaler='MinMaxScaler'),X,Y)
        scaled_by_year_object.scale_by_year()
        
        train_df,test_df=utils.get_train_test_dfs(df_engineered_features,tst_ratio,train_recent=None)
        X_train,Y_train,X_test,Y_test=utils.convert_to_passable_features_df(train_df,test_df)
        self.train_df,self.test_df=train_df,test_df
        self.X_train,self.Y_train,self.X_test,self.Y_test=X_train,Y_train,X_test,Y_test
        self.scaled_by_year_object=scaled_by_year_object
        
    def reset_test_train(self,tst_ratio):
        df_engineered_features=self.df_engineered_features
        train_df,test_df=utils.get_train_test_dfs(df_engineered_features,tst_ratio,train_recent=None)
        X_train,Y_train,X_test,Y_test=utils.convert_to_passable_features_df(train_df,test_df)
        self.train_df,self.test_df=train_df,test_df
        self.X_train,self.Y_train,self.X_test,self.Y_test=X_train,Y_train,X_test,Y_test
        

    def get_uncorrelated_variables(self):
        '''
        Run VIF to iteratievely select features that have a VIF score <10 (10 default)
        '''
        X_check=self.scaled_by_year_object.scale_by_year_dic[min(self.X_test['adjusted_year'])][0][final_features].copy(deep=True)
        final_features_use=utils.feature_correlation_check(X_check)
        self.final_features_use=final_features_use
    
    def run_model(self,model_use,all_feats,param_dict,model_name,max_train_data=None):
        '''
        Args:
            model_use-> either LinearRegression,ElasticNet,LinearSVR
            all_feats-> list of lists, where every list is a set of features to check
            param_dict -> all hyperparameters of model to test out
            
        returns best model description (features and hyper parameters that perform best in CV) and coeffecients over time in a DataFrame
            
        '''
        res=utils.evaluate_model(model_use,param_dict,all_feats,self.X_train,self.X_test,self.Y_train,self.Y_test,self.scaled_by_year_object,max_train_data=max_train_data)
        feats=res['feats']
        mW=utils.model_wrapper_walk_fwd(model_use(**res['arg']),self.X_train,self.X_test,self.Y_train,self.Y_test,scale_by_year_dic=self.scaled_by_year_object.scale_by_year_dic,features=feats,min_num_years_train=7,max_train_data=max_train_data)
        mW.fit()
        coef_stability,t_stat_stability=utils.plot_coef_stability(mW,model_name)
        return res,coef_stability,mW
    
    def run_pipeline(self):
        '''
        Run and get everything set up and just leave the model implementation out
        '''
        self.cleaned_data()
        self.add_features()
        self.run_scaler()
        self.show_post_scaled_variance_on_all_data()
        self.get_uncorrelated_variables()

        
if __name__=='__main__':
    pipeline_run=pipeline()
    pipeline_run.run_pipeline()
    
    all_feats=[copy.deepcopy(pipeline_run.final_features_use)]
    param_dict={'fit_intercept':[False],'normalize':[False],'alpha':[.1],'l1_ratio':[.02]}
    model_use=ElasticNet
    model_name='Elastic'
    res_elastic,coef_stability_elastic=pipeline_run.run_model(model_use,all_feats,param_dict,model_name)