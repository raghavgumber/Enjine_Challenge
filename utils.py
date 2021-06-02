# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:10:16 2021

@author: Raghav Gumber
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import itertools
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import copy
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.graphics.gofplots import qqplot
sns.set()

raw_features=['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM']

features_for_spread=['12M_MOMENTUM','WEEKLY_MOMENTUM','SALES','TOTAL_ASSETS','R&D_EXP']
add_features=['12M_SALES_ratio','TOTAL_ASSETS_SALES_ratio','R&D_EXP_SALES_ratio','12M_to_Weekly_Avg_YTD','R&D_12M_ratio']
final_features=[raw_features+add_features+[feat+'_YTD_AVG_spread' for feat in features_for_spread]\
                +[feat+'_id_AVG_spread' for feat in features_for_spread]+[feat+'_YTD_AVG' for feat in features_for_spread]+\
                [feat+'_id_AVG' for feat in features_for_spread]][0]

fit_by_id_cols=['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM', '12M_MOMENTUM_YTD_AVG_spread', \
                'WEEKLY_MOMENTUM_YTD_AVG_spread', 'SALES_YTD_AVG_spread', 'TOTAL_ASSETS_YTD_AVG_spread'\
               '12M_MOMENTUM_YTD_AVG', 'WEEKLY_MOMENTUM_YTD_AVG', 'SALES_YTD_AVG', 'TOTAL_ASSETS_YTD_AVG']\
+['12M_SALES_ratio','TOTAL_ASSETS_SALES_ratio','R&D_EXP_SALES_ratio']#['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', 'WEEKLY_MOMENTUM']#[*X_train.columns]




class model_wrapper_walk_fwd():
    def __init__(self,model,X_train,X_test,Y_train,Y_test,scale_by_year_dic,\
                 id_col='security_id',features=['12M_MOMENTUM', 'R&D_EXP', 'SALES', 'TOTAL_ASSETS', \
                                                'WEEKLY_MOMENTUM', '12M_MOMENTUM_YTD_AVG_spread', \
                                                'WEEKLY_MOMENTUM_YTD_AVG_spread', 'SALES_YTD_AVG_spread', \
                                                'TOTAL_ASSETS_YTD_AVG_spread', '12M_MOMENTUM_id_AVG_spread', \
                                                'WEEKLY_MOMENTUM_id_AVG_spread', 'SALES_id_AVG_spread', \
                                                'TOTAL_ASSETS_id_AVG_spread']+['12M_SALES_ratio','TOTAL_ASSETS_SALES_ratio','R&D_EXP_SALES_ratio'],min_num_years_train=10,max_train_data=None,to_scale=True):
    
        '''
        Args:
            model: model object from Models class to implement with args
            X_train,X_test,Y_train,Y_test: dataframes passed in of the data
            scale_by_year_dic: dictionary containing the scaler model for every year (where scaler is fit to data pre year)
            id_col: rfers to security id to ignore
            features: list of features to consider in X
            min_num_years_train: minimum number of years to train before stating CV
            
        '''
        self.model=model
 
        self.features=features
        self.X_train=X_train[features+[id_col,'adjusted_year']].copy(deep=True)
        self.X_test=X_test[features+[id_col,'adjusted_year']].copy(deep=True)
        self.to_scale=to_scale
        self.id_col=id_col
        if to_scale:
            self.Y_train=Y_train.copy(deep=True)
            self.Y_test=Y_test.copy(deep=True)
        else:
            self.Y_train=X_train['SALES']*(1+Y_train).copy(deep=True)
            self.Y_test=X_test['SALES']*(1+Y_test).copy(deep=True)
            
        self.min_num_years_train=min_num_years_train
        self.scale_by_year_dic=scale_by_year_dic
        self.max_train_data=max_train_data

        self.min_train_year= min(self.X_train['adjusted_year'])
        self.stats_by_year_training={}

    def fit(self):
        '''
        fit a model for every year iteratvely on the data available
        '''
        to_scale=self.to_scale
        X_train=self.X_train
        X_test=self.X_test
        Y_train=self.Y_train
        Y_test=self.Y_test
        features=self.features
        min_num_years_train=self.min_num_years_train
        min_train_year= min(self.X_train['adjusted_year'])
        min_train_year=self.min_train_year
        scale_by_year_dic=self.scale_by_year_dic
        max_train_data=self.max_train_data
        model_by_year={}
        train_cv_pred=[]
        actual_recomb=[]
        model=self.model

        
        
        for year in sorted(X_train['adjusted_year'].unique())[min_num_years_train:]:
            #print("training on data till {0} and testing on data for year {1}".format(year-1,year))
            X_train_curr_year=X_train.loc[X_train['adjusted_year']<year].copy(deep=True)
            Y_train_curr_year=Y_train.loc[X_train_curr_year.index]
            #print(X_train_curr_year.shape,Y_train_curr_year.shape)
       
            X_test_curr_year=X_train.loc[X_train['adjusted_year']==year].copy(deep=True)
            Y_test_curr_year=Y_train.loc[X_test_curr_year.index]
            
            scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=scale_by_year_dic[year]
            scaled_X_train,scaled_X_test=scaled_X_train[features],scaled_X_test[features]
            
            
            
                
            if max_train_data!=None:
                #print('reducing train data')
                X_train_curr_year=X_train_curr_year.iloc[-self.max_train_data:]
                Y_train_curr_year=Y_train_curr_year.loc[X_train_curr_year.index]  
                scaled_X_train=scaled_X_train.iloc[-self.max_train_data:]
                #print(scaled_X_train.shape,Y_train_curr_year.shape)
            
        
            model_year=copy.deepcopy(model)
            
            if to_scale:
                model_year.fit(scaled_X_train,Y_train_curr_year)
            else:
                model_year.fit(scaler_X.inverse_transform(scaled_X_train),Y_train_curr_year)
            self.stats_by_year_training[year]=model_year.stats
            model_by_year[year]=model_year
            self.model_by_year=model_by_year
            train_cv_pred_curr_year=pd.Series(model_year.predict(scaled_X_test),index=scaled_X_test.index)
            #print(pd.concat([train_cv_pred_curr_year,Y_test_curr_year],axis=1))
            train_cv_pred.append(train_cv_pred_curr_year)
            actual_recomb.append(Y_test_curr_year)

        true_train=Y_train.loc[X_train['adjusted_year']>=min_train_year+min_num_years_train].rename('true_train')
      
        actual_recomb=pd.concat(actual_recomb,axis=0).loc[true_train.index].rename('actual_recomb')
        train_cv_pred=pd.concat(train_cv_pred,axis=0).loc[true_train.index].rename('train_cv_pred')
        
        int_df=pd.concat([actual_recomb,true_train,train_cv_pred],axis=1)
        int_df['act-true']=int_df['actual_recomb']-int_df['true_train']
        int_df['pred-true']=int_df['train_cv_pred']-int_df['true_train']
        self.int_df=int_df
        
                
        scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=scale_by_year_dic[min(X_test['adjusted_year'])]
        scaled_X_train,scaled_X_test=scaled_X_train[features],scaled_X_test[features]
        Y_train_use=Y_train.copy(deep=True)
        if max_train_data!=None:
            #print('reducing train data')

             
            scaled_X_train=scaled_X_train.iloc[-self.max_train_data:].copy(deep=True)
            Y_train_use=Y_train.loc[X_train_curr_year.index].copy(deep=True)
            #print(scaled_X_train.shape,Y_train_curr_year.shape)

        model_till_now=copy.deepcopy(model)
        
        
        if to_scale:
            model_till_now.fit(scaled_X_train,Y_train_use)
            
        else:
            model_till_now.fit(scaler_X.inverse_transform(scaled_X_train),Y_train_use)
        #print(model_till_now.coef_)
        model_by_year['prior_years_all']=copy.deepcopy(model_till_now)
        self.stats_by_year_training['prior_years_all']=model_till_now.stats
        for year in sorted(X_test['adjusted_year'].unique()):
            X_test_curr_year=X_test.loc[X_test['adjusted_year']==year].copy(deep=True)
            Y_test_curr_year=Y_test.loc[X_test_curr_year.index].copy(deep=True)
            

            X_train_curr_year=pd.concat([X_train.copy(deep=True),X_test.loc[X_test['adjusted_year']<year].copy(deep=True)],axis=0).copy(deep=True)
            Y_train_curr_year=pd.concat([Y_train,Y_test.loc[X_test.loc[X_test['adjusted_year']<year].copy(deep=True).index]],axis=0).copy(deep=True)
            
            scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=scale_by_year_dic[year]
            scaled_X_train,scaled_X_test=scaled_X_train[features],scaled_X_test[features]
          
            if max_train_data!=None:
                print('reducing train data')
                X_train_curr_year=X_train_curr_year.iloc[-self.max_train_data:].copy(deep=True)
                Y_train_curr_year=Y_train_curr_year.iloc[-self.max_train_data:].copy(deep=True)
                print(X_train_curr_year.index)
                print(Y_train_curr_year.index)
                scaled_X_train=scaled_X_train.iloc[-self.max_train_data:].copy(deep=True)
                
            model_year=copy.deepcopy(model)
            if to_scale:
                model_year.fit(scaled_X_train,Y_train_curr_year)
            else:
                model_year.fit(scaler_X.inverse_transform(scaled_X_train),Y_train_curr_year)
            
            model_by_year[year]=model_year
        self.model_by_year=model_by_year
        self.cv_mse_score=self.scoring(mean_squared_error,self.int_df['train_cv_pred'],self.int_df['true_train'])
            
        return self     

    def predict(self,X,walkfwd_testing=True):
        '''
        predict results from X data given based on the year of the data by referring to its specific scaler and model
        '''
        features=self.features
        id_col=self.id_col
        to_scale=self.to_scale
        model_by_year=self.model_by_year
        min_train_year= min(self.X_train['adjusted_year'])
        min_test_year= min(self.X_test['adjusted_year'])
        scale_by_year_dic=self.scale_by_year_dic
        ser=[]
        max_train_data=self.max_train_data
        for year in X['adjusted_year'].unique():

            
            
            if walkfwd_testing:
                dic_lookup=min(2020,year+1)
                scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=scale_by_year_dic[dic_lookup]
                
                
                if year<=min_train_year:
                    dic_lookup='prior_years_all'
                    
            else:
                dic_lookup='prior_years_all'
                scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=scale_by_year_dic[min_test_year]
            
            scaled_X_train,scaled_X_test=scaled_X_train[features],scaled_X_test[features]
                
                
            

            X_test_curr_year=X.loc[X['adjusted_year']==year].copy(deep=True)
            X_test_curr_year=X_test_curr_year[features+[id_col,'adjusted_year']]
            if to_scale:
                scaled_X_test_curr_year=scaler_X.transform(X_test_curr_year)  
            else:
                scaled_X_test_curr_year=X_test_curr_year.copy(deep=True)
            scaled_X_test_curr_year=scaled_X_test_curr_year[features]
            #print(scaled_X_test_curr_year.head())
            ser.append(pd.Series(model_by_year[dic_lookup].predict(scaled_X_test_curr_year),index=scaled_X_test_curr_year.index))
        return pd.concat(ser,axis=0).loc[X.index]
    def scoring(self,scorer,pred,Y):
        return scorer(pred,Y)








class customScaler():
    def __init__(self,scalerFuncDic,fit_by_id=[],id_col='',exclude_cols=['adjusted_year','security_id','year','date'],default_scaler='MinMaxScaler'):
        '''
        Args:
            scalerFuncDic: dictionary {column:scaler}, where every column has its own scaler (MinMaxScaler or StandardScaler), if not present, assume MinMax(default_scaler)
            fit_by_id: columns to scale by their ids else, scale over time
            id_col: id column in data, if not present just ""
            exclude_cols: columns not to scale
            
            
        '''
        self.scalerFuncDic=scalerFuncDic
        self.default_scaler=default_scaler
        self.fit_by_id=fit_by_id
        self.id_col=id_col
        self.exclude_cols=exclude_cols
    def limit_scale(self,scalerFunc,ser):
        '''
        dont let the scaled value go higher than abs. 4 in case of standard scaler
        '''
        if scalerFunc=='MinMaxScaler':
            ser=ser.apply(lambda x: min(1,max(0,x)))
        elif scalerFunc=='StandardScaler':
            ser=ser.apply(lambda x: min(4,abs(x))*np.sign(x))
        else:
            ser=ser
        return ser


    def fit(self,X):
        '''
        fit the scaler to the data, X
        '''
        scalerFuncDic=self.scalerFuncDic
            
            
        id_col=self.id_col
        if type(X)==pd.Series:
            X=pd.DataFrame(X)
        if self.fit_by_id!=[]:
            dfs_by_id={i:X.loc[X[id_col]==i].copy(deep=True) for i in X[id_col].unique()}
            self.dfs_by_id=dfs_by_id
        all_col_scalers={}
        for col in X.columns:
            if col in self.scalerFuncDic.keys():
                scalerFunc=scalerFuncDic[col]
            else:
                scalerFunc=self.default_scaler
            if col in self.fit_by_id:
                col_fit={sec_id:eval(scalerFunc+'()').fit(self.dfs_by_id[sec_id][[col]]) for sec_id in self.dfs_by_id.keys()}
                col_fit['all']=eval(scalerFunc+'()').fit(X[[col]])
            elif col in self.exclude_cols:
                continue
            else:
                col_fit={'all':eval(scalerFunc+'()').fit(X[[col]])}
                
            all_col_scalers[col]=col_fit
        self.all_col_scalers=all_col_scalers

    def transform_col(self,X,col):
        '''
        Args:
            X: dataframe
            Col: column to scale
        Transform column col in X
        '''
        id_col=self.id_col
        if col in self.scalerFuncDic.keys():
            scalerFunc=self.scalerFuncDic[col]
        else:
            scalerFunc=self.default_scaler
        if type(X)==pd.Series:
            X=pd.DataFrame(X)
        og_index=X.index.copy(deep=True)
        if col in self.exclude_cols:
            return X[col]
        elif not(col in self.fit_by_id):
            ser=pd.Series(data=self.all_col_scalers[col]['all'].transform(X[[col]]).T[0],index=og_index).rename(col)
            return self.limit_scale(scalerFunc,ser)
        else:
            dic={}
            for sec_id in X[id_col].unique():
                try:
                    dic[sec_id]=pd.Series(self.all_col_scalers[col][sec_id].transform(X.loc[X[id_col]==sec_id][[col]]).T[0],index=X.loc[X[id_col]==sec_id].index)
                except:
                    dic[sec_id]=pd.Series(np.zeros(len(X.loc[X[id_col]==sec_id].index)),index=X.loc[X[id_col]==sec_id].index)
                    dic[sec_id]=pd.Series(self.all_col_scalers[col]['all'].transform(X.loc[X[id_col]==sec_id][[col]]).T[0],index=X.loc[X[id_col]==sec_id].index)
            ser=pd.concat([*dic.values()],axis=0).loc[og_index].rename(col)
            return self.limit_scale(scalerFunc,ser)
            
    def transform(self,X):
        '''
        Args:
            X: DF
        transforms every column in X
        '''
        if type(X)==pd.Series:
            X=pd.DataFrame(X)
        scaled_X_train=pd.concat([self.transform_col(X,col) for col in X.columns],axis=1)
        
        return scaled_X_train
    def inverse_transform_col(self,X,col):
        '''
        
        Args:
            X: scaled DF
            Col: Col
        Transform it back to unscaled
        '''
        id_col=self.id_col
        if type(X)==pd.Series:
            X=pd.DataFrame(X)
        og_index=X.index.copy(deep=True)
        
        if col in self.exclude_cols:
            return X[col]
        elif not(col in self.fit_by_id):
            
            return pd.Series(data=self.all_col_scalers[col]['all'].inverse_transform(X[[col]]).T[0],index=og_index).rename(col)
        else:
            dic={}
            for sec_id in X[id_col].unique():
                try:
                    dic[sec_id]=pd.Series(self.all_col_scalers[col][sec_id].inverse_transform(X.loc[X[id_col]==sec_id][[col]]).T[0],index=X.loc[X[id_col]==sec_id].index)
                except:
                    dic[sec_id]=pd.Series(np.zeros(len(X.loc[X[id_col]==sec_id].index)),index=X.loc[X[id_col]==sec_id].index)
                    dic[sec_id]=pd.Series(self.all_col_scalers[col]['all'].inverse_transform(X.loc[X[id_col]==sec_id][[col]]).T[0],index=X.loc[X[id_col]==sec_id].index)
            
            return pd.concat([*dic.values()],axis=0).loc[og_index].rename(col)
    def inverse_transform(self,X):
        '''
        
        Args:
            X: scaled DF
        
        Transform it back to unscaled
        '''
        if type(X)==pd.Series:
            X=pd.DataFrame(X)
        return pd.concat([self.inverse_transform_col(X,col) for col in X.columns],axis=1)


class scaled_data_by_year():
    def __init__(self,cScalerX,cScalerY,X,Y):
        '''
        Args:
            cScalerX,cScalerY:Scaler objects (any scaler or customScaler)  for X and Y of data
            X,Y: the data to scale
        Stores a dictionary that has scalers set for every year of the data till the last year, so 2020 in dic
        refers to scaler fit on data till 2019 inclusive
            
            
        '''
        self.cScalerX=cScalerX
        self.cScalerY=cScalerY
        
        self.X=X
        self.Y=Y
    def get_scaled_vals(self,X_train,Y_train,X_test,Y_test):
        
        scaler_X,scaler_Y=copy.deepcopy(self.cScalerX),copy.deepcopy(self.cScalerY)
        scaler_X.fit(X_train)
        scaler_Y.fit(Y_train)
        scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test=scaler_X.transform(X_train),scaler_Y.transform(Y_train),scaler_X.transform(X_test),scaler_Y.transform(Y_test)
        return scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,copy.deepcopy(scaler_X)
    def scale_by_year(self):
        X=self.X
        Y=self.Y
        scale_by_year_dic={}
        for year in sorted(X['adjusted_year'].unique()[1:]):
            print("Scaling on data till {0} for data for year {1}".format(year-1,year))
            X_train_curr_year=X.loc[X['adjusted_year']<year].copy(deep=True)
            Y_train_curr_year=Y.loc[X_train_curr_year.index]

            X_test_curr_year=X.loc[X['adjusted_year']==year].copy(deep=True)
            Y_test_curr_year=Y.loc[X_test_curr_year.index]
            scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=\
            self.get_scaled_vals(X_train_curr_year,Y_train_curr_year,X_test_curr_year,Y_test_curr_year)   
            
            scale_by_year_dic[year]=scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X
        self.scale_by_year_dic=scale_by_year_dic


def get_train_test_dfs(df,tst_ratio=.2,train_recent=None):
    '''
    Args:
        df: df of data to split for train and test
        tst_ratio: amount to preserve for esting, but in reality it is always a bit more because it picks the closest near year to match the ratio
        train_recent: if not none, an integer stating how far to select data for training - i.e rolling window
    returns train and test df
    '''
    
    train_ratio=1-tst_ratio

    ratio_by_year=(df.groupby('adjusted_year').count()['security_id']/len(df)).cumsum()
    if train_ratio==1:
        cutoffYear=ratio_by_year.index.max()
    else:
        cutoffYear=min(ratio_by_year.loc[ratio_by_year>=train_ratio].index)
    train_df=df.loc[df['adjusted_year']<cutoffYear]
    test_df=df.loc[df['adjusted_year']>=cutoffYear]
    if train_recent!=None:
        train_df=train_df.iloc[-train_recent:]
        
    return train_df,test_df



def convert_to_passable_features_df(train_df,test_df,tgt_variable='SalesChange',cols_x=final_features+['security_id','adjusted_year']):
    '''
    Args:
        train_df,test_df: DFs from get_train_test_dfs
        tgt_variable: string identifying the name of dependent variable
        
    
        
    '''
    X_train,Y_train,X_test,Y_test=train_df[cols_x],train_df[tgt_variable],test_df[cols_x],test_df[tgt_variable]
    return X_train,Y_train,X_test,Y_test


def convert_all_params(param_dict):
    all_names = sorted(param_dict)
    combinations = list(itertools.product(*(param_dict[Name] for Name in all_names)))
    return [{name:val for name,val in zip(all_names,combo)} for combo in combinations]

def model_maker(model,args):
    return model(**args)


def get_scaled_vals(X_train,Y_train,X_test,Y_test,fit_by_id_cols,scalerFuncDic={}):
    scaler_X,scaler_Y=customScaler(scalerFuncDic=scalerFuncDic,fit_by_id=fit_by_id_cols,id_col='security_id'),customScaler(scalerFuncDic=scalerFuncDic,fit_by_id=[])
    scaler_X.fit(X_train)
    scaler_Y.fit(Y_train)

    scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test=scaler_X.transform(X_train),scaler_Y.transform(Y_train),scaler_X.transform(X_test),scaler_Y.transform(Y_test)
    return scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X

def feature_correlation_check(X_check,cutoff=10,model_name='linear_regression',\
                              features_og=final_features):
    too_high_var_feats=cutoff
    features_in_use=[*X_check.columns]

    while too_high_var_feats>0:
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X_check[features_in_use].values, i) for i in range(X_check[features_in_use].shape[1])]
        vif["features"] = X_check[features_in_use].columns
        vif=vif.set_index('features')
        too_high_var_feats=len(vif.loc[vif['VIF Factor']>=cutoff])
        features_in_use=[*vif.loc[vif['VIF Factor']<cutoff].index]



    features=[*vif.index]
    
    vif.plot(kind='bar',title='VIF plot features remaining')
    plt.show()
    plt.savefig('Images/Variance_inflation_Factor_Check{}.png'.format(model_name))
    return features


def variance_plot_features_post_scaling(df_engineered_features,fit_by_id_cols,scalerFuncDic={}):
    train_df,test_df=get_train_test_dfs(df_engineered_features,tst_ratio=0,train_recent=None)

    X_train,Y_train,X_test,Y_test=convert_to_passable_features_df(train_df,test_df)
    scaled_X_train,scaled_Y_train,scaled_X_test,scaled_Y_test,scaler_X=get_scaled_vals(X_train,Y_train,X_test,Y_test,fit_by_id_cols=fit_by_id_cols,scalerFuncDic={})
    df_scaled_cov=pd.concat([scaled_X_train,Y_train],axis=1).cov()
    var=pd.Series(np.array(df_scaled_cov).diagonal(),index=df_scaled_cov.index)
    var=var.loc[~var.index.isin(['security_id','adjusted_year','SalesChange'])]
    var.plot(figsize=(10,5),kind='bar',title='variance of features post scaling')
    plt.savefig('Images/Variance of Features Post Scaling.png')
    plt.show()
    
    df_scaled_corr=pd.concat([scaled_X_train,Y_train],axis=1).corr()
    fig, ax = plt.subplots(figsize=(15,12)) 
    ax = sns.heatmap(df_scaled_corr, cmap="YlGnBu",annot=True)
    ax.set_title('Correlation of all Features Post Scaling')
    plt.savefig('Images/Correlation_added_features_post_scaling.png')
    plt.show()




def evaluate_model(model_use,param_dict,all_feats,X_train,X_test,Y_train,Y_test,scaled_by_year_object,max_train_data=None):
    all_args=convert_all_params(param_dict)
    #print(all_args)
    i=0

    all_model_perf=[]
    for arg in all_args:
        for features_keep in all_feats:
            
       
            model=model_use(**arg)
   
            mW=model_wrapper_walk_fwd(model,X_train,X_test,Y_train,Y_test,scale_by_year_dic=scaled_by_year_object.scale_by_year_dic,features=features_keep,min_num_years_train=7,max_train_data=max_train_data)
            #mW=model_wrapper_walk_fwd(model,scaled_X_train,scaled_X_test,Y_train,Y_test,features=final_features,scale_by_year_dic=scaled_by_year_object.scale_by_year_dic)
            mW.fit()
            X_train_cv=X_train.loc[mW.int_df.index].copy(deep=True)
            Y_train_cv=Y_train.loc[mW.int_df.index].copy(deep=True)
            pred_train_actual_Sales_walkfwd,actualNextYearSales_train=X_train_cv['SALES']*(1+mW.int_df['train_cv_pred']),X_train_cv['SALES']*(1+mW.int_df['true_train'])

            res={'arg':arg,'feats':features_keep}
            res['MSE_train_walk_fwd_actual_sales']=mean_squared_error(pred_train_actual_Sales_walkfwd,actualNextYearSales_train)


            all_model_perf.append(res)
            i+=1

            print('done model {0},performance on train return {1}'.format(i-1,mW.cv_mse_score))

            all_model_perf.append(res)
    
    best_arg=sorted(all_model_perf,key=lambda x: x['MSE_train_walk_fwd_actual_sales'],reverse=False)[0]['arg']
    feats=sorted(all_model_perf,key=lambda x: x['MSE_train_walk_fwd_actual_sales'],reverse=False)[0]['feats']
    print(best_arg)
   
    model=model_use(**best_arg)

    mW=model_wrapper_walk_fwd(model,X_train,X_test,Y_train,Y_test,scale_by_year_dic=scaled_by_year_object.scale_by_year_dic,features=feats,min_num_years_train=7)
    #mW=model_wrapper_walk_fwd(model,scaled_X_train,scaled_X_test,Y_train,Y_test,features=final_features,scale_by_year_dic=scaled_by_year_object.scale_by_year_dic)
    mW.fit()

    res={'arg':best_arg,'feats':feats,'cv_mse_score':mW.cv_mse_score}


    pred_test_no_walk_fwd=mW.predict(X_test,walkfwd_testing=False).rename('NoWalkFwd')
    pred_test_walk_fwd=mW.predict(X_test,walkfwd_testing=True).rename('walkFwd')

    pred_test_actual_Sales_no_walkfwd,pred_test_actual_Sales_walkfwd,actualNextYearSales_test=X_test['SALES']*(1+pred_test_no_walk_fwd),X_test['SALES']*(1+pred_test_walk_fwd),X_test['SALES']*(1+Y_test)

    X_train_cv=X_train.loc[mW.int_df.index].copy(deep=True)
    Y_train_cv=Y_train.loc[mW.int_df.index].copy(deep=True)
    pred_train_actual_Sales_walkfwd,actualNextYearSales_train=X_train_cv['SALES']*(1+mW.int_df['train_cv_pred']),X_train_cv['SALES']*(1+mW.int_df['true_train'])


    res['MSE_train_walk_fwd_ret']=mean_squared_error(mW.int_df['train_cv_pred'],mW.int_df['true_train'])
    res['MSE_test_no_walk_fwd_ret']=mean_squared_error(pred_test_no_walk_fwd,Y_test)
    res['MSE_test_walk_fwd_ret']=mean_squared_error(pred_test_walk_fwd,Y_test)

    res['MSE_test_walk_fwd_actual_sales']=mean_squared_error(pred_test_actual_Sales_walkfwd,actualNextYearSales_test) 
    res['MSE_train_walk_fwd_actual_sales']=mean_squared_error(pred_train_actual_Sales_walkfwd,actualNextYearSales_train)


    res['MSE_train_walk_fwd_ret_base']=mean_squared_error(0*mW.int_df['train_cv_pred'],mW.int_df['true_train'])
    res['MSE_test_no_walk_fwd_ret_base']=mean_squared_error(0*pred_test_no_walk_fwd,Y_test)
    res['MSE_test_walk_fwd_ret_base']=mean_squared_error(0*pred_test_walk_fwd,Y_test)

    res['MSE_test_walk_fwd_actual_sales_base']=mean_squared_error(X_test['SALES'],actualNextYearSales_test) 
    res['MSE_train_walk_fwd_actual_sales_base']=mean_squared_error(X_train_cv['SALES'],actualNextYearSales_train)

    return res


def linearity_test(mW,model_name,feats):
    
    fitted_vals=pd.Series(mW.model_by_year['prior_years_all'].predict(mW.scale_by_year_dic[min(mW.X_test['adjusted_year'])][0][feats]),index=mW.Y_train.index)
    resids=(fitted_vals-mW.Y_train)
    
    scale_by_year_dic=mW.scale_by_year_dic
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    sns.regplot(x=fitted_vals, y=mW.Y_train, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    
    
    qqplot(resids, line='s',ax=ax[2])
    ax[2].set_title('QQ Plot')
    plt.show()
    
    plt.savefig('Images/Normality test_{}.png'.format(model_name))
    plt.show()
    return np.mean(resids)


def plot_coef_stability(mW,name,show_Tstat=True):
    coef_stability=pd.concat([mW.stats_by_year_training[year]['coef'].rename(year) for year in mW.stats_by_year_training.keys()],axis=1)
    t_stat_stability=pd.concat([mW.stats_by_year_training[year]['tstat'].rename(year) for year in mW.stats_by_year_training.keys()],axis=1)
    
    if show_Tstat:
        fig, ax = plt.subplots(2,1,figsize=(15,15))

        coef_stability.plot(ax=ax[0],title='coef_stability')
        t_stat_stability.plot(ax=ax[1],title='t_stat_stability')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
    else:
        fig, ax = plt.subplots(1,1,figsize=(15,8))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        coef_stability.plot(ax=ax,title='coef_stability')
    feats=[*coef_stability.index]
    plt.savefig('Images/stability_plot_{}.png'.format(name))
    linearity_test(mW,name,feats)
    return coef_stability,t_stat_stability


