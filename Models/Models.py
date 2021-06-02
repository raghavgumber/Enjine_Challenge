# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:47:21 2021

@author: Raghav Gumber
"""

from sklearn import linear_model
from scipy import stats
import numpy as np
from sklearn import svm
import pandas as pd

class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics    
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2,axis=0) / float(X.shape[0] - X.shape[1])
        
        se = np.array([
            np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
                                                   

        self.t = (self.coef_ / se)[0]
        

        stats=pd.DataFrame(data=np.stack([self.coef_,self.t]).T,columns=['coef','tstat'],index=X.columns)
        self.stats=stats
        
        return self
    
class ElasticNet(linear_model.ElasticNet):
    """
    ElasticNet class after sklearn's, but calculate t-statistics 
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(ElasticNet, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(ElasticNet, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2,axis=0) / float(X.shape[0] - X.shape[1])
        
        se = np.array([
            np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
                                                   

        self.t = (self.coef_ / se)[0]
        

        stats=pd.DataFrame(data=np.stack([self.coef_,self.t]).T,columns=['coef','tstat'],index=X.columns)
        self.stats=stats
        
        return self
    
class LinearSVR(svm.LinearSVR):
    """
    LinearSVR class after sklearn's, but calculate t-statistics
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearSVR, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearSVR, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2,axis=0) / float(X.shape[0] - X.shape[1])
        
        se = np.array([
            np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
                                                   

        self.t = (self.coef_ / se)[0]
        

        stats=pd.DataFrame(data=np.stack([self.coef_,self.t]).T,columns=['coef','tstat'],index=X.columns)
        self.stats=stats
        
        return self