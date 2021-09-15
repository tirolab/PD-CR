# -*- coding: utf-8 -*-
# Written by i3s and the TIRO-lab
# Last update : 19 august 2020 by Michel Barlaud

import functions_l2 as ff
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale as scale
from sklearn.preprocessing import normalize as normalize

if __name__ == '__main__':

    
    DATADIR = 'datas/'
    df_X = pd.read_csv(DATADIR+'LUNG.csv',delimiter=';', decimal=".",header=0,encoding="ISO-8859-1")    
    df_names = df_X['Name']
    feature_names=df_names[1:].values.astype(str)
    X = df_X.iloc[1:,1:].values.astype(float).transpose()
    Yr= df_X.iloc[0,1:].values.astype(float)
    nbr_clusters = len(np.unique(Yr))
    feature_names = df_names.values.astype(str)[1:]

    # Data Preprocessiong
    #X=normalize(X,norm='l1',axis=1)
    X=np.log(X+1)
    X=X-np.mean(X,axis=0)
    X=scale(X,axis=0)
    #X=scale(X,axis=1)
    X=X/ff.normest(X)
    
    
    
    # Processing
    ff.run_primal_dual_L2_eta(X,Yr,nbr_clusters,feature_names,clusternames=None,                                   
                                   niter = 40,
                                   tau = 2,
                                   beta = 1,
                                   eta = 300,
                                   rho = 1,
                                   random_seed = 4,
                                   saveres=False,# True if we want to save the results to the outputPath
                                   outputPath='results/')
   