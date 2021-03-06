# -*- coding: utf-8 -*-
# Written by i3s and the TIRO-lab
# Last update : O5 august 2020

import functions as ff
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale as scale
from sklearn.preprocessing import normalize as normalize

if __name__ == '__main__':

    
    DATADIR = 'datas/'
    df_X = pd.read_csv(DATADIR+'BRAIN_IDENT.csv',delimiter=';', decimal=",",header=0,encoding="ISO-8859-1")    
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
    
    # algorithm list that you want to compare
    alglist = ['plsda','RF']
    
    # Processing
    _,nbm, accG, loss, W_mean, timeElapsed,\
    topGenes, normW, topGenes_normW,\
    topGenes_mean, normW_mean, topGenes_normW_mean,\
    acctest,accTestCompare,df_timeElapsed,W_all =\
    ff.run_primal_dual_L1N_eta_compare(X,Yr,nbr_clusters,alglist,feature_names,
                                   niter = 30,
                                   beta = 0.25,
                                   eta = 30,#100 for Ident 
                                   delta=1.0,
                                   random_seed = 3,
                                   saveres=False,# True if we want to save the results to the outputPath
                                   outputPath='results/')
    df_featureList = ff.rankFeatures(X,Yr,alglist,feature_names)
    df_featureList.append(ff.TopGenbinary(W_mean, feature_names))
    df_featureList.append(ff.TopGenbinary(W_all, feature_names))