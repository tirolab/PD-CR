# -*- coding: utf-8 -*-
# Written by i3s and the TIRO-lab - UCA, Nice, FRANCE (D.CHARDIN, M.BARLAUD)
# Last update : O5 august 2020

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functions as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale as scale
from sklearn.preprocessing import normalize as normalize
from sklearn.neighbors import KernelDensity

if __name__ == '__main__':
   
    DATADIR = 'datas/'
    df_X = pd.read_csv(DATADIR+'LUNG.csv',delimiter=';', decimal=".",header=0,encoding="ISO-8859-1")
    #df_X = pd.read_csv(DATADIR+'BRAIN.csv',delimiter=';', decimal=".",header=0,encoding="ISO-8859-1")    

    df_names = df_X['Name']
    feature_names=df_names[1:].values.astype(str)
    X = df_X.iloc[1:,1:].values.astype(float).transpose()
    Yr= df_X.iloc[0,1:].values.astype(float).reshape((-1,1))
    Idents=list(df_X.columns[1:])
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
    Ypred,Confidence,Index,Ident,YR,ref =ff.basic_run_eta_molecule(X,Yr,Idents,nbr_clusters,
                       genenames=None,
                       clusternames=None,
                       niter=30,
                       beta=0.25,
                       delta=1.0,
                       eta = 100,
                       nfold=4,
                       random_seed = 2)
    
    Index=np.ravel(Index)
    Ypred=np.ravel(Ypred)
    Yoriginal=YR[Index]
    Ident = [i_temp.loc[:,0].to_numpy() for i_temp in Ident]; Ident=np.ravel(Ident)
    rho=np.ravel(Confidence)
    rho = rho.reshape((-1,1))
    
    for i in range(len(rho)):
        rho[i,0]=round(rho[i,0],3)
    df_confidence=pd.DataFrame(Index,columns=['Index'])
    df_confidence['Ident']=Ident
    df_confidence['Ypred']=Ypred
    df_confidence['rho']=rho
    df_confidence['Yoriginal']=Yoriginal
    
    # plot the histogram
    ff.rhoHist(rho,n_equal_bins=100)   
    
    #X_plot = np.linspace(-1, 1, 100)[:, np.newaxis]
    X_plot = np.linspace(-1, 1, 100)
    X_plot = X_plot.reshape((-1,1))
    #bins = np.linspace(-1, 1, 50)

    fig, ax = plt.subplots(figsize=(8, 4))
   
    # tophat KDE
    #kde = KernelDensity(kernel='tophat', bandwidth=0.1).fit(rho)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.04).fit(rho)
    log_dens = kde.score_samples(X_plot)
    ax.fill(X_plot, np.exp(log_dens), fc='#AAAAFF')
    #ax.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    #ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")
    
    # calculate falserate according to different value of epsilon
    # eps==> maximum value of epsilon
    # num_eps==> the number of epsilon
    falseRate = ff.Predrejection(df_confidence,eps=0.8,num_eps=100) 

