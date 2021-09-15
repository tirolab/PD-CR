# -*- coding: utf-8 -*-
import os 
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
import time
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

__all__=['proj_l1ball',
            'proj_l21ball',
            'proj_l12ball',
            'proj_nuclear',
            'sort_weighted_projection',
            'sort_weighted_proj',
            'centroids',
            'class2indicator',
            'nb_Genes',
            'select_feature_w',
            'compute_accuracy',
            'predict_L1',
            'predict_Fro',
            'predict_FISTA',
            'normest',
            'merge_topGene_norm',
            'merge_topGene_norm_acc',
            'compare_2topGenes',
            'heatmap_classification',
            'heatmap_normW',
            'drop_cells',
            'FISTA_Primal',
            'primal_dual_L1N',
            'primal_dual_L1acc',
            'primal_dual_L1OR',
            'primal_dual_L1W',
            'primal_dual_L2',
            'primal_dual_Nuclear',
            'primal_dual_L21',
            'basic_run_eta',
            'basic_run_tabeta',
            'run_FISTA_eta',
            'run_primal_dual_L1N_eta',
            'run_primal_dual_L1acc_eta',
            'run_primal_dual_L1OR_eta',
            'run_primal_dual_L1W_eta',
            'run_primal_dual_L2_eta',
            'run_primal_dual_Nuclear_eta',
            'run_primal_dual_L21_eta',
            'run_FISTA_tabeta',
            'run_primal_dual_L1N_tabeta',
            'run_primal_dual_L1acc_tabeta',
            'run_primal_dual_L1OR_tabeta',
            'run_primal_dual_L1W_tabeta',
            'run_primal_dual_L2_tabeta',
            'run_primal_dual_L21_tabeta',
            'run_primal_dual_Nuclear_tabEtastar'
            ]

help_info = '''
This file contains functions as follow:
| Functions                          | Input                                                                        | Output                    | Type     |
| :--------------------------------- | :--------------------------------------------------------------------------: | :-----------------------: | -------: |
| proj_l1ball                        | y, eta                                                                       | Vproj                     | Function |
| sort_weighted_projection           | y, eta, w, n                                                                 | Vproj                     | Function |
| sort_weighted_proj                 | y, eta, w, n                                                                 | Vproj                     | Function |
| centroids                          | XW, Y, k                                                                     | mu                        | Function |
| class2indicator                    | y, k                                                                         | Y                         | Function |
| nb_Genes                           | w                                                                            | nbG,indGene_w             | Function |
| select_feature_w                   | w, featurenames                                                              | features,normW            | Function |
| compute_accuracy                   | idxR, idx, k                                                                 | Acc_glob,tab_acc          | Function |
| predict_L1                         | Xtest, W, mu                                                                 | Y_predict                 | Function |
| predict_Fro                        | Xtest, W, mu                                                                 | Y_predict                 | Function |
| predict_FISTA                      | Xtest, W, mu                                                                 | Y_predict                 | Function |
| normest                            | X, tol, maxiter                                                              | norm_e                    | Function |
| merge_topGene_norm                 | topGenes, normW, clusternames                                                | df_topGenes_normW         | Function |
| merge_topGene_norm_acc             | topGenes, normW, clusternames, acctest, saveres, file_tag, outputPath        | df_topGenes_normW         | Function |
| compare_2topGenes                  | topGenes1,topGenes2, normW1,normW2, lst_col, nbr_limit, printOut             | out                       | Function |
| heatmap_classification             | Ytest, YR, clusternames, rotate, draw_fig, save_fig, func_tag, outputPath    | Heatmap_matrix            | Function |
| heatmap_normW                      | normW, clusternames, nbr_L, rotate, draw_fig, save_fig, func_tag, outputPath | Heatmap_matrix            | Function |
| drop_cells                         | X,Y,n_fold                                                                   | X_new, Y_new              | Function |
| FISTA_Primal                       | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss     | Function |
| primal_dual_L1N                    | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| primal_dual_L1acc                  | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| primal_dual_L1OR                   | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| primal_dual_L1W                    | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| primal_dual_Nuclear                | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| primal_dual_L2                     | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| primal_dual_L21                    | X,YR,k,param                                                                 | w,mu,nbGenes_fin,loss,Z   | Function |
| basic_run_eta                      | (See funcion help info)                                                      | (See funcion help info)   | Script   |
| basic_run_tabeta                   | (See funcion help info)                                                      | (See funcion help info)   | Script   |
| run_FISTA_eta                      | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_L1N_eta            | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_L1acc_eta          | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_L1OR_eta           | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_L1W_eta            | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_L2_eta             | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_Nuclear_eta        | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_primal_dual_L21_eta            | (Same with basic_run_eta)                                                    | (Same with basic_run_eta) | Script   |
| run_FISTA_tabeta                   | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_L1N_tabeta         | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_L1acc_tabeta       | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_L1OR_tabeta        | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_L1W_tabeta         | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_L2_tabeta          | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_L21_tabeta         | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |
| run_primal_dual_nuclear_tabEtastar | (Same with run_tabeta)                                                       | (Same with run_tabeta)    | Script   |

'''

def proj_l1ball(V,eta,threshold=0.001):
    '''
    Note that the y should be better 1D or after some element-wise operation, the results will turn to be un predictable.
    This function will automatically reshape the y as (m,), where m is the y.size, or the y.shape[0]*y.shape[1].
    '''
    if type(V) is not np.ndarray: 
        V = np.array(V)
    if V.ndim >1:
        V = np.reshape(V,(-1,))
    w= np.maximum( np.absolute(V)-np.amax([np.amax( (np.cumsum(np.sort(np.absolute(V),axis = 0)[::-1],axis = 0) - eta)\
         /(np.arange(V.shape[0])+1) ),0]), 0)*np.sign(V)
    w[np.where(np.abs(w)<threshold)] = 0
    return w

def proj_l21ball(V,eta,axis=1,threshold=0.001):
    '''
    w{np.array} the array to project
    eta {int} the radius of l1 ball
    axis{int, 2-tuple of ints, None}, optional
        Axis or axes along which a norm is calculated
        -If axis is an integer, it specifies the axis of x along which to compute 
         the vector norms. 
        -If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and 
         the matrix norms of these matrices are computed. 
        -If axis is None then either a vector norm (when x is 1-D) or a matrix norm 
         (when x is 2-D) is returned.
    '''
    w = V.copy()
    Y = np.linalg.norm(V,2,axis=axis)
    T = proj_l1ball(Y,eta).reshape(Y.shape)
    max_TY = np.maximum(T,Y)
    if axis==0:
        w = np.multiply(w,np.divide(T,max_TY,out=np.zeros_like(T), where=max_TY!=0))
    else:
        order = tuple(np.arange(len(w.shape)))
        new_order = (order[axis],)+order[:axis]+order[axis+1:]
        reverse_order = (order[axis],)+(0,)+order[axis+1:]
        w = np.multiply(np.transpose(w,new_order),np.divide(T,max_TY,out=np.zeros_like(T), where=max_TY!=0))
        w = np.transpose(w,reverse_order)
    w[np.where(np.abs(w)<threshold)] = 0
    return w

def proj_l12ball(V,eta,axis=1,threshold=0.001):    
    '''
    TO DO:
        - tensor support
        - optimization
        
    '''  
    tol=0.001
    lst_f = []
    test=eta*eta
    
    if axis==0:
        V = V.T
    Vshape = V.shape
    lmbda = 0
    p = np.ones(Vshape[0],dtype=int)*(Vshape[1]-1) # to change in case of tensor
    delta = np.zeros(Vshape[0])
    V_abs = np.abs(V) #maybe transposed if change the value of axis
    
    V0 = np.sort(V_abs,axis=1)[:,::-1]
    sgn = np.sign(V)
    V_sum = np.cumsum(V0,axis=1)
#    K = 10
#    for k in range(K):
    while  test>tol : 
        # update lambda      
        sum0 = np.array(list(map(lambda x,y:y[x],p,V_sum)))
        sum1 = np.sum(np.power(sum0/(1+lmbda*p),2))
        sum2 = np.sum(p*(np.power(sum0,2)/np.power(1+lmbda*p,3)))
        test = sum1-eta*eta
        lmbda = lmbda + test/(2*sum2)
        lst_f.append(test)
        # update p
        p = np.argmax(V_sum/(1+lmbda*np.arange(1,Vshape[1]+1)),axis=1)
    
    delta = lmbda*(np.array(list(map(lambda x,y:y[x],p,V_sum)))/(1+lmbda*p))
    W = V_abs-delta.reshape((-1,1))
    W[W<0]=0
    W = W*sgn
    W[np.where(np.abs(W)<threshold)] = 0
    return W

    
def proj_nuclear(V,eta_star,threshold=0.001):
        L,S0,R = np.linalg.svd(V,full_matrices=False)
 #      norm_nuclear = S0.sum() 
        vs1 = proj_l1ball(S0.reshape((-1,)),eta_star)
        S1 = vs1.reshape(S0.shape)
        w = np.matmul(L, S1[..., None] * R)
        w[np.where(np.abs(w)<threshold)] = 0
        return w

def sort_weighted_projection(y, eta, w, n=None):
    '''
    Weighted projection on l1 ball. 
    When w = ones(y.shape) this function is equivalent to proj_l1ball(y,eta)
    It's not a simple 
    Compared to the original version, we have modified :
    * Changed a by eta
    * Regard x as the output and give it a initial value of zeros
    * Fixed n as the length of x if it's not given
    '''
    # ==== changed part
    if type(y) is not np.ndarray: 
        y = np.array(y)
    if y.ndim >1:
        y = np.reshape(y,(-1,))
    if w.ndim >1:
        w = np.reshape(w,(-1,))
    if np.any(w<0):
        raise ValueError('sort_weighted_projection: The weight should be positive')
    sign_y = np.sign(y) 
    y0 = y*sign_y
    x = np.zeros(y.shape)
    if n is None:
        n = len(x)
    z = np.divide(y0,w)
    p = np.argsort(z)[::-1]
    WYs = 0.0
    Ws = 0.0   
    for j in p:
        WYs += w[j]*y0[j]
        Ws += w[j]*w[j]
        if ((WYs - eta) / Ws) > z[j]:
            break   
    WYs -= w[j]*y0[j]
    Ws -= w[j]*w[j]
    L = (WYs - eta) / Ws
    if n == len(x):
        x = np.maximum(0,y0-w*L)
    else:
        for i in range(n):
            x[i] = max(0,y0[i]-w[i]*L)
    x *= sign_y
    return x

def sort_weighted_proj(y, eta, w, n=None):
    '''
    Weighted projection on l1 ball. V2
    When w = ones(y.shape) this function is equivalent to proj_l1ball(y,eta)
    It's not a simple 
    Instead of using for loop, we use np.cumsum
    '''
    # ==== changed part
    if type(y) is not np.ndarray: 
        y = np.array(y)
    if y.ndim >1:
        y = np.reshape(y,(-1,))
    if np.any(w<0):
        raise ValueError('sort_weighted_projection: The weight should be positive')
    sign_y = np.sign(y) 
    y0 = y*sign_y
    x = np.zeros(y.shape)
    if n is None:
        n = len(x)
    z = np.divide(y0,w)
    p =  np.argsort(z)[::-1]
    yp=y0[p]
    wp=w[p]
    Wys = np.cumsum(yp*wp)
    Ws = np.cumsum(wp*wp)
    L = (Wys -eta) /Ws
    ind = np.where(L>z[p])[0]
    if ind.size==0:# all elements of (Wys -eta) /Ws are <z[p], so L is the last one
        L = L[-1]
    else:
        L = L[ind[0]]
    if n == len(x):
        x = np.maximum(0,y0-w*L)
    else:
        x[0:n] = np.maximum(0,y0[0:n]-w[0:n]*L)
    x *= sign_y
    return x

def centroids(XW,Y,k):
    Y = np.reshape(Y,-1)
    d = XW.shape[1]
    mu = np.zeros((k,d))
    '''
    since in python the index starts from 0 not from 1, 
    here the Y==i will be change to Y==(i+1)
    Or the values in Y need to be changed
    '''
    for i in range(k):
        C = XW[Y==(i+1),:]
        mu[i,:] = np.mean(C,axis=0)
    return mu

def class2indicator(y,k):
    if len(y.shape)>1:
        # Either throw exception or transform y, here the latter is chosen.
        # Note that a list object has no attribute 'flatten()' as np.array do,
        # We use x = np.reshape(y,-1) instead of x = y.flatten() in case of 
        # the type of 'list' of argument y
        y = np.reshape(y,-1)   

    n = len(y)
    Y = np.zeros((n,k)) # dtype=float by default
    '''
    since in python the index starts from 0 not from 1, 
    here the y==i in matlab will be change to y==(i+1)
    '''
    for i in range(k):
        Y[:,i] = (y==(i+1))
    return Y


def nb_Genes2(w):
    # Return the number of selected genes from the matrix (numpy.ndarray) w
    ind_genes = np.linalg.norm(w,axis=1)
    indGene_w = np.where(ind_genes>0)[0]
    nbG = indGene_w.nonzero()[0].size
    return nbG,indGene_w

def nb_Genes(w):
    # Return the number of selected genes from the matrix (numpy.ndarray) w
    ind_genes = np.linalg.norm(w,axis=1)
    indGene_w = np.where(ind_genes>0)[0]
    nbG =  len(w.reshape(-1).nonzero()[0])
    return nbG,indGene_w
 
    
def select_feature_w(w,featurenames):
    k = w.shape[1]
    d = w.shape[0]
    lst_features = []
    lst_norm=[]
    for i in range(k):
        s_tmp = w[:,i]                          # the i-th column
        f_tmp = np.abs(s_tmp)                   # the absolute values of this column
        ind = np.argsort(f_tmp)[::-1]           # the indices of the sorted abs column (descending order)
        f_tmp = np.sort(f_tmp)[::-1]            # the sorted abs column (descending order)
        nonzero_inds = np.nonzero(f_tmp)[0]     # the nonzero indices
        lst_f = []
        lst_n=[]
        if len(nonzero_inds)>0:
            nozero_ind = nonzero_inds[-1] #choose the last nonzero index 
            if nozero_ind ==0:
                lst_f.append(featurenames[ind[0]])
                lst_n.append(s_tmp[ind[0]])
            else:
                for j in range(nozero_ind+1):
                    lst_f.append(featurenames[ind[j]])
                    lst_n = s_tmp[ind[0:(nozero_ind+1)]]
        lst_features.append(lst_f)
        lst_norm.append(lst_n)
            
    n_cols_f = len(lst_features)
    n_rows_f = max(map(len,lst_features)) #maxmum subset length 
    n_cols_n = len(lst_norm)
    n_rows_n = max(map(len,lst_norm))

    
    for i in range(n_cols_f):
        ft = np.array(lst_features[i])
        ft.resize(n_rows_f,refcheck=False)
        nt = np.array(lst_norm[i])
        nt.resize(n_rows_n,refcheck=False)
        if i ==0:
            features = ft;normW=nt;continue
        features = np.vstack((features,ft))
        normW = np.vstack((normW,nt))
    features = features.T
    normW = normW.T
    return features,normW

def compute_accuracy(idxR,idx,k):
    """
    # ===============================
    #----- INPUT
    # idxR : real labels
    # idx  : estimated labels
    # k    : number of class
    #----- OUTPUT
    # ACC_glob : global accuracy
    # tab_acc  : accuracy per class
    # ===============================
    """
 
    # Note that Python native sum function works better on list than on numpy.array
    # while numpy.sum function works better on numpy.array than on list.
    # So it will choose numpy.array as the default type for idxR and idx
    if type(idxR) is not np.array:
        idxR = np.array(idxR)
    if type(idx) is not np.array:
        idx = np.array(idx)
    if idxR.ndim==2 and 1 not in idxR.shape:
        idxR = np.reshape(idxR,(-1,1))
    if idx.ndim==1:
        idx = np.reshape(idx,idxR.shape)
    # Global accuracy
    y = np.sum(idxR==idx)
    ACC_glob = y/len(idxR)
    
    # Accuracy per class
    tab_acc = np.zeros((1,k))
    '''
    since in python the index starts from 0 not from 1, 
    here the idx(ind)==j in matlab will be change to idx[ind]==(j+1)
    '''    
    for j in range(k):
        ind = np.where(idxR==(j+1))[0]
        if len(ind)==0:
            tab_acc[0,j] = 0.0
        else:
            tab_acc[0,j] = int(np.sum(idx[ind]==(j+1)))/len(ind)
    return ACC_glob,tab_acc

def predict_L1(Xtest,W,mu):
    # Chambolle_Predict
    k = mu.shape[0]
    m = Xtest.shape[0]
    Ytest = np.zeros((m,1))
    
    for i in range(m):
        distmu = np.zeros((1,k))
        XWi = np.matmul(Xtest[i,:],W)
        for j in range(k):
            distmu[0,j] = np.linalg.norm(XWi - mu[j,:],1)
        Ytest[i] = np.argmin(distmu)+1 #Since in Python the index starts from 0
    return Ytest

def predict_Fro(Xtest,W,mu):
    # Chambolle_Predict
    k = mu.shape[0]
    m = Xtest.shape[0]
    Ytest = np.zeros((m,1))
    
    for i in range(m):
        distmu = np.zeros((1,k))
        XWi = np.matmul(Xtest[i,:],W)
        for j in range(k):
            distmu[0,j] = np.linalg.norm(XWi - mu[j,:])
        Ytest[i] = np.argmin(distmu)+1 #Since in Python the index starts from 0
    return Ytest

def predict_FISTA(Xtest,W,mu):
    # Chambolle_Predict
    k = mu.shape[0]
    m = Xtest.shape[0]
    Ytest = np.zeros((m,1))
    
    for i in range(m):
        distmu = np.zeros((1,k))
        XWi = np.matmul(Xtest[i,:],W)
        for j in range(k):
            distmu[0,j] = np.linalg.norm(XWi - mu[j,:],2)
        Ytest[i] = np.argmin(distmu)+1 #Since in Python the index starts from 0
    return Ytest

def normest(X,
            tol=1.0e-6,
            maxiter=100):
    # import necessary modules
    import scipy.sparse
    import numpy as np
    import warnings
    
    if scipy.sparse.issparse(X):
        x = np.array(np.sum(np.abs(X),axis=0))
        x = np.reshape(x,max(x.shape))
    elif type(X)==np.matrix:
        x = np.sum(np.abs(np.asarray(X)),axis=0)
        x = np.reshape(x,max(x.shape))
    else:
        x = np.sum(np.abs(X),axis=0)
        
    norm_e = np.linalg.norm(x)
    if norm_e == 0:
        return norm_e
    
    x = x/norm_e
    norm_e0 = 0
    count = 0
    while np.abs(norm_e - norm_e0) > tol*norm_e:
        norm_e0 = norm_e
        Xx = np.matmul(X,x)
        if np.count_nonzero(Xx)==0:
            Xx = np.random.rand(Xx.shape[0])
        x = np.matmul(X.T,Xx)
        normx = np.linalg.norm(x)
        norm_e = normx/np.linalg.norm(Xx)
        x = x/normx
        count+=1
        if count > maxiter:
            warnings.warn('Normest::NotConverge:the number of iterations exceeds {} times.\nThe error is {}, the tolerance is {}'\
                          .format(maxiter,np.abs(norm_e - norm_e0),tol),RuntimeWarning)
            break
    return norm_e

def merge_topGene_norm(topGenes,normW,clusternames):
    """
    # =====================================================================
    # It merge the two output from function select_features_w into a new 
    # pandas.DataFrame whose columns will be the elements in clusternames
    # and each of the column will have two subcolumns: topGenes and weight
    # 
    #----- INPUT
    # topGenes      : ndarray of top Genes chosen by select_features_w
    # normW         : normWeight of each genes given by select_features_w
    # clusternames  : A list of the names of each class.
    #----- OUTPUT
    # df_res        : A DataFrame with each colum the first subcolumn the genes
    #                  and second subcolumn their norm of weight
    # =====================================================================
    """
    if topGenes.shape!=normW.shape:
        raise ValueError('The dimension of the two input should be the same')
    m,n = topGenes.shape
    nbC = len(clusternames)
    res = np.dstack((topGenes,normW))
    res = res.reshape(m,2*n)
    lst_col = []
    for i in range(nbC):
        lst_col.append((clusternames[i],'topGenes'))
        lst_col.append((clusternames[i],'Weights'))
    df_res = pd.DataFrame(res,columns=lst_col)
    df_res.columns = pd.MultiIndex.from_tuples(df_res.columns,names=['CluserNames','Attributes'])
    return df_res

def merge_topGene_norm_acc(topGenes,normW,clusternames,acctest,nbr_features=30,
                            saveres=False,file_tag=None,outputPath='../results/'):
    """
    # ===============================================================================================   \n
    # Based on the function merge_topGebe_norm, replace the column name for                             \n
    # normW by the accuracy                                                                             \n
    #----- INPUT                                                                                         \n
    # topGenes      (ndarray or DataFrame)  : Top Genes chosen by select_features_w                     \n
    # normW         (ndarray or DataFrame)  : The normWeight of each genes given by select_features_w   \n
    # clusternames  (list or array)         : A list of the names of each class                         \n
    # acctest       (list or array)         : The list of the test accuracy                             \n
    # saveres       (optional, boolean)     : True if we want to save the result to local               \n
    # file_tag      (optional, string)      : A file tag which will be the prefix of the file name      \n
    # outputPath    (optional, string)      : The output Path of the file                               \n
    # ----- OUTPUT                                                                                       \n
    # df_res        : A DataFrame with each colum the first subcolumn the genes                         \n
    #                  and second subcolumn their norm of weight                                        \n
    # ===============================================================================================   \n
    """
    if type(topGenes) is pd.DataFrame:
        topGenes = topGenes.values
    if type(normW) is pd.DataFrame:
        normW = normW.values
    if topGenes.shape!=normW.shape:
        raise ValueError('The dimension of the two input should be the same')
    m,n = topGenes.shape
    nbC = len(clusternames)
    res = np.dstack((topGenes,normW))
    res = res.reshape(m,2*n)
    lst_col = []
    acctest_mean=acctest.values.tolist()[4]
    for i in range(nbC):
        lst_col.append((clusternames[i],'topGenes'))
        astr=str(acctest_mean[i])
        lst_col.append((astr,'Weights'))
    df_res = pd.DataFrame(res[0:nbr_features,:],columns=lst_col)
    df_res.columns = pd.MultiIndex.from_tuples(df_res.columns,names=['CluserNames','Attributes'])
    if saveres:
        df_res.to_csv('{}{}_Heatmap of Acc_normW_Topgenes.csv'.format(outputPath,file_tag),sep=';')
    return df_res

def compare_2topGenes(topGenes1,topGenes2,normW1=None,normW2=None,lst_col=None,nbr_limit=30,printOut=False):
    """
    #=======================================================================================
    # Compare column by column the elements between to topGenes, it choose for 
    # each column first "nbr" elements to check.
    # The two topGenes should be in same size of columns
    # ----- INPUT
    # topGenes1, topGenes2  (DataFrame)         : Two topGenes to be compared
    # normW1, normW2        (DataFrame,optional): Two matrix of weights correspondent. Default: None 
    # lst_col               (list, optional)    : If given, only the chosen column will be compared. Default: None
    # nbr_limit             (scalar, optional)  : Number of the lines to be compared. Default: 30
    # printOut              (boolean, optional) : If True, the comparison result will be shown on screen. Default: False
    # ----- OUTPUT  
    # out                   (string)            : It returns a string of the comparing result as output.
    #=======================================================================================
    """
    import pandas as pd
    import numpy as np
    if type(topGenes1) != type(topGenes2):
        raise ValueError('The two topGenes to be compared should be of the same type.')
    if type(topGenes1) is not pd.DataFrame:
        col = ['C'+str(i) for i in topGenes1.shape[1]]
        topGenes1=pd.DataFrame(topGenes1,columns=col)
        topGenes2=pd.DataFrame(topGenes2,columns=col)
    
    out = []
    out.append('Comparing the two TopGenes:\n')
    # After the benchmark, the appended list and then converted to whole string seems to be the least consuming
    
    list_name = list(topGenes1.columns)
    if lst_col is not None:
        list_name = [list_name[ind] for ind in lst_col]
    for name in list_name:
        out.append('{0:{fill}{align}40}\n'.format(' Class %s '%name,fill='=',align='^'))
        col_1 = np.array(topGenes1[[name]],dtype=str)
        col_2 = np.array(topGenes2[[name]],dtype=str)
        
        # Here np.nozero will return a tuple of 2 array corresponding the first
        # and the second dimension while the value of second dimension will 
        # always be 0. So the first dimension's last location+1 will be the length
        # of nonzero arrays and that it's just the location of the first zero
        # element
        length_nonzero_1 = np.nonzero(col_1)[0][-1] + 1 
        length_nonzero_2 = np.nonzero(col_2)[0][-1] + 1 
        # np.nonzero will not detect '0.0' as zero type
        if all(col_1=='0.0'):
            length_nonzero_1 = 0
        if all(col_2=='0.0'):
            length_nonzero_2 = 0
        length_min = min(length_nonzero_1,length_nonzero_2)
        # Check if at least one of the classes contains only zero and avoid the error
        if length_min ==0 and length_nonzero_1==length_nonzero_2:
            out.append('* Warning: No feature is selected for both two class\n Skipped for this class')
            continue
        elif length_min==0 and length_nonzero_1>0:
            out.append('* Warning: No feature is selected for this class in TopGenes2\n')
            out.append('* All {} elements are included only in topGenes1:\n'.format(min(length_nonzero_1,nbr_limit)))
            for k in range(min(length_nonzero_1,nbr_limit)):
                if normW1 is None:
                    out.append(' (%s)\n'%(str(col_1[k,0])))
                else:
                    out.append(' (%s, %s)\n'%(str(col_1[k,0]),normW1[[name]].iloc[k,0]))
            continue
        elif length_min==0 and length_nonzero_2>0:
            out.append('* Warning: No feature is selected for this class in TopGenes1\n')
            out.append('* All {} elements are included only in topGenes2:\n'.format(min(length_nonzero_2,nbr_limit)))
            for k in range(min(length_nonzero_2,nbr_limit)):
                if normW2 is None:
                    out.append(' (%s)\n'%(str(col_2[k,0])))
                else:
                    out.append(' (%s, %s)\n'%(str(col_2[k,0]),normW2[[name]].iloc[k,0]))
            continue
        
        if  length_min < nbr_limit:
            length = length_min
            out.append('* Warning: In this column, the 1st topGenes has {} nozero elements\n* while the 2nd one has {} nonzero elements\n'\
                       .format(length_nonzero_1,length_nonzero_2))
            out.append('* So only first %d elements are compared\n\n'%length_min)
        else:
            length = nbr_limit
        set_1 = col_1[0:length]
        set_2 = col_2[0:length]
        set_common = np.intersect1d(set_1,set_2)# Have in common
        set_o1=np.setdiff1d(set_1,set_2)#Exclusively in topGenes1
        set_o2=np.setdiff1d(set_2,set_1)#Exclusively in topGenes2
        lc=len(set_common)
        # print exclusively in topGenes1
        out.append('Included exclusively in first topGenes: {} elements in total.\n'\
              .format(length-lc))
        if length-lc >0:
            if normW1 is None:
                out.append('Details:(Name)\n')
            else:
                out.append('Details:(Name,Weight)\n')
            idx_i,idx_j=np.where(topGenes1[[name]].isin(set_o1))
            for i,j in zip(idx_i,idx_j):
                if normW1 is None:
                    out.append(' (%s)\n'%str(set_1[i,j]))
                else:
                    out.append(' (%s, %s)\n'%(str(set_1[i,j]),str(normW1[[name]].iloc[i,j])))
        out.append('\nNumber of elements in common:{}\n'.format(lc))
        # print exclusively in topGenes1
        out.append('\nIncluded exclusively in second topGenes: {} elements in total.\n'\
              .format(length-lc))
        if length-lc >0:
            if normW2 is None:
                out.append('Details:(Name)\n')
            else:
                out.append('Details:(Name,Weight)\n')
            idx_i,idx_j=np.where(topGenes2[[name]].isin(set_o2))
            for i,j in zip(idx_i,idx_j):
                if normW2 is None:
                    out.append(' (%s)\n'%str(set_2[i,j]))
                else:
                    out.append(' (%s, %s)\n'%(str(set_2[i,j]),str(normW2[[name]].iloc[i,j])))
        out.append('{:-<40}\n'.format(''))
    out=''.join(out)
    if printOut==True:
        print(out)
    return out
    
def heatmap_classification(Ytest,YR,clusternames,
                           rotate=45,
                           draw_fig=False,
                           save_fig=False,
                           func_tag=None,
                           outputPath='../results/'):
    '''
    #=====================================================
    # It takes the predicted labels (Ytest), true labels (YR)  
    # and a list of the names of clusters (clusternames)  
    # as input and provide the heatmap matrix as the output 
    #=====================================================
    '''
    k = len(np.unique(YR)) # If we need to automatically find a k
    Heatmap_matrix = np.zeros((k,k))
    for i in (np.arange(k)+1):
        for j in (np.arange(k)+1):
            a = np.where(Ytest[YR==i]==j,1,0).sum()# number Ytest ==j where YR==i
            b = np.where(YR==i,1,0).sum()
            Heatmap_matrix[i-1,j-1] = a/b
        
    # Plotting
    if draw_fig == True:
        plt.figure()
        annot=False
        if k >10:
            annot=False
        if clusternames is not None:
            axes=sns.heatmap(Heatmap_matrix,
                             cmap='jet',annot=annot, fmt='.2f',
                             xticklabels=clusternames, yticklabels=clusternames)
        else:
            axes=sns.heatmap(Heatmap_matrix,
                             cmap='jet',annot=annot, fmt='.2f')   
        axes.set_xlabel('Predicted true positive',fontsize=14)
        axes.set_ylabel('Ground true',fontsize=14)
        axes.tick_params(labelsize=7)
        plt.xticks(rotation=rotate)
        axes.set_title('Heatmap of confusion Matrix',fontsize=14)
        plt.tight_layout()
        if save_fig == True:
            plt.savefig('{}{}_Heatmap_of_confusion_Matrix.png'.format(outputPath,func_tag))
    return Heatmap_matrix

def heatmap_normW(normW,clusternames=None,nbr_l=10,
                  rotate=45,
                  draw_fig=False,
                  save_fig=False,
                  func_tag=None,
                  outputPath='../results/'):
    '''
    #=====================================================
    # It takes the predicted labels (Ytest), true labels (YR)  
    # and the number of clusters (k) as input and provide the 
    # heatmap matrix as the output 
    #=====================================================
    '''
    A = np.abs(normW)
    AN = A/A[0,:]
    if normW.shape[0]<nbr_l:
        nbr_l = normW.shape[0]
    ANR = AN[0:nbr_l,:]
    annot=False
    if draw_fig == True:
        plt.figure(figsize=(10,6))
#        axes2=sns.heatmap(ANR,cmap='jet',annot=annot,fmt='.3f') 
        if clusternames is None:
            axes2=sns.heatmap(ANR,cmap='jet',annot=annot,fmt='.3f',
                              yticklabels=np.linspace(1,nbr_l,num=nbr_l,endpoint=True,dtype=int))  
        else:
            axes2=sns.heatmap(ANR,cmap='jet',annot=annot,fmt='.3f',
                              xticklabels=clusternames,
                              yticklabels=np.linspace(1,nbr_l,num=nbr_l,endpoint=True,dtype=int))
            plt.xticks(rotation=rotate)
        axes2.set_ylabel('Features',fontsize=14)
        axes2.set_xlabel('Clusters',fontsize=14)
        axes2.tick_params(labelsize=7)
        axes2.set_title('Heatmap of Matrix W',fontsize=14)
        plt.tight_layout()
        if save_fig == True:
            plt.savefig('{}{}_Heatmap_of_signature.png'.format(outputPath,func_tag))        
    return ANR

def drop_cells(X,Y,n_fold):
    """
    # ====================================================================
    # This function will detect whether the size of the first dimension of
    # X is divisible by n_fold. If not, it will remove the n_diff rows from
    # the biggest class(with the largest size in Y) where n_diff=len(Y)%n_fold
    #
    # ---- Input
    # X         : The data
    # Y         : The label
    # n_fold    : The number of fold
    # --- Output
    # X_new, Y_new : The new data and the new label
    # =====================================================================
    """
    m,d = X.shape
    if m%n_fold ==0:
        return X,Y
    n_diff = m%n_fold
    # choose in the biggest class to delete
    #  Find the biggest class
    lst_count = []
    for i in np.unique(Y):
        lst_count.append(np.where(Y==i,1,0).sum())
    ind_max = np.unique(Y)[np.argmax(lst_count)]
    lst_inds = np.where(Y==ind_max)[0]
    
    # Delete n_diff elements in the biggest class
    lst_del = np.random.choice(lst_inds,n_diff)
    X_new = np.delete(X,lst_del,0)
    Y_new = np.delete(Y,lst_del,0)
    return X_new,Y_new


# ===================== Algorithms =======================================

def FISTA_Primal(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # niter : The number of iterations
    # gamma : The hyper parameter gamma
    # eta   : The eta to calculate the projection on l1 ball
    # * isEpsilon is not used in the original file in Matlab
    # --- Output
    # w             : The projection matrix
    # mu            : The centers
    # nbGenes_fin   : The number of genes of the final step
    # loss          : The loss for each iteration
    # ====================================================================    
    """ 
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'gamma'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    gamma = param['gamma']
    tol = 1.0e-3
    a = 4 #chambolle method
    n,d = X.shape
    # === With class2indicator():
    #Y = class2indicator(YR,k)
    # === With Onehotencoder:
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    
    loss = np.zeros(niter)
    
    XtX = np.matmul(X.T,X)
    XtY = np.matmul(X.T,Y)
    
    w = np.ones((d,k))
    V = np.ones((d,k))
    t_old = 1
    
    for i in range(niter):
        
        grad_w = np.matmul(XtX,w)-XtY

        # gradient step
        w_g = w - gamma*grad_w

        #Projection on the l1 ball
        V_new = proj_l1ball(np.reshape(w_g,d*k),eta,threshold=tol)

        #V_new[np.where(np.abs(V_new)<tol)] = 0
        # Reshape back
        V_new = np.reshape(V_new,(d,k))

        # Chambolle method
        t_new  = (i+a+1)/a 
        alpha = (t_old - 1)/t_new
        w = V_new + alpha* (V_new - V)

        V = V_new
        t_old = t_new
        
        loss[i] = np.linalg.norm(Y-np.matmul(X,w),'fro')**2
    #end iteratons
    #mu = centroids(np.matmul(X,w),YR,k)
    mu = np.identity(k)
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    return w,mu,nbGenes_fin,loss

def FISTA_Primal_L21(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'gamma', 'eta'
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'gamma'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    gamma = param['gamma']
    tol = 1.0e-3
    a = 4 # prameter chambolle method
    n,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    
    loss = np.zeros(niter)
    
    XtX = np.matmul(X.T,X)
    XtY = np.matmul(X.T,Y)
    
    w = np.ones((d,k))
    V = np.ones((d,k))
    t_old = 1
    
    for i in range(niter):
        grad_w = np.matmul(XtX,w)-XtY

        # gradient step
        w_g = w - gamma*grad_w

        #Projection on the l1 ball
        V_new = proj_l21ball(w_g,eta,axis=1,threshold=tol)

        #V_new[np.where(np.abs(V_new)<tol)] = 0

        # Chambolle method
        t_new  = (i+a+1)/a 
        alpha = (t_old - 1)/t_new
        w = V_new + alpha* (V_new - V)

        V = V_new
        t_old = t_new
        
        loss[i] = np.linalg.norm(Y-np.matmul(X,w),'fro')**2
    #end iteratons
    mu = centroids(np.matmul(X,w),YR,k)
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    return w,mu,nbGenes_fin,loss

def FISTA_Primal_L12(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'gamma', 'eta'
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'gamma'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    gamma = param['gamma']
    tol = 1.0e-3
    a = 4 # prameter chambolle method
    n,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    
    loss = np.zeros(niter)
    
    XtX = np.matmul(X.T,X)
    XtY = np.matmul(X.T,Y)
    
    w = np.ones((d,k))
    V = np.ones((d,k))
    t_old = 1
    
    for i in range(niter):
        grad_w = np.matmul(XtX,w)-XtY

        # gradient step
        w_g = w - gamma*grad_w

        #Projection on the l1 ball
        V_new = proj_l12ball(w_g,eta,axis=1,threshold=tol)

        #V_new[np.where(np.abs(V_new)<tol)] = 0

        # Chambolle method
        t_new  = (i+a+1)/a 
        alpha = (t_old - 1)/t_new
        w = V_new + alpha* (V_new - V)

        V = V_new
        t_old = t_new
        
        loss[i] = np.linalg.norm(Y-np.matmul(X,w),'fro')**2
    #end iteratons
    mu = centroids(np.matmul(X,w),YR,k)
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    return w,mu,nbGenes_fin,loss

def primal_dual_L1N(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma', 'beta', 'tau2' and 'delta'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    #normY = np.linalg.norm(Y,2)
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','delta','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    delta = param['delta'] 
    tau2 = param['tau2']
    #beta = param['beta']
        
    # === END check block ===
    
    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)
        # Reshape
        V = np.reshape(V,d*k)
        
        V = proj_l1ball(V,eta)
        
        #V[np.where(np.abs(V)<0.001)] = 0
        
        # Reshape back
        w_new = np.reshape(V,(d,k))
        # no gamma here
        #w_new = w_new + gamma*(w_new - w_old) =>
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        #mu = mu_new + gamma*(mu_new - mu_old) =>
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        mu_old = mu_new
        w_old = w_new
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z


def primal_dual_L1acc(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma' 'beta', 'tau2' and 'delta' 
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    #normY = np.linalg.norm(Y,2)
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','delta','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    delta = param['delta'] 
    tau2 = param['tau2']
    #beta = param['beta']
        
    # === END check block ===

    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)
        # Reshape
        V = np.reshape(V,d*k)
        
        V = proj_l1ball(V,eta)
        
        #V[np.where(np.abs(V)<0.001)] = 0
        
        # Reshape back
        w_new = np.reshape(V,(d,k))
        # no gamma here
        #w_new = w_new + gamma*(w_new - w_old) =>
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        #mu = mu_new + gamma*(mu_new - mu_old) =>
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        # Acc part
        theta = 1/(np.sqrt(1+sigma*delta))
        Z_new = Z_new + theta*(Z_new-Z_old)
        sigma = sigma*theta
        tau = tau/theta
        tau2 = tau2/theta
        
        # Acc part End
        
        mu_old = mu_new
        w_old = w_new
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z

def primal_dual_L1OR(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma', 'beta', 'tau2','delta'
    #           and 'gamma'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    #           (The value of gamma should be between -1.0 and 1.0)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    #normY = np.linalg.norm(Y,2)
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','delta','tau2','beta','gamma'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    elif param['gamma'] <-1.0 or param['gamma']>1.0:
        raise ValueError('Invalid value for input param.gamma, it should betweenn -1.0 and 1.0')
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    delta = param['delta'] 
    tau2 = param['tau2']
    beta = param['beta']
    gamma = param['gamma']
        
    # === END check block ===
    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)
        # Reshape
        V = np.reshape(V,d*k)
        
        V = proj_l1ball(V,eta)
        
        #V[np.where(np.abs(V)<0.001)] = 0
        
        # Reshape back
        w_new = np.reshape(V,(d,k))
        # no gamma here
        #w_new = w_new + gamma*(w_new - w_old) =>
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        #mu = mu_new + gamma*(mu_new - mu_old) =>
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        mu_old = mu_new + gamma*(mu_new - mu_old)
        w_old = w_new + gamma*(w_new - w_old)
        Z_old = Z_new + gamma*(Z_new - Z_old)
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    return w,mu,nbGenes_fin,loss,Z

def primal_dual_L1W(X,YR,k,param):
    """
    # ====================================================================
    # The weighted PDL1N algo. It will update the weight every 30 iteration,
    # so for the first 30 iterations, the weight will always be ones. 
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma', 
    #            'beta', 'tau2' ,'eps' and 'delta'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    #   
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    #normY = np.linalg.norm(Y,2)
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','delta','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    delta = param['delta'] 
    tau2 = param['tau2']
    eps = param['eps']
    #beta = param['beta']       
    # === END check block ===
       
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    omega = np.ones(d*k)
    step = 30 # maybe available as input in the future
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)
        # Reshape
        V = np.reshape(V,d*k)
        
        if i >0 and i%step==0:
            omega = 1/(np.abs(V)+eps)
        V = sort_weighted_proj(V,eta,omega)
        
        #V[np.where(np.abs(V)<0.001)] = 0
        
        # Reshape back
        w_new = np.reshape(V,(d,k))
        # no gamma here
        #w_new = w_new + gamma*(w_new - w_old) =>
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        #mu = mu_new + gamma*(mu_new - mu_old) =>
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        mu_old = mu_new
        w_old = w_new
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z


def primal_dual_Nuclear(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta_star', 'tau', 'rho','sigma', 'tau2','delta'
    #           and 'gamma'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta_star', 'tau', 'rho','sigma','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta_star = param['eta_star']
    delta = param['delta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    tau2 = param['tau2']
        
    # === END check block ===

    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)

        # Nuclear constraint
        w = proj_nuclear(V,eta_star)

        w = 2*w - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)

        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        mu_old = mu_new
        w_old = w
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    mu = mu_new
    nbGenes_fin,_ = nb_Genes(w)
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z



def primal_dual_L2(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma' 'beta' and 'tau2'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
#    normY = np.linalg.norm(Y,'fro')
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    tau2 = param['tau2']
    #beta = param['beta']
        
    # === END check block ===

    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)
        # Reshape
        V = np.reshape(V,d*k)
        
        V = proj_l1ball(V,eta)
        
        #V[np.where(np.abs(V)<0.001)] = 0
        
        # Reshape back
        w_new = np.reshape(V,(d,k))
        # no gamma here
        #w_new = w_new + gamma*(w_new - w_old) =>
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        #mu = mu_new + gamma*(mu_new - mu_old) =>
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w)))
        
        # Frobenius 
        Z_new = Z / np.maximum(np.linalg.norm(Z,'fro'),1)
        
        
        mu_old = mu_new
        w_old = w_new
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),'fro')
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z


def primal_dual_L21(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma', 'beta', 'tau2' and 'delta'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    #normY = np.linalg.norm(Y,2)
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','delta','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    delta = param['delta'] 
    tau2 = param['tau2']
    #beta = param['beta']
        
    # === END check block ===
    
    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)

        w_new = proj_l21ball(V,eta,axis=1)
        w_new[np.where(np.abs(w_new)<0.001)] = 0
        
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        mu_old = mu_new
        w_old = w_new
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z

def primal_dual_L12(X,YR,k,param):
    """
    # ====================================================================
    # ---- Input
    # X     : The data
    # YR    : The label. Note that this should be an 2D array.
    # k     : The number of class
    # param : A type dict paramter which must have keys:
    #           'niter', 'eta', 'tau', 'rho','sigma', 'beta', 'tau2' and 'delta'
    #         Normally speaking: 
    #           (The default value for beta is 0.25.)
    #           (IF not given, the value of the 'tau2' will be calculated by
    #            tau2 = 0.25*(1/(np.sqrt(m)*normY)). Note that this normY is 
    #            the 2-norm of the OneHotEncode of the YR given.)
    #           (Default value of the 'delta' is 1.0)
    # --- Output
    # w             : The projection matrix of size (d,k)
    # mu            : The centers of classes
    # nbGenes_fin   : The number of genes of the final result
    # loss          : The loss for each iteration
    # Z             : The dual matrix of size (m,k)
    # =====================================================================
    """
   
    
    m,d = X.shape
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()
    #normY = np.linalg.norm(Y,2)
    
    # === Check the validness of param and the initialization of the params ===
    if type(param) is not dict:
        raise TypeError('Wrong type of input argument param',type(param))
        
    lst_params = ['niter', 'eta', 'tau', 'rho','sigma','delta','tau2','beta'] # necessary params
    if any(x not in param.keys() for x in lst_params):
        raise ValueError('Missing parameter in param.\n Need {}.\n Got {} '.format(lst_params,list(param.keys())))
    niter = param['niter']
    eta = param['eta']
    tau = param['tau']
    rho = param['rho']
    sigma = param['sigma']
    delta = param['delta'] 
    tau2 = param['tau2']
    #beta = param['beta']
        
    # === END check block ===
    
    
    # Initialization
    w_old = np.ones((d,k))
    Z_old = np.ones((m,k))
    mu_old = np.eye(k,k)
    Ik = np.eye(k,k)
    loss = np.zeros(niter)
    
    # Main Block
    for i in range(niter):
        V = w_old + tau*np.matmul(X.T,Z_old)

        w_new = proj_l12ball(V,eta,axis=1)
        #w_new[np.where(np.abs(w_new)<0.001)] = 0
        
        w = 2*w_new - w_old
        
        mu_new = (mu_old + rho*tau2*Ik-tau2*np.matmul(Y.T,Z_old)) / (1 + tau2*rho)
        
        mu = 2*mu_new - mu_old
        
        Z = (Z_old + sigma*(np.matmul(Y,mu) - np.matmul(X,w))) / (1 + sigma*delta)
        
        Z_new = np.maximum(np.minimum(Z,1),-1)
        
        mu_old = mu_new
        w_old = w_new
        Z_old = Z_new
        
        loss[i] = np.linalg.norm(np.matmul(Y,mu_new)-np.matmul(X,w_new),1) \
                    + 0.5*(np.linalg.norm(Ik-mu_new,'fro')**2)
    # End loop
    Z = Z_old
    w = w_new
    mu = mu_new
    nbGenes_fin = nb_Genes(w)[0]
    loss = loss/loss[0]
    
    return w,mu,nbGenes_fin,loss,Z


# ================================== Part 2 ====================================   
# ===================== Base Launch functions (scripts) ========================
def basic_run_eta(func_algo, func_predict,
                       X,YR, k,
                       genenames=None,
                       clusternames=None,
                       niter=30,
                       rho=1,
                       tau=4,
                       beta=0.25,
                       delta=1.0,
                       eta = None,
                       eta_star = None,
                       eps = None,
                       gamma = 1,
                       nfold=4,
                       rng = 1,          
                       showres=True,
                       keepfig = False,
                       saveres=False,
                       outputPath='../results/'):
    '''
    # =====================================================================
    # Basic function to launch the algorithm of some specific parameters.
    # - Input:
    #   - func_algo (necessary)     : The function of the algorithm
    #   - func_predict (necessary)  : The function to predict
    #   - X (necessary)             : The data
    #   - YR (necessary)            : The labels for the data
    #   - k (necessary)             : The number of the clusters
    #
    #   - genenames (optional)      : The names of the features of the data
    #                                 if not given, it will be
    #                                   ['Gene 1','Gene 2',...]
    #
    #   - clusternames (optional)   : The clusternames of the data
    #                                 if not given, it will be
    #                                   ['Class 1', 'Class 2',...]
    #
    #   - niter (optional)          : The number of iterations
    #
    #   - rho, tau, beta, delta,    : The hyper-parameters for the algo
    #     eta, gamma, etc (optional) 
    #
    #   - nfold      (optional)     : The number of the folds of the cross validation
    #
    #   - rng        (optional)     : The seed to control the random funcion
    #
    #   - showres    (optional)     : Boolean value. True if we want to show 
    #                                   the results, plot the figures etc.
    #
    #   - saveres    (optional)     : Boolean value. True to save the results
    #
    #   - outputPath (optional)     : String value. The output path.
    #
    # - Output:
    #   - mu                        : The centroids
    #   - nbm                       : Number of genes
    #   - accG                      : Global accuracy
    #   - loss                      : Loss for each iterations
    #   - W_mean                    : Mean weight matrix for all folds
    #   - timeElapsed               : Time elapsed for one fold
    #   - (And the tables)          : df_topGenes, df_normW, df_topG_normW, 
    #                                 df_topGenes_mean, df_normW_mean, 
    #                                 df_topG_normW_mean, df_acctest
    # ======================================================================
    '''
    
    np.random.seed(rng) # reproducible
    if not os.path.exists(outputPath): # make the directory if it does not exist
        os.makedirs(outputPath)
    if nfold <1:
        raise ValueError("Invalid value for 'nfold', expect integer >= 1, got{}".format(nfold))
    n,d = X.shape
    # parameter checking
    if genenames is None:
        genenames = ['Gene {}'.format(i+1) for i in range(d)]
    if clusternames is None:
        clusternames = ['Class {}'.format(i+1) for i in range(k)]

    if YR.ndim==1: # In case that OneHotEncoder get 1D array and raise a TypeError
        YR = YR.reshape(-1,1)
    Y = OneHotEncoder(categories='auto').fit_transform(YR).toarray()    
    normY = normest(Y)
    normY2 = normY**2
    
    param = {}
    param['niter'] = niter
    param['rho'] = rho
    param['tau'] = tau
    tau2 = beta*(1/(np.sqrt(n)*normY))
    param['tau2'] = tau2
    eps_tmp = 1/(1 + tau2*rho*0.25)
    sigma = 1.0/(tau + (tau2*eps_tmp*normY2))# Converge until 2.6 for L1Nel
    param['sigma'] = sigma
    param['delta'] = delta
    param['beta']= beta
    param['eta'] = eta
    param['eta_star'] = eta_star
    param['gamma'] = gamma
    if eps is not None:
        param['eps']  = eps
  
    # Parameters printing
    print('\nStarts trainning for {} with'.format(func_algo.__name__))
    print('{:>6}:{:<6}'.format('niter',niter))
    if 'fista' in func_algo.__name__.lower():
        print('{:>6}:{:<6}'.format('eta',eta))
        print('{:>6}:{:<6}'.format('gamma',delta))
    elif 'or' in func_algo.__name__.lower():
        print('{:>6}:{:<6}'.format('eta',eta))
        print('{:>6}:{:<6}'.format('rho',rho))
        print('{:>6}:{:<6}'.format('tau',tau))
        print('{:>6}:{:<6}'.format('beta',beta))
        print('{:>6}:{:<6}'.format('tau_mu',tau2))
        print('{:>6}:{:<6}'.format('sigma',sigma))
        print('{:>6}:{:<6}'.format('delta',delta))
        print('{:>6}:{:<6}'.format('gamma',delta))
    elif '_l2' in func_algo.__name__.lower():
        print('{:>6}:{:<6}'.format('eta',eta))
        print('{:>6}:{:<6}'.format('rho',rho))
        print('{:>6}:{:<6}'.format('tau',tau))
        print('{:>6}:{:<6}'.format('beta',beta))
        print('{:>6}:{:<6}'.format('tau_mu',tau2))
        print('{:>6}:{:<6}'.format('sigma',sigma))
    elif 'nuclear' in func_algo.__name__.lower():
        print('{:>6}:{:<6}'.format('eta_star',eta_star))
        print('{:>6}:{:<6}'.format('rho',rho))
        print('{:>6}:{:<6}'.format('tau',tau))
        print('{:>6}:{:<6}'.format('beta',beta))
        print('{:>6}:{:<6}'.format('tau_mu',tau2))
        print('{:>6}:{:<6}'.format('sigma',sigma))
        print('{:>6}:{:<6}'.format('delta',delta))
    else:
        print('{:>6}:{:<6}'.format('eta',eta))
        print('{:>6}:{:<6}'.format('rho',rho))
        print('{:>6}:{:<6}'.format('tau',tau))
        print('{:>6}:{:<6}'.format('beta',beta))
        print('{:>6}:{:<6}'.format('tau_mu',tau2))
        print('{:>6}:{:<6}'.format('sigma',sigma))
        print('{:>6}:{:<6}'.format('delta',delta))

    # Initialization
    nbG = np.zeros(nfold,dtype=int) # Number of genes for each fold
    accuracy_train = np.zeros((nfold,k+1))
    accuracy_test = np.zeros((nfold,k+1))
    W0 = np.zeros((d,k,nfold)) # w in each fold 
    mu0 = np.zeros((k,k,nfold))
    W_mean = np.zeros((d,k))
    loss_iter0 = np.zeros((nfold,niter)) # loss for each iteration of each fold
    # W_mean stores w for each eta, where w is the mean of W0 along its third axis 
    nbG = np.zeros(nfold) 
        
    meanclassi = np.zeros(nfold)
    if nfold>1:
        # Dropping the cells randomly if the n%d is not zero
        # For more details please see instructions in drop_cells
        # X,YR = drop_cells(X,YR,nfold)
        Y_PDS = np.zeros(YR.shape)
        kf = KFold(n_splits=nfold,random_state=rng,shuffle=True)
        for i,(train_ind, test_ind) in enumerate(kf.split(YR)):
            print('{:-<30}'.format(''))
            print('{message:^6} {f1} / {f2}'.format(message='fold',f1=i+1,f2=nfold))
            print('-> {} classification...'.format(func_algo.__name__))
            # ========== Training =========
            Xtrain = X[train_ind]
            Xtest = X[test_ind]
            Ytrain = YR[train_ind]
            Ytest = YR[test_ind]       
        
            startTime = time.perf_counter()
            w,mu,nbGenes,loss = func_algo(Xtrain,Ytrain,k,param)[0:4]
            endTime = time.perf_counter()
            timeElapsed = endTime - startTime
                
            print('-> Completed.\n-> Time Elapsed:{:.4}s'.format(timeElapsed))
            
            W0[:,:,i] = w
            mu0[:,:,i] = mu
        
            loss_iter0[i,:] = loss
            # ========== Accuracy =========
            Ytrain_pred = func_predict(Xtrain,w,mu)
            Ytest_pred = func_predict(Xtest,w,mu)
            accuracy_train[i,0],accuracy_train[i,1:k+1] = compute_accuracy(Ytrain,Ytrain_pred,k)
            accuracy_test[i,0],accuracy_test[i,1:k+1] = compute_accuracy(Ytest,Ytest_pred,k)
        
            meanclassi[i] = np.mean(accuracy_test[i,1:k+1])
            nbG[i] = nbGenes
            Y_PDS[test_ind] = Ytest_pred
            print('{:-<30}'.format(''))
            # end kfold loop 
        nbm = int(nbG.mean())
        accG = np.mean(accuracy_test[:,0],axis=0)
        Meanclass = meanclassi.mean()
        W_mean = np.mean(W0,axis=2)
        mu_mean = np.mean(mu0,axis=2)
        normfro = np.linalg.norm(w,'fro')
    
        print('Training step ends.\n')
    elif nfold ==1:
        # shuffle
        ind = np.arange(n)
        np.random.shuffle(ind)
        indx  = int(0.75*n)
        # seperate
        train_ind,test_ind = ind[:indx], ind[indx:]
        Xtrain,Xtest = X[train_ind], X[test_ind]
        Ytrain,Ytest = YR[train_ind], YR[test_ind]
        # performing algo
        startTime = time.perf_counter()
        W_mean,mu_mean,nbm,loss = func_algo(Xtrain,Ytrain,k,param)[0:4]
        endTime = time.perf_counter()
        timeElapsed = endTime - startTime
        # predict
        Ytrain_pred = func_predict(Xtrain,W_mean,mu_mean)
        Ytest_pred = func_predict(Xtest,W_mean,mu_mean)
        Y_PDS = func_predict(X,W_mean,mu_mean)
        accuracy_train[0,0],accuracy_train[0,1:k+1] = compute_accuracy(Ytrain,Ytrain_pred,k)
        accuracy_test[0,0],accuracy_test[0,1:k+1] = compute_accuracy(Ytest,Ytest_pred,k)
        accG = np.mean(accuracy_test[:,0],axis=0)
    
    # Class size
    Ctab = []
    size_class = np.zeros(k) # Size of each class (real)
    size_class_est = np.zeros(k) # Size of each class (estimated)
    for j in range(k):
        size_class[j] = (YR==(j+1)).sum()
        size_class_est[j] = (Y_PDS==(j+1)).sum()
        Ctab.append('Class {}'.format(j+1))
    
    df_szclass = pd.DataFrame(size_class,index=Ctab,columns=['Class Size'])
    df_szclass_est = pd.DataFrame(size_class_est,index=Ctab,columns=['Class Size'])
     
    # Data accuracy
    accuracy_train = np.vstack((accuracy_train,np.mean(accuracy_train,axis=0)))
    accuracy_test = np.vstack((accuracy_test,np.mean(accuracy_test,axis=0)))
    ind_df = []
    for i_fold in range(nfold):
        ind_df.append('Fold {}'.format(i_fold+1))
    ind_df.append('Mean')
    columns = ['Global']
    if clusternames is None:
        columns += Ctab
    else:
        columns += clusternames
    df_accTrain = pd.DataFrame(accuracy_train,index=ind_df,columns=columns)
    df_acctest = pd.DataFrame(accuracy_test,index=ind_df,columns=columns) 
    
    # Feature selection
    print('Selecting features from whole dataset...',end='')
    # Mean of each fold
    topGenes_mean,normW_mean = select_feature_w(W_mean,genenames)
    df_topGenes_mean = pd.DataFrame(topGenes_mean,columns=clusternames)
    df_normW_mean = pd.DataFrame(normW_mean,columns=clusternames)
    df_topG_normW_mean = merge_topGene_norm(topGenes_mean,normW_mean,clusternames)
    # All data
    if nfold>1:
        w,mu,nbGenes,loss = func_algo(X,YR,k,param)[0:4]
    elif nfold==1:
        w,mu = W_mean, mu_mean
    topGenes,normW = select_feature_w(w,genenames)
    df_topGenes = pd.DataFrame(topGenes,columns=clusternames)
    df_normW=pd.DataFrame(normW,columns=clusternames)
    df_topG_normW = merge_topGene_norm(topGenes,normW,clusternames)
    print('Completed.\n')
    
    
    # Results
    if showres == True:
        # Two heatmaps
        M_heatmap_classification = heatmap_classification(Y_PDS,YR,clusternames,rotate=60)
        M_heatmap_signature = heatmap_normW(normW_mean,clusternames,nbr_l=30,rotate=60)
        print('Size class (real):')
        print(df_szclass)
        print('\nSize class (estimated):')
        print(df_szclass_est)
        print('\nAccuracy Train')
        print(df_accTrain)
        print('\nAccuracy Test')
        print(df_acctest)
        if keepfig == False:
            plt.close("all")     
        fig_lossIter = plt.figure(figsize=(8,6))  
        plt.plot(np.arange(niter,dtype=int)+1,loss)
        msg_eta = '$\eta$:%.2f'%eta if eta is not None else ''
        msg_etaS = '$\eta*$:%.2f'%eta_star if eta_star is not None else ''
        plt.title('loss for each iteration {} {}\n ({})'.format(msg_eta,msg_etaS,func_algo.__name__),fontsize=18)
        plt.ylabel('Loss',fontsize=18)
        plt.xlabel('Iteration',fontsize=18)
        plt.xticks(np.linspace(1,niter,num=6,endpoint=True,dtype=int))
        plt.xlim(left=1,right=niter)
        plt.ylim((0,1))
        
    # Saving Result
    if saveres==True:
        # define two nametags
        nametag_eta = '_eta-%d'%eta if eta is not None else ''
        nametag_etaS = '_etaStar-%d'%eta_star if eta_star is not None else ''
        # save loss
        filename_loss = 'loss_{}_beta-{}_delta-{}{}{}_niter-{}.txt'.format(func_algo.__name__,beta,delta, nametag_eta,nametag_etaS,niter)
        np.savetxt(outputPath + filename_loss,loss)
        # define function name tag for two heatmaps
        func_tag = func_algo.__name__ + nametag_eta + nametag_etaS + '_niter-{}'.format(niter)
        # Save heatmaps
        filename_heat = '{}{}_Heatmap_of_confusion_Matrix.npy'.format(outputPath,func_tag)
        np.save(filename_heat,M_heatmap_classification)
        filename_heat = '{}{}_Heatmap_of_signature_Matrix.npy'.format(outputPath,func_tag)
        np.save(filename_heat,M_heatmap_signature)

        df_acctest.to_csv('{}{}_AccuracyTest.csv'.format(outputPath,func_tag),sep=';')
        # df_topG_normW_mean.to_csv('{}{}_TopGenesAndNormW.csv'.format(outputPath,func_tag),sep=';')

        # Other possiblilities to save
        # fig_lossIter.savefig('{}{}{}{}_niter-{}_loss_iters.png'.format(outputPath,func_algo.__name__,nametag_eta,nametag_etaS,niter))
        # All data
        #df_topGenes.to_csv('{}{}_TopGenes.csv'.format(outputPath,func_algo.__name__),sep=';')
        #df_normW.to_csv('{}{}_NormW.csv'.format(outputPath,func_algo.__name__),sep=';')
        df_topG_normW.to_csv('{}{}_TopGenesAndNormW.csv'.format(outputPath,func_tag),sep=';')
        # Mean of each fold
        #df_topGenes_mean.to_csv('{}{}_TopGenes_mean.csv'.format(outputPath,func_algo.__name__),sep=';')  
        #df_normW_mean.to_csv('{}{}_NormW_mean.csv'.format(outputPath,func_algo.__name__),sep=';') 
        #df_topG_normW_mean.to_csv('{}{}_TopGenesAndNormW_mean.csv'.format(outputPath,func_algo.__name__),sep=';')
        
    # return (mu_mean, nbm, accG, loss, W_mean, timeElapsed, \
    #         df_topGenes, df_normW, df_topG_normW,\
    #         df_acctest)
    return (mu_mean, nbm, accG, loss, W_mean, timeElapsed, \
        df_topGenes, df_normW, df_topG_normW, df_acctest,\
        df_topGenes_mean, df_normW_mean, df_topG_normW_mean)
            
   
def basic_run_tabeta(func_algo, func_predict,
                       X,YR,k,
                       genenames=None,
                       clusternames=None,
                       niter=30,
                       rho=1,
                       tau=1.5,
                       beta=0.25,
                       delta=1.0,
                       tabeta=[100,200,400],
                       gamma = 1,
                       eps = None,
                       nfold=4,
                       rng = 1,          
                       showres=True,
                       saveres=False,
                       keepfig=False,
                       outputPath='../results/'):
    """
    It shares the same input as function basic_run_eta except that the eta is 
    replaced by tabeta. It has also all the output of that of the basic_run_eta
    but it has its own 5 more output:
        nbm_etas,accG_etas,loss_iter,W_mean_etas,timeElapsed_etas
        
    Note : For now the funciton will save the output, show the figures and save the 
    results for only the last experiment, i.e. only for the last eta.
    This mechanism will be changed in the future update.
    """

    n_etas = len(tabeta) 
    n,d = X.shape
    # W_mean_etas stores w for each eta, where w is the mean of W0 along its third axis 
    W_mean_etas = np.zeros((d,k,n_etas)) 
    loss_iter = np.zeros((n_etas,niter)) # loss for each iteration of each eta
    nbm_etas = np.zeros(n_etas,dtype=int)
    accG_etas = np.zeros(n_etas)
    timeElapsed_etas = np.zeros(n_etas)
    for i,eta in enumerate(tabeta):
        if i == (n_etas-1):
            mu, nbm, accG, loss, W_mean, timeElapsed, \
            topGenes_mean, normW_mean, topGenes_normW_mean, \
            acctest =\
            basic_run_eta(func_algo, func_predict,
                                 X,YR,k,genenames,clusternames,
                                 eta=eta,
                                 niter=niter,
                                 rho=rho,tau=tau,
                                 beta=beta,
                                 delta=delta,
                                 gamma = gamma,
                                 eps = eps,
                                 nfold=nfold,
                                 rng = rng,    
                                 showres=True,
                                 saveres=saveres,
                                 keepfig = keepfig,
                                 outputPath=outputPath)
        else:
            nbm, accG, loss, W_mean, timeElapsed =\
            basic_run_eta(func_algo, func_predict,
                                 X,YR,k,genenames,clusternames,
                                 eta=eta,
                                 niter=niter,
                                 rho=rho,tau=tau,
                                 beta=beta,
                                 delta=delta,
                                 eps = eps,
                                 gamma = gamma,
                                 nfold=nfold,
                                 rng = rng,
                                 showres=False,
                                 saveres=False,
                                 outputPath=outputPath)[1:6]
        nbm_etas[i] = nbm
        accG_etas[i] = accG
        loss_iter[i,:] = loss
        W_mean_etas[:,:,i] = W_mean
        timeElapsed_etas[i] = timeElapsed
    
    if showres == True:
        file_tag = func_algo.__name__
        fig_avn = plt.figure(figsize=(8,6))  
        plt.plot(nbm_etas,accG_etas,'bo-',linewidth=3)
        plt.title('Figure: Accuracy VS Number of genes \n({})'.format(file_tag),
                  fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.xlabel('Number of genes',fontsize=16)
        plt.xlim([min(nbm_etas),max(nbm_etas)])
    
    nbm_etas = pd.DataFrame(nbm_etas,index=tabeta)
    accG_etas = pd.DataFrame(accG_etas,index=tabeta)
    loss_iter = pd.DataFrame(loss_iter,index=tabeta,columns=np.linspace(1,niter,niter,endpoint=True,dtype=int)).transpose()
    timeElapsed_etas = pd.DataFrame(timeElapsed_etas,index=tabeta)

    if saveres:
        nbm_etas.to_csv('{}{}_Num_Features.csv'.format(outputPath,func_algo.__name__),sep=';')
        accG_etas.to_csv('{}{}_Acc_tabEta.csv'.format(outputPath,func_algo.__name__),sep=';')
    return mu, nbm, accG, loss, W_mean, timeElapsed, \
            topGenes_mean, normW_mean, topGenes_normW_mean, \
            acctest, \
            nbm_etas,accG_etas,loss_iter,W_mean_etas,timeElapsed_etas

# ================================== Part 3 ====================================       
         # ===================== Exact algos ===========================
def run_FISTA_eta(X,YR,k,genenames=None,clusternames=None,
              niter=30,
              eta=500,
              beta=0.25,
              delta = 1.0,
              gamma = 1.0,
              nfold = 4,
              random_seed = 1, 
              showres= False,
              saveres=False,
              keepfig = False,
              outputPath='../results/'):    
    return basic_run_eta(FISTA_Primal,predict_FISTA,
                                 X,YR,k,genenames,clusternames,
                                 niter=niter,nfold= nfold, 
                                 beta=beta,delta=delta,eta = eta, gamma=gamma,  
                                 showres=showres,rng=random_seed,                  
                                 saveres=saveres,keepfig=keepfig,outputPath=outputPath)

def run_FISTA_L21_eta(X,YR,k,genenames=None,clusternames=None,
              niter=30,
              eta=500,
              beta=0.25,
              delta = 1.0,
              gamma = 1.0,
              nfold = 4,
              random_seed = 1, 
              showres= False,
              saveres=False,
              keepfig = False,
              outputPath='../results/'):    
    return basic_run_eta(FISTA_Primal_L21,predict_FISTA,
                                 X,YR,k,genenames,clusternames,
                                 niter=niter,nfold= nfold, 
                                 beta=beta,delta=delta,eta = eta, gamma=gamma,  
                                 showres=showres,rng=random_seed,                  
                                 saveres=saveres,keepfig=keepfig,outputPath=outputPath)
    
def run_FISTA_L12_eta(X,YR,k,genenames=None,clusternames=None,
              niter=30,
              eta=500,
              beta=0.25,
              delta = 1.0,
              gamma = 1.0,
              nfold = 4,
              random_seed = 1, 
              showres= False,
              saveres=False,
              keepfig = False,
              outputPath='../results/'):    
    return basic_run_eta(FISTA_Primal_L12,predict_FISTA,
                                 X,YR,k,genenames,clusternames,
                                 niter=niter,nfold= nfold, 
                                 beta=beta,delta=delta,eta = eta, gamma=gamma,  
                                 showres=showres,rng=random_seed,                  
                                 saveres=saveres,keepfig=keepfig,outputPath=outputPath)   
    
            
def run_primal_dual_L1N_eta(X,YR,k,genenames=None,clusternames=None,
                        niter=30,
                        rho = 1,
                        tau = 1.5,
                        beta = 0.25,
                        delta = 1.0,
                        eta = 500,
                        nfold = 4,   
                        random_seed = 1,    
                        showres = True,
                        saveres = False,
                        keepfig = False,
                        outputPath='../results/'):    
    return basic_run_eta(primal_dual_L1N,predict_L1,
                         X,YR,k,genenames,clusternames,niter=niter,
                         beta=beta,delta=delta,eta=eta,rho=rho,tau=tau,
                         nfold= nfold,rng=random_seed,
                         showres = showres,saveres=saveres,keepfig=keepfig,
                         outputPath=outputPath)
        
def run_primal_dual_L1acc_eta(X,YR,k,genenames=None,clusternames=None,
                          niter=30,
                          rho = 1,
                          tau = 1.5,
                          beta = 0.25,
                          delta = 1.0,
                          eta = 500,
                          nfold = 4,
                          random_seed = 1,         
                          showres = True,
                          keepfig = False,
                          saveres = False,
                          outputPath='../results/'): 
    return  basic_run_eta(primal_dual_L1acc,predict_L1,
                          X,YR,k,genenames,clusternames,niter=niter,
                          beta=beta,delta=delta,eta=eta,rho=rho,tau=tau,
                          nfold= nfold,rng=random_seed, 
                          showres = showres,saveres=saveres,keepfig=keepfig,
                          outputPath=outputPath)
  
def run_primal_dual_L1OR_eta(X,YR,k,genenames=None,clusternames=None,
                          niter=30,
                          rho = 1,
                          tau = 1.5,
                          beta = 0.25,
                          delta = 1.0,
                          eta = 500,
                          gamma=0.9,
                          nfold = 4,
                          random_seed = 1,          
                          showres = True,
                          saveres = False,
                          keepfig = False,
                          outputPath='../results/'): 
    return  basic_run_eta(primal_dual_L1OR,predict_L1,
                          X,YR,k,genenames,clusternames,niter=niter,
                          rho=rho,tau=tau,beta=beta,delta=delta,eta=eta,
                          gamma=gamma,nfold= nfold, rng=random_seed,  
                          showres = showres,saveres=saveres,keepfig=keepfig,
                          outputPath=outputPath)    
    
def run_primal_dual_L1W_eta(X,YR,k,genenames=None,clusternames=None,
                        niter=30,
                        rho = 1,
                        tau = 1.5,
                        beta = 0.25,
                        delta = 1.0,
                        eta = 500,
                        nfold = 4, 
                        eps = 0.1,
                        random_seed = 1,    
                        showres = True,
                        saveres = False,
                        keepfig = False,
                        outputPath='../results/'):    
    return basic_run_eta(primal_dual_L1W,predict_L1,
                         X,YR,k,genenames,clusternames,niter=niter,eps=eps,
                         beta=beta,delta=delta,eta=eta,rho=rho,tau=tau,
                         nfold= nfold,rng=random_seed,
                         showres = showres,saveres=saveres,keepfig=keepfig,
                         outputPath=outputPath)
    
def run_primal_dual_L2_eta(X,YR,k,genenames=None,clusternames=None,
                          niter=30,
                          rho = 1,
                          tau = 4,
                          beta = 8.0,
                          eta = 500,
                          nfold = 4,
                          random_seed = 1,            
                          showres = True,
                          saveres = False,
                          keepfig = False,
                          outputPath='../results/'): 
    return  basic_run_eta(primal_dual_L2,predict_Fro,
                          X,YR,k,genenames,clusternames,niter=niter,
                          rho=rho,tau=tau,beta=beta,eta = eta,
                          nfold= nfold, rng=random_seed,
                          showres = showres,saveres=saveres,keepfig=keepfig,
                          outputPath=outputPath)

def run_primal_dual_Nuclear_eta(X,YR,k,genenames=None,clusternames=None,
                        niter=30,
                        rho = 1,
                        tau = 1.5,
                        beta = 0.25,
                        delta = 1.0,
                        eta_star = 500,
                        nfold = 4,   
                        random_seed = 1,    
                        showres = True,
                        saveres = False,
                        keepfig = False,
                        outputPath='../results/'):    
    return basic_run_eta(primal_dual_Nuclear,predict_L1,
                         X,YR,k,genenames,clusternames,niter=niter,
                         beta=beta,delta=delta,eta_star=eta_star,rho=rho,tau=tau,
                         nfold= nfold,rng=random_seed,
                         showres = showres,saveres=saveres,keepfig=keepfig,
                         outputPath=outputPath)

def run_primal_dual_L21_eta(X,YR,k,genenames=None,clusternames=None,
                        niter=30,
                        rho = 1,
                        tau = 1.5,
                        beta = 0.25,
                        delta = 1.0,
                        eta = 500,
                        nfold = 4,   
                        random_seed = 1,    
                        showres = True,
                        saveres = False,
                        keepfig = False,
                        outputPath='../results/'):    
    return basic_run_eta(primal_dual_L21,predict_L1,
                         X,YR,k,genenames,clusternames,niter=niter,
                         beta=beta,delta=delta,eta=eta,rho=rho,tau=tau,
                         nfold= nfold,rng=random_seed,
                         showres = showres,saveres=saveres,keepfig=keepfig,
                         outputPath=outputPath)
    
def run_primal_dual_L12_eta(X,YR,k,genenames=None,clusternames=None,
                        niter=30,
                        rho = 1,
                        tau = 1.5,
                        beta = 0.25,
                        delta = 1.0,
                        eta = 500,
                        nfold = 4,   
                        random_seed = 1,    
                        showres = True,
                        saveres = False,
                        keepfig = False,
                        outputPath='../results/'):    
    return basic_run_eta(primal_dual_L12,predict_L1,
                         X,YR,k,genenames,clusternames,niter=niter,
                         beta=beta,delta=delta,eta=eta,rho=rho,tau=tau,
                         nfold= nfold,rng=random_seed,
                         showres = showres,saveres=saveres,keepfig=keepfig,
                         outputPath=outputPath)

# tab_eta
def run_FISTA_tabeta(X,YR,k,genenames=None,clusternames=None,
                 niter=30,
                 tabeta=[100,200,300,400,500],
                 gamma=1.0,
                 nfold=4,
                 random_seed = 1,    
                 showres = True,
                 saveres = False,
                 keepfig = False,
                 outputPath='../results/'):
    return basic_run_tabeta(FISTA_Primal,predict_FISTA,
                      X,YR,k,genenames,clusternames,
                      niter=niter,tabeta=tabeta,gamma=gamma,nfold=nfold,
                      rng=random_seed, 
                      showres=showres,saveres=saveres, keepfig=keepfig,outputPath=outputPath)
      
def run_primal_dual_L1N_tabeta(X,YR,k,genenames=None,clusternames=None,
                           niter=30,
                           rho=1,
                           tau=4,
                           beta=0.25,
                           nfold=4,
                           delta=1.0,
                           random_seed = 1,    
                           tabeta = [10,20,50,75,100,200,300],
                           showres = True,
                           keepfig = False,
                           saveres = False,
                           outputPath='../results/'):     
    return basic_run_tabeta(primal_dual_L1N,predict_L1,
                      X,YR,k,genenames=genenames,clusternames=clusternames,
                      niter=niter, tabeta=tabeta,
                      rho=rho, tau=tau, beta=beta, nfold=nfold, delta=delta,
                      rng=random_seed, 
                      showres=showres,saveres=saveres,keepfig=keepfig,outputPath=outputPath)
  
def run_primal_dual_L1acc_tabeta(X,YR,k,genenames=None,clusternames=None,
                           niter=30,
                           rho=1,
                           tau=4,
                           beta=0.25,
                           nfold=4,
                           delta=1.0,
                           random_seed = 1,    
                           tabeta = [10,20,50,75,100,200,300],
                           showres = True,
                           keepfig = False,
                           saveres = False,
                           outputPath='../results/'):     
    return basic_run_tabeta(primal_dual_L1acc,predict_L1,
                      X,YR,k,genenames=genenames,clusternames=clusternames,
                      niter=niter, tabeta=tabeta,
                      rho=rho, tau=tau, beta=beta, nfold=nfold, delta=delta,
                      rng=random_seed, 
                      showres=showres,saveres=saveres,keepfig=keepfig,outputPath=outputPath)

def run_primal_dual_L1OR_tabeta(X,YR,k,genenames=None,clusternames=None,
                           niter=30,
                           rho=1,
                           tau=4,
                           beta=0.25,
                           nfold=4,
                           delta=1.0,
                           gamma=0.9,
                           random_seed = 1,    
                           tabeta = [10,20,50,75,100,200,300],
                           showres = True,
                           saveres = False,
                           keepfig = False,
                           outputPath='../results/'):     
    return basic_run_tabeta(primal_dual_L1OR,predict_L1,
                      X,YR,k,genenames=genenames,clusternames=clusternames,
                      niter=niter, tabeta=tabeta,
                      rho=rho, tau=tau, beta=beta, nfold=nfold, delta=delta,
                      rng=random_seed, 
                      showres=showres,saveres=saveres,keepfig=keepfig,outputPath=outputPath)
    
def run_primal_dual_L2_tabeta(X,YR,k,genenames=None,clusternames=None,
                           niter=30,
                           rho=1,
                           tau=4,
                           beta=8,
                           nfold=4,
                           delta=1.0,
                           random_seed = 1,    
                           tabeta = [10,20,50,75,100,200,300],
                           showres = True,
                           saveres = False,
                           keepfig = False,
                           outputPath='../results/'):     
    return basic_run_tabeta(primal_dual_L2,predict_Fro,
                      X,YR,k,genenames=genenames,clusternames=clusternames,
                      niter=niter, tabeta=tabeta,
                      rho=rho, tau=tau, beta=beta, nfold=nfold, delta=delta,
                      rng=random_seed, 
                      showres=showres,saveres=saveres,keepfig=keepfig,outputPath=outputPath)

def run_primal_dual_L1W_tabeta(X,YR,k,genenames=None,clusternames=None,
                           niter=30,
                           rho=1,
                           tau=4,
                           beta=0.25,
                           nfold=4,
                           delta=1.0,
                           eps = 0.1,
                           random_seed = 1,    
                           tabeta = [10,20,50,75,100,200,300],
                           showres = True,
                           keepfig = False,
                           saveres = False,
                           outputPath='../results/'):     
    return basic_run_tabeta(primal_dual_L1W,predict_L1,
                      X,YR,k,genenames=genenames,clusternames=clusternames,
                      niter=niter, tabeta=tabeta,
                      rho=rho, tau=tau, beta=beta, nfold=nfold, delta=delta, eps = eps,
                      rng=random_seed, 
                      showres=showres,saveres=saveres,keepfig=keepfig,outputPath=outputPath)


def run_primal_dual_L21_tabeta(X,YR,k,genenames=None,clusternames=None,
                           niter=30,
                           rho=1,
                           tau=4,
                           beta=0.25,
                           nfold=4,
                           delta=1.0,
                           random_seed = 1,    
                           tabeta = [10,20,50,75,100,200,300],
                           showres = True,
                           keepfig = False,
                           saveres = False,
                           outputPath='../results/'):     
    return basic_run_tabeta(primal_dual_L21,predict_L1,
                      X,YR,k,genenames=genenames,clusternames=clusternames,
                      niter=niter, tabeta=tabeta,
                      rho=rho, tau=tau, beta=beta, nfold=nfold, delta=delta,
                      rng=random_seed, 
                      showres=showres,saveres=saveres,keepfig=keepfig,outputPath=outputPath)

def run_primal_dual_Nuclear_tabEtastar(X,YR,k,
                       genenames=None,
                       clusternames=None,
                       niter=30,
                       rho=1,
                       tau=1.5,
                       beta=0.25,
                       delta=1.0,
                       tabEtastar=[100,200,400],
                       gamma = 1,
                       nfold=4,
                       rng = 1,          
                       showres=True,
                       saveres=False,
                       keepfig=False,
                       outputPath='../results/'):
    """
    It shares the same input as function basic_run_eta except that the eta is 
    replaced by tabeta. It has also all the output of that of the basic_run_eta
    but it has its own 5 more output:
        nbm_etas,accG_etas,loss_iter,W_mean_etas,timeElapsed_etas
        
    Note : For now the funciton will save the output, show the figures and save the 
    results for only the last experiment, i.e. only for the last eta.
    This mechanism will be changed in the future update.
    """

    n_etas = len(tabEtastar) 
    n,d = X.shape
    # W_mean_etas stores w for each eta, where w is the mean of W0 along its third axis 
    W_mean_etas = np.zeros((d,k,n_etas)) 
    loss_iter = np.zeros((n_etas,niter)) # loss for each iteration of each eta
    nbm_etas = np.zeros(n_etas,dtype=int)
    accG_etas = np.zeros(n_etas)
    timeElapsed_etas = np.zeros(n_etas)
    for i,eta in enumerate(tabEtastar):
        if i == (n_etas-1):
            # mu, nbm, accG, loss, W_mean, timeElapsed,topGenes, normW, topGenes_normW,\
            mu, nbm, accG, loss, W_mean, timeElapsed, \
            topGenes_mean, normW_mean, topGenes_normW_mean, \
            acctest =\
            basic_run_eta(primal_dual_Nuclear, predict_L1,
                                 X,YR,k,genenames,clusternames,
                                 eta_star=eta,
                                 niter=niter,
                                 rho=rho,tau=tau,
                                 beta=beta,
                                 delta=delta,
                                 gamma = gamma,
                                 nfold=nfold,
                                 rng = rng,    
                                 showres=True,
                                 saveres=saveres,
                                 keepfig = keepfig,
                                 outputPath=outputPath)
        else:
            mu, nbm, accG, loss, W_mean, timeElapsed =\
            basic_run_eta(primal_dual_Nuclear, predict_L1,
                                 X,YR,k,genenames,clusternames,
                                 eta_star=eta,
                                 niter=niter,
                                 rho=rho,tau=tau,
                                 beta=beta,
                                 delta=delta,
                                 gamma = gamma,
                                 nfold=nfold,
                                 rng = rng,
                                 showres=False,
                                 saveres=False,
                                 outputPath=outputPath)[0:6]
        accG_etas[i] = accG
        loss_iter[i,:] = loss
        W_mean_etas[:,:,i] = W_mean
        timeElapsed_etas[i] = timeElapsed

    if showres == True:
        fig_avn = plt.figure(figsize=(8,6))  
        plt.plot(tabEtastar,accG_etas,'bo-',linewidth=3)
        plt.title('Figure: Accuracy VS eta$^{\star}$ \n(%s)'%('primal_dual_Nuclear'),
                  fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.xlabel('eta$^{\star}$',fontsize=16)
        plt.xlim([min(tabEtastar),max(tabEtastar)])
    
    accG_etas = pd.DataFrame(accG_etas,index=tabEtastar)
    loss_iter = pd.DataFrame(loss_iter,index=tabEtastar,columns=np.linspace(1,niter,niter,endpoint=True,dtype=int)).transpose()
    timeElapsed_etas = pd.DataFrame(timeElapsed_etas,index=tabEtastar)
    if saveres:
        accG_etas.to_csv('{}{}_Acc_tabEta.csv'.format(outputPath,'primal_dual_Nuclear'),sep=';')
    return mu, nbm, accG, loss, W_mean, timeElapsed, \
            topGenes_mean, normW_mean, topGenes_normW_mean, \
            acctest, \
            accG_etas,loss_iter,W_mean_etas,timeElapsed_etas


#=============================================================================    
    
if __name__ == '__main__':

#    script_primal_Fista()
    print('This is just a storage file for functions. So... nothing happened.')
    print(help_info)

    
    
