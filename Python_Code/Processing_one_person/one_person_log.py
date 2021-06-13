
# %%
import numpy as np
from matplotlib import pyplot as plt
import copy,os

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from matplotlib import font_manager
import matplotlib as mpl
zhfont1 = font_manager.FontProperties(fname='SimHei.ttf')
from mpl_toolkits.axes_grid1 import make_axes_locatable

from collections import Counter


print("this file contains hdp-hsmm in action")


# %%
def loading_data(path):
    import pandas as pd
    import os
    d=[]
    v=[]
    a=[]
    datas=[]
    dirlist = os.listdir(path)
    for idx in range(len(dirlist)):
        data = pd.read_csv(os.path.join(path,dirlist[idx]))
    #print(data.head())
    #print(data.tail(3))
    #print(data.columns)
        delta_d = np.array(data['range'])
        delta_v = np.array(data['rangerate'])
        acc = np.array(data['ax'])
        datas.append([delta_d,delta_v,acc])
    #print(acc.max())
    #print(np.shape(new_data))   (564,3)
    return datas
    ###############data is np.array



datas_orig = loading_data('car-following/10106')

iter = 10
kappa_0=0.75
init=2.


# %%
####################
###normalize data###
####################
def initializing(datas):
    d=[]
    v=[]
    a=[]
    for data in datas:
        data=np.array(data)
        print(np.shape(data))
        d=d+list(data[0].T)
        v=v+list(data[1].T)
        a=a+list(data[2].T)
    data_info=[[np.mean(d),np.std(d)],[np.mean(v),np.std(v)],[np.mean(a),np.std(a)]]
    datas_new=[]
    for data in datas:
        data = np.array(data)
        d_new = (data[0]-data_info[0][0])/data_info[0][1]
        v_new = (data[1]-data_info[1][0])/data_info[1][1]
        a_new = (data[2]-data_info[2][0])/data_info[2][1]
        datas_new.append(np.array([d_new.T,v_new.T,a_new.T]).T)
    return datas_new,data_info
datas,data_info = initializing(datas_orig)


# %%

######################
###hdp-hsmm process###
######################
def hdp_hsmm(datas):
    #limit truncation level
    Nmax = 25
    #hyperparameters
    obs_dim = datas[0].shape[1]
    #print(obs_dim) (3)
    d=[]
    v=[]
    a=[]
    for data in datas:
        d=d+list(data.T[0])
        v=v+list(data.T[1])
        a=a+list(data.T[2])
    data_all=np.array([np.array(d),np.array(v),np.array(a)])
    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.cov(data_all),
                    'kappa_0':kappa_0,
                    'nu_0':obs_dim+2}
    dur_hypparams = {'alpha_0':2*40,
                    'beta_0':2}
    #generate distributions
    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    dur_distns = [pyhsmm.distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]
    
    #defining model
    posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
        alpha_a_0=1.,alpha_b_0=1.,
        gamma_a_0=1.,gamma_b_0=1.,
        init_state_concentration=init,
        obs_distns=obs_distns,
        dur_distns=dur_distns
    )
    #return posteriormodel

#def add_data(model,data,trunc=80):
    for i in range(len(datas)):
        posteriormodel.add_data(datas[i],trunc=80)
    #return model


#running model
#def running_model(model,iteration=5):
    logs = []
    for idx in progprint_xrange(iter):  ###defining iteration number,normal 150
        posteriormodel.resample_model()
        log = posteriormodel.log_likelihood()
        logs.append(log)
    #plotting_logs(logs)
  
    return posteriormodel,logs

###drawing log figures


# %%
#########################
#  hdp-hmm process  #
#########################
def hdp_hmm(datas):
    # Set the weak limit truncation level
    Nmax = 25
    d=[]
    v=[]
    a=[]
    for data in datas:
        d=d+list(data.T[0])
        v=v+list(data.T[1])
        a=a+list(data.T[2])
    data_all=np.array([np.array(d),np.array(v),np.array(a)])

    obs_dim = datas[0].shape[1]
    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.cov(data_all),
                    'kappa_0':kappa_0,
                    'nu_0':obs_dim+2}

    ### HDP-HMM without the sticky bias
    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    posteriormodel = pyhsmm.models.WeakLimitHDPHMM(alpha_a_0=1.,alpha_b_0=1.,gamma_a_0=1.,gamma_b_0=1.,        init_state_concentration=init,            obs_distns=obs_distns)
    
    return posteriormodel



# %%
def hdp_s_hmm(datas):
    # Set the weak limit truncation level
    Nmax = 25
    d=[]
    v=[]
    a=[]
    for data in datas:
        d=d+list(data.T[0])
        v=v+list(data.T[1])
        a=a+list(data.T[2])
    data_all=np.array([np.array(d),np.array(v),np.array(a)])
    # and some hyperparameters
    obs_dim = datas[0].shape[1]
    obs_hypparams = {'mu_0':np.zeros(obs_dim),
                    'sigma_0':np.cov(data_all),
                    'kappa_0':kappa_0,
                    'nu_0':obs_dim+2}
    
    from pybasicbayes.distributions.multinomial import GammaCompoundDirichlet
    kappa = GammaCompoundDirichlet(Nmax,100,1).concentration
    obs_distns = [pyhsmm.distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
    posteriormodel = pyhsmm.models.WeakLimitStickyHDPHMM(
            kappa=float(kappa),
            alpha_a_0=1.,alpha_b_0=1.,
        gamma_a_0=1.,gamma_b_0=1.,
        init_state_concentration=init,
            obs_distns=obs_distns)
    
    for data in datas:
        posteriormodel.add_data(data)

    logs=[]
    for idx in progprint_xrange(iter):
        posteriormodel.resample_model()
        log=posteriormodel.log_likelihood()
        logs.append(log)

    return posteriormodel,logs


# %%
#########################waiting for downing
def plotting_logs(all_logs):
    x = np.arange(len(all_logs))
    #plt.plot(x,all_logs[0],color='red',label='HDP-HMM')
    #plt.plot(x,all_logs[1],color='blue',label='sticky HDP-HMM')
    plt.plot(x,all_logs,color='green',label='HDP-HSMM')
    plt.title('对数似然',fontsize=15,fontproperties=zhfont1)
    plt.xlabel('重复采样次数',fontsize=14,fontproperties=zhfont1)
    plt.ylabel('对数似然值',fontsize=14,fontproperties=zhfont1)
    plt.legend(fontsize=12)
    plt.tick_params(labelsize=12)



# %%
line=np.floor(len(datas)/10.)
num=0
data_seps=list()
data_sep=list()
for i in range(len(datas)):
    if num<line:
        data_sep.append(datas[i])
    else:
        num=0
        data_seps.append(data_sep)
        data_sep=list()
    num=num+1



# %%
for i in range(10):
    train_data=list()
    data_seping=list(data_seps)
    test_data=data_seping.pop(i)
    train_data=data_seping
    for sep in data_seping:
        train_data=train_data+[data for data in sep]

print(np.shape(train_data))
print(np.shape(train_data[0]))
iteration=180
hmm_model=hdp_hmm(datas)
shmm_model=hdp_s_hmm(train_data)
logs=[]
# for i in progprint_xrange(iteration):
#     j=int(i//(iteration/9))
#     if i==0 or (not j==j_):
#         for data in train_data[j]:
#             hmm_model.add_data(data)
#     hmm_model.resample_model()
#     log=hmm_model.log_likelihood()
#     logs.append(log)
#     j_=j
plotting_logs(logs)

# %%
#pre_log=hsmm_model.predictive_likelihoods(test_data,forecast_horizons=range(1,5))




# %%



