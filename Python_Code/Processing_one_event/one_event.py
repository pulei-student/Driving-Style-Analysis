
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
    datas=[]
    data = pd.read_csv(path)
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



datas_orig = loading_data('car-following/10106/2.csv')

iter = 200
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
print(np.shape(datas[0]))



# %%

######################
###hdp-hsmm process###
######################
def hdp_hsmm(datas,iter):
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


hdp_hsmm_model,hdp_hsmm_logs= hdp_hsmm(datas,iter)


# %%
#########################
#  hdp-hmm process  #
#########################
def hdp_hmm(datas,iter):
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
    for data in datas:
        posteriormodel.add_data(data)

    logs = []
    for idx in progprint_xrange(iter):
        posteriormodel.resample_model()
        log = posteriormodel.log_likelihood()
        logs.append(log)

    return posteriormodel,logs
hdp_hmm_model,hdp_hmm_logs = hdp_hmm(datas,iter)



# %%
def hdp_s_hmm(datas,iter):
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
        if abs(log)<1e-3 or (idx>1 and abs(log-logs[idx-2])<1e-6):
            progprint_xrange(idx+1)
            break

    return posteriormodel,logs
hdp_s_hmm_model,hdp_s_hmm_logs = hdp_s_hmm(datas,iter)


# %%
#########################waiting for downing
def plotting_logs(all_logs):
    plt.figure(figsize=(6,5))
    x = np.arange(len(all_logs[0]))
    plt.plot(x,all_logs[0],color='red',label='HDP-HMM')
    plt.plot(x,all_logs[1],color='blue',label='sticky HDP-HMM')
    plt.plot(x,all_logs[2],color='green',label='HDP-HSMM')
    plt.title('对数似然',fontsize=20,fontproperties=zhfont1)
    plt.xlabel('重复采样次数',fontsize=18,fontproperties=zhfont1)
    plt.ylabel('对数似然值',fontsize=18,fontproperties=zhfont1)
    plt.legend(fontsize=17)
    plt.tick_params(labelsize=15)
plotting_logs([hdp_hmm_logs,hdp_s_hmm_logs,hdp_hsmm_logs])


# %%
###############################
###drawing state and feature###
###############################
from matplotlib.lines import Line2D
def show_figures(model):
    plotting_state(model) #written in models.py
    plotting_feature(model) #defined in this file
    plt.show()

def plotting_state(model,fig=None,plot_slice=slice(None),update=False,draw=True):
    #update = upfeature_ax, date and (fig is not None)
    fig = fig if fig else model.make_figure()
    stateseq_axs = _get_state_axes(model,fig)
    assert len(stateseq_axs) == len(model.states_list)
    sp2_artists = \
        [artist for s,ax in zip(model.states_list,stateseq_axs)
            for artist in plot_stateseq(model,s,ax,plot_slice,update=update,draw=False)]
        
    if draw: plt.draw()
    plt.title('时间序列划分',fontproperties=zhfont1)
    return sp2_artists

def plotting_state_plain(model,fig=None,plot_slice=slice(None),update=False,draw=True):
    #update = upfeature_ax, date and (fig is not None)
    fig = fig if fig else model.make_figure()
    stateseq_axs = _get_state_axes(model,fig)
    assert len(stateseq_axs) == len(model.states_list)
    sp3_artists = \
        [artist for s,ax in zip(model.states_list,stateseq_axs)
            for artist in plot_stateseq_plain(model,s,ax,plot_slice,update=update,draw=False)]
        
    if draw: plt.draw()
    plt.title('各参数的时间序列',fontproperties=zhfont1)
    return sp3_artists

def _get_state_axes(model,fig):
    #sz = self._fig_sz

    if hasattr(fig,'_stateseq_axs'):
        return fig._stateseq_axs
    else:
        if len(model.states_list) <= 2:
            gs = GridSpec(4,1)
            #feature_ax = plt.subplot(gs[:sz,:])
            stateseq_axs = [plt.subplot(gs[idx]) for idx in range(len(model.states_list))]
        else:
            gs = GridSpec(1,2)
            sgs = GridSpecFromSubplotSpec(len(model.states_list),1,subplot_spec=gs[1])

            #feature_ax = plt.subplot(gs[0])
            stateseq_axs = [plt.subplot(sgs[idx]) for idx in range(len(model.states_list))]

        for ax in stateseq_axs:
            ax.grid('off')
            
        fig._stateseq_axs = stateseq_axs
        return stateseq_axs

def plot_stateseq(model,s,ax=None,plot_slice=slice(None),update=False,draw=True):
    s = model.states_list[s] if isinstance(s,int) else s
    ax = ax if ax else plt.gca()
    state_colors = model._get_colors(scalars=True)

    _plot_stateseq_pcolor(model,s,ax,state_colors,plot_slice,update)
    data_values_artist = _plot_stateseq_data_values(model,s,ax,state_colors,plot_slice,update)

    if draw: plt.draw()

    return [data_values_artist]

def plot_stateseq_plain(model,s,ax=None,plot_slice=slice(None),update=False,draw=True):
    s = model.states_list[s] if isinstance(s,int) else s
    ax = ax if ax else plt.gca()
    state_colors = model._get_colors(scalars=True)
    ax.set_xlim((0,len(model.datas[0])))
    _plot_stateseq_pcolor_plain(model,s,ax,state_colors,plot_slice,update)
    data_values_artist = _plot_stateseq_data_values(model,s,ax,state_colors,plot_slice,update)

    plt.grid(False)
    plt.show()

    return [data_values_artist]


def _plot_stateseq_pcolor(model,s,ax=None,state_colors=None,
        plot_slice=slice(None),update=False,color_method=None):
    from pyhsmm.util.general import rle

    s = model.states_list[s] if isinstance(s,int) else s
    ax = ax if ax else plt.gca()
    state_colors = state_colors if state_colors \
            else model._get_colors(scalars=True,color_method=color_method)

    data = s.data[plot_slice].T
    real_data = np.array(list(data[idx]*data_info[idx][1]+data_info[idx][0] for idx in range(len(data_info))))
    stateseq = s.stateseq[plot_slice]

    stateseq_norep, durations = rle(stateseq)
    datamin, datamax = np.floor(real_data[0].min()/5)*5, np.ceil(real_data[0].max()/5)*5 
    datamin_,datamax_ = np.floor(min(real_data[1].min(),real_data[2].min())),np.ceil(max(real_data[1].max(),real_data[2].max()))

    x, y = np.hstack((0,durations.cumsum())), np.array([datamin,datamax])
    C = np.atleast_2d([state_colors[state] for state in stateseq_norep])

    s._pcolor_im = ax.pcolormesh(x,y,C,vmin=0,vmax=1,alpha=0.3)
    
    ax_ = ax.twinx()
    
    ax.set_xlim((0,len(stateseq)))
    ax.set_ylim((datamin,datamax))
    ax.set_xlabel('跟车时间/s',loc='right',fontproperties=zhfont1)
    ax.set_xticks(list(np.arange(0,len(data.T),100)))
    ax.set_xticklabels(list(np.arange(0,len(data.T)/100)*10))
    ax.set_yticks(list(np.arange(datamin,datamax+1,10)))
    ax.set_yticklabels(list(np.arange(datamin,datamax+1,10)))
    ax.set_ylabel('相对距离/[m]',fontproperties=zhfont1)
 
    ax_.set_ylim((datamin_,datamax_))
    ax_.set_yticks(list(np.arange(datamin_,datamax_+1,2)))
    ax_.set_yticklabels(list(np.arange(datamin_,datamax_+1,2)))
    ax_.set_ylabel('相对速度[m/s]，加速度'+r'$[m/s^2]$',fontproperties=zhfont1)

def _plot_stateseq_pcolor_plain(model,s,ax=None,state_colors=None,
        plot_slice=slice(None),update=False,color_method=None):
    from pyhsmm.util.general import rle

    s = model.states_list[s] if isinstance(s,int) else s
    ax = ax if ax else plt.gca()
    
    state_colors = state_colors if state_colors \
            else model._get_colors(scalars=True,color_method=color_method)

    data = s.data[plot_slice].T
    real_data = np.array(list(data[idx]*data_info[idx][1]+data_info[idx][0] for idx in range(len(data_info))))
 
    datamin, datamax = np.floor(real_data[0].min()/5)*5, np.ceil(real_data[0].max()/5)*5 
    datamin_,datamax_ = np.floor(real_data[1].min()),np.ceil(real_data[1].max())
    datamin__,datamax__=np.floor(real_data[2].min()),np.ceil(real_data[2].max())

    plt.xlim(0,len(data[0]))
    ax.set_xlim((0,len(data[0])))
    ax.set_ylim((datamin,datamax))
    ax.set_xlabel('跟车时间/s',loc='right',fontproperties=zhfont1)
    ax.set_xticks(list(np.arange(0,len(data.T),100)))
    ax.set_xticklabels(list(np.arange(0,len(data.T)/100)*10))
    ax.set_yticks(list(np.arange(datamin,datamax+1,10)))
    ax.set_yticklabels(list(np.arange(datamin,datamax+1,10)))
    ax.set_ylabel('相对距离/[m]',fontproperties=zhfont1)
    
    ax_ = ax.twinx()
    ax_.set_xlim((0,len(data[0])))
    ax_.set_ylim((datamin_,datamax_))
    ax_.set_yticks(list(np.arange(datamin_,datamax_+1,2)))
    ax_.set_yticklabels(list(np.arange(datamin_,datamax_+1,2)))
    ax_.set_ylabel('相对速度[m/s],加速度'+r'$[m/s^2]$',fontproperties=zhfont1)

    

def _plot_stateseq_data_values(model,s,ax,state_colors,plot_slice,update):
    from matplotlib.collections import LineCollection
    from pyhsmm.util.general import AR_striding, rle

    data = s.data[plot_slice]
    stateseq = s.stateseq[plot_slice]
    real_data = np.array(list(data.T[idx]*data_info[idx][1]+data_info[idx][0] for idx in range(len(data_info))))

    colorseq = np.tile(np.array([state_colors[state] for state in stateseq[:-1]]),data.shape[1])
    
    datamin, datamax = np.floor(real_data[0].min()/5)*5, np.ceil(real_data[0].max()/5)*5 
    datamin_,datamax_ = np.floor(min(real_data[1].min(),real_data[2].min())),np.ceil(max(real_data[1].max(),real_data[2].max()))
    draw_data = list()
    draw_data.append(real_data[0])
    mid,whole = (datamin+datamax)/2.,(datamax-datamin)/2.
    mid_,whole_ = (datamin_+datamax_)/2.,(datamax_-datamin_)/2.
    draw_data.append((real_data[1]-mid_)/whole_*whole+mid)
    draw_data.append((real_data[2]-mid_)/whole_*whole+mid)
    draw_data = np.array(draw_data)
    
    if update and hasattr(s,'_data_lc'):
        s._data_lc.set_array(colorseq)
    else:
        ts = np.arange(len(stateseq))
        segments = np.vstack(
            [AR_striding(np.hstack((ts[:,None], scalarseq[:,None])),1).reshape(-1,2,2)
                for scalarseq in draw_data])
        lc = s._data_lc = LineCollection(segments)
        one_list = np.ones(len(stateseq))
        z =[500,100,0]
        new_color = np.array(list(one_list*200)+list(one_list*100)+list(one_list*0))
        lc.set_array(new_color)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.set_xlim(0,len(stateseq))
        ax.autoscale()
        proxies = [make_proxy(item,lc) for item in z]
        ax.legend(proxies,['相对距离','相对速度','加速度'],prop=zhfont1,bbox_to_anchor=(0.8,-0.2),ncol=3)
    return s._data_lc

def make_proxy(zvalue, scalar_mappable, **kwargs):
    color = scalar_mappable.cmap(zvalue)
    return Line2D([0, 1], [0, 1], color=color, **kwargs)

def plotting_feature(model,fig=None,plot_slice=slice(None),update=False,draw=True):
    update = update and (fig is not None)
    sp1_artists = _plot_3d_data_scatter(model,state_colors=model._get_colors(scalars=True))
    plt.draw()

    return sp1_artists


def _plot_3d_data_scatter(model,ax=None,state_colors=None,plot_slice=slice(None),update=False):
    data = np.array(model.datas).T
    real_data = np.array(list(data[idx]*data_info[idx][1]+data_info[idx][0] for idx in range(len(data_info)))).reshape(len(data),len(data[0]))
    datamin = [np.floor(real_data[0].min()/5)*5,np.floor(real_data[1].min()),np.floor(real_data[2].min())]
    datamax = [np.ceil(real_data[0].max()/5)*5,np.ceil(real_data[1].max()),np.ceil(real_data[2].max())]
    
    fig = plt.figure(figsize=(12,12))
    fontsize=20
    #plt.suptitle('按照参数种类表示',fontsize=23)
    ax1 = fig.add_subplot(221,projection='3d')
    plt.tick_params(labelsize=fontsize-10)
    ax1.set_xlabel('相对距离[m]',fontsize=fontsize,fontproperties=zhfont1)
    ax1.set_xlim((datamin[0],datamax[0]))
    ax1.set_xticks(list(np.arange(datamin[0],datamax[0]+1,10)))
    
    ax1.set_ylabel('相对速度[m/s]',fontsize=fontsize,fontproperties=zhfont1)
    ax1.set_ylim((datamin[1],datamax[1]))
    ax1.set_yticks(list(np.arange(datamin[1],datamax[1])))
    
    ax1.set_zlabel('加速度'+r'$[m/s^2]$',fontsize=fontsize,fontproperties=zhfont1)
    ax1.set_zlim((datamin[2],datamax[2]))
    ax1.set_zticks(list(np.arange(datamin[2],datamax[2],0.5)))

    state_colors = state_colors if state_colors \
            else model._get_colors(scalars=True)
    artists = []
    

    for s, data in zip(model.states_list,model.datas):
        data = data[plot_slice]
        colorseq = [state_colors[state] for state in s.stateseq[plot_slice]]
        if update and hasattr(s,'_data_scatter'):
            s._data_scatter.set_offsets(data[:,:2])
            s._data_scatter.set_color(colorseq)
        else:
            s._data_scatter = ax1.scatter3D(real_data[0],real_data[1],real_data[2],c=colorseq,s=120)
        artists.append(s._data_scatter)

    ax2 = plt.subplot(2,2,2)
    plt.tick_params(labelsize=fontsize-5)
    artists.append(ax2.scatter(real_data[0],real_data[1],c=colorseq,s=120))
    ax2.set_xlabel('相对距离[m]',fontsize=fontsize,fontproperties=zhfont1)
    ax2.set_xlim((datamin[0],datamax[0]))
    ax2.set_xticks(list(np.arange(datamin[0],datamax[0]+1,10)))
    ax2.set_ylabel('相对速度[m/s]',fontsize=fontsize,fontproperties=zhfont1)
    ax2.set_ylim((datamin[1],datamax[1]))
    ax2.set_yticks(list(np.arange(datamin[1],datamax[1]+0.01)))
    plt.grid(True)

    ax3 = plt.subplot(2,2,3)
    plt.tick_params(labelsize=fontsize-5)
    artists.append(ax3.scatter(real_data[2],real_data[1],c=colorseq,s=120))
    ax3.set_xlabel('加速度'+r'$[m/s^2]$',fontsize=fontsize,fontproperties=zhfont1)
    ax3.set_xlim((datamin[2],datamax[2]))
    ax3.set_xticks(list(np.arange(datamin[2],datamax[2]+0.01,0.5)))
    ax3.set_ylabel('相对速度[m/s]',fontsize=fontsize,fontproperties=zhfont1)
    ax3.set_ylim((datamin[1],datamax[1]))
    ax3.set_yticks(list(np.arange(datamin[1],datamax[1]+0.01)))
    plt.grid(True)

    ax4 = plt.subplot(2,2,4)
    plt.tick_params(labelsize=fontsize-5)
    artists.append(ax4.scatter(real_data[0],real_data[2],c=colorseq,s=120))
    ax4.set_xlabel('相对距离[m]',fontsize=fontsize,fontproperties=zhfont1)
    ax4.set_xlim((datamin[0],datamax[0]))
    ax4.set_xticks(list(np.arange(datamin[0],datamax[0]+1,10)))
    ax4.set_ylabel('加速度'+r'$[m/s^2]$',fontsize=fontsize,fontproperties=zhfont1)
    ax4.set_ylim((datamin[2],datamax[2]))
    ax4.set_yticks(list(np.arange(datamin[2],datamax[2]+0.01,0.5)))
    plt.grid(True)

    plt.subplots_adjust(wspace=0.4,hspace=0.3)
    return artists
plotting_state_plain(hdp_hmm_model)
show_figures(hdp_hmm_model)
show_figures(hdp_s_hmm_model)
show_figures(hdp_hsmm_model)


# %%
class get_states_list():
    def __init__(self,seq):        
        self.stateseq=seq

class k_means():
    def __init__(self,model,k=1):
        self.get_state = get_states_list
        self.data_info = data_info
        self.make_figure=model.make_figure
        self.old_datas = model.datas
        self.stateseqs = model.stateseqs
        #self.states_list = model.states_list[0]
        self.state_colors = model._get_colors(scalars=True)
        self._get_colors = model._get_colors
        self.datas = self.get_k_data(k)
        #self.groups,self.new_seq = self.seg_into_group(self.stateseq)
        ###get num and time of state
        stating_counter = Counter(self.new_seqs)
        self.num_state = len(list(stating_counter))

        #self.stateseqs = get_seq(stateseq)

    def get_k_data(self,k=1):
        events_groups,self.new_seqs = self.seg_into_group(self.stateseqs)
        #print(type(self.new_seqs[0]))
        #print(events_groups)
        new_datas = [self.get_k_means(group_data,k) for group_data in events_groups]
        self.states_list=[self.get_state(seq=self.new_seqs)]
        self.time_state = [self.get_time_state(group_data) for group_data in events_groups]
        #print('get_k_data done')
        return new_datas

    def get_time_state(self,groups):
        duration_time = []
        #print('get_time_state done')
        for group in groups:
            #print(group,'group')
            duration_time.append(len(group))
        return duration_time

    def seg_into_group(self,seqs):
        datas = self.old_datas
        #data = data.T
        events_groups=[]
        new_seqs=[]
        
        assert len(datas)==len(seqs)
        for i in range(len(datas)):
            #print(len(data))
            data=datas[i]
            seq=seqs[i]
            groups = list()
            group = [data[0]]
            new_seq = list()
            for idx in range(1,len(data)):    #############
                if idx==len(data)-1:
                    groups.append(np.array(group))
                    new_seq.append(seq[idx])
                
                if seq[idx-1]==seq[idx]:
                    group.append(data[idx])
                else:
                    groups.append(np.array(group))
                    group=[data[idx]]
                    new_seq.append(seq[idx-1])
            events_groups.append(groups)
            new_seqs=new_seqs+new_seq
        return events_groups,np.array(new_seqs)

    def get_k_means(self,groups,k=1):
        from sklearn.cluster import KMeans
        new_data = []
        for group in groups:
            kmodel = KMeans(n_clusters=k)
            kmodel.fit(group)
            new_data.append(kmodel.cluster_centers_[0])
        return new_data


hmm_k =k_means(hdp_hmm_model)
print('共有',str(hmm_k.num_state),'种驾驶模式')
print('分了',str(len(hmm_k.new_seqs)),'个驾驶片段')
print('各个驾驶片段行驶的时间为',str(hmm_k.time_state),'*0.1s')


shmm_k =k_means(hdp_s_hmm_model)
print('共有',str(shmm_k.num_state),'种驾驶模式')
print('分了',str(len(shmm_k.new_seqs)),'个驾驶片段')
print('各个驾驶片段行驶的时间为',str(shmm_k.time_state),'*0.1s')


hsmm_k =k_means(hdp_hsmm_model)
print('共有',str(hsmm_k.num_state),'种驾驶模式')
print('分了',str(len(hsmm_k.new_seqs)),'个驾驶片段')
print('各个驾驶片段行驶的时间为',str(hsmm_k.time_state),'*0.1s')



# %%
plotting_feature(hsmm_k)

# %%
def get_seg_d(data):
    data = np.array(data).T
    threshold = [[59.26,20.02,5.00],[-1.19,-0.2,0.25,1.23],[-0.20,-0.06,0.07,0.20]]
    real_data = np.array(list(data[idx]*data_info[idx][1]+data_info[idx][0] for idx in range(len(data_info)))).reshape(len(data),len(data[0]))
    final_data = real_data.T
    LD_data,ND_data,CD_data = [],[],[]
    for point in final_data:
        if point[0]>threshold[0][0]:
            LD_data.append(point)
        elif point[0]>=threshold[0][1]:
            ND_data.append(point)
        elif point[0]>=threshold[0][2]:
            CD_data.append(point)
        else:
            print('Data has a wrong delta_d')
    return [LD_data,ND_data,CD_data],threshold
#range_datas=[LD_data,ND_data,CD_data]
def get_seg_v_a(range_datas,threshold):
    analyze_data = []
    for range_data in range_datas:
        group_data = np.ones(shape=(len(range_data),2))*20
        for idx in range(len(range_data)):
            if range_data[idx][1]<threshold[1][0]:
                group_data[idx][1] = -2
            elif range_data[idx][1]<threshold[1][1]:
                group_data[idx][1]= -1
            elif range_data[idx][1]<threshold[1][2]:
                group_data[idx][1]=0
            elif range_data[idx][1]<threshold[1][3]:
                group_data[idx][1]= 1
            else:
                group_data[idx][1] = 2
            
            if range_data[idx][2]<threshold[2][0]:
                group_data[idx][0] = -2
            elif range_data[idx][2]<threshold[2][1]:
                group_data[idx][0]= -1
            elif range_data[idx][2]<threshold[2][2]:
                group_data[idx][0]=0
            elif range_data[idx][2]<threshold[2][3]:
                group_data[idx][0]= 1
            else:
                group_data[idx][0] = 2

        analyze_data.append(group_data)
    return analyze_data


# %%
def get_prob(analyze_data):
    all_counts = []
    for idx in range(len(analyze_data)):
        counts = np.zeros((5,5)) 
        for h in range(len(analyze_data[idx])):
            for i in range(5):
                for j in range(5):
                    if (analyze_data[idx][h] == [i-2,j-2]).all():
                        counts[i][j] = counts[i][j]+1
        if not len(analyze_data[idx]):
            pass
        else:
            counts = counts/float(len(analyze_data[idx]))
        all_counts.append(counts)
    return all_counts


all_counts=np.zeros((3,5,5))
for data in hsmm_k.datas:
    range_datas,thre = get_seg_d(data)
    anylyze_data=get_seg_v_a(range_datas,thre)
    all_counts = all_counts+np.array(get_prob(anylyze_data))
    


# %%

def plotting_style(all_counts):
    Y = ['急减','缓减','匀速','缓加','急加']
    X = ['快近','渐近','维持','渐远','快远']
    fig = plt.figure(figsize=(12,12))
    axes = []
    img = []
    sizes=20
    titles = ['远距离','中距离','短距离']
    for idx in range(len(all_counts)): 
        axes.append(fig.add_subplot(2,2,idx+1))
        axes[-1].set_ylabel('加速度'+r'$a_x$',size=sizes,fontproperties=zhfont1)
        axes[-1].set_xticks(np.linspace(0.5,4.5,5,endpoint=True))
        axes[-1].set_xticklabels(X,fontproperties=zhfont1,size=sizes-2)
        axes[-1].set_xlabel('相对速度'+r'$\Delta$v',size=sizes,fontproperties=zhfont1)
        axes[-1].set_yticks(np.linspace(0.5,4.5,5,endpoint=True))
        axes[-1].set_yticklabels(Y,fontproperties=zhfont1,size=sizes-2)
        axes[-1].set_title(titles[idx],fontproperties=zhfont1,size=sizes)
        plt.subplots_adjust(wspace=0.4,hspace=0.3)
        img.append(axes[-1].pcolormesh(all_counts[idx],cmap = mpl.cm.Spectral_r))
        
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right",size='5%',pad=0.05)
        #print(np.linspace(0.5,4.5,5,endpoint=True))
        norm = mpl.colors.Normalize(vmin=0,vmax=all_counts[idx].max())
        cmap = mpl.cm.Spectral_r
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)

all_counts=all_counts/len(hsmm_k.datas)
plotting_style(all_counts)
#plt.xla


# %%

# %%



