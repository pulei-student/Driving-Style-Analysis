# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
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


dir='10151'
datas_orig = loading_data(os.path.join('car-following',dir))

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
        #print(np.shape(data))
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
print(np.shape(datas[1]))


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
        posteriormodel.add_data(datas[i],trunc=30)
    #return model


#running model
#def running_model(model,iteration=5):
    logs = []
    for idx in progprint_xrange(iter):  ###defining iteration number,normal 150
        posteriormodel.resample_model()
        log = posteriormodel.log_likelihood()
        logs.append(log)
        if abs(log)<1e-3 or (idx>1 and abs(log-logs[idx-2])<1e-6):
            progprint_xrange(idx+1)
            break
    #plotting_logs(logs)


    
    return posteriormodel,logs

###drawing log figures


hdp_hsmm_model,hdp_hsmm_logs= hdp_hsmm(datas,iter)

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
plotting_logs(hdp_hsmm_logs)


# %%
class get_states_list():
    def __init__(self,seq):        
        self.stateseq=seq

class k_means():
    def __init__(self,model,k=1):
        self.get_state = get_states_list
        self.data_info = data_info
        self.datas = model.datas
        self.stateseqs = model.stateseqs
        #self.states_list = model.states_list[0]
        self.state_colors = model._get_colors(scalars=True)
        self._get_colors = model._get_colors
        self.new_datas = self.get_k_data(k)
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
        self.states_list=[self.get_state(seq=new_seq) for new_seq in self.new_seqs]
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
        datas = self.datas
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


hsmm_k =k_means(hdp_hsmm_model)
print('共有',str(hsmm_k.num_state),'种驾驶模式')
#print('分了',str(len(hsmm_k.new_seqs)),'个驾驶片段')
#print('各个驾驶片段行驶的时间为',str(hsmm_k.time_state),'*0.1s')


# shmm_k =k_means(hdp_s_hmm_model)
# print('共有',str(shmm_k.num_state),'种驾驶模式')
# print('分了',str(len(shmm_k.new_seq)),'个驾驶片段')
# print('各个驾驶片段行驶的时间为',str(shmm_k.time_state),'*0.1s')

# hsmm_k =k_means(hdp_hsmm_model)
# print('共有',str(hsmm_k.num_state),'种驾驶模式')
# print('分了',str(len(hsmm_k.new_seq)),'个驾驶片段')
# print('各个驾驶片段行驶的时间为',str(hsmm_k.time_state),'*0.1s')

#plotting_feature(hsmm_k)


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
for data in hsmm_k.new_datas:
    range_datas,thre = get_seg_d(data)
    anylyze_data=get_seg_v_a(range_datas,thre)
    all_counts = all_counts+np.array(get_prob(anylyze_data))
    


# %%
sizes=20
def plotting_style(all_counts):
    Y = ['急减','缓减','匀速','缓加','急加']
    X = ['快近','渐近','维持','渐远','快远']
    fig = plt.figure(figsize=(12,12))
    axes = []
    img = []
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
        img.append(axes[-1].pcolormesh(all_counts[idx],cmap = mpl.cm.Spectral_r))
        plt.subplots_adjust(wspace=0.4,hspace=0.3)
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right",size='5%',pad=0.05)
        #print(np.linspace(0.5,4.5,5,endpoint=True))
        norm = mpl.colors.Normalize(vmin=0,vmax=all_counts[idx].max())
        cmap = mpl.cm.Spectral_r
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)

all_counts=all_counts/len(hsmm_k.datas)
plotting_style(all_counts)
#plt.xla
plt.show()


# %%
import pandas as pd
if not os.path.isdir('all_counts'):
    os.makedirs('all_counts')
if not os.path.isdir(os.path.join('all_counts',dir)):
    os.makedirs(os.path.join('all_counts',dir))
for i in range(len(all_counts)):
    range_counts=all_counts[i]
    frame_counts=pd.DataFrame(range_counts)
    frame_counts.to_csv(os.path.join('all_counts',dir,str(i)+'.csv'))



# %%



