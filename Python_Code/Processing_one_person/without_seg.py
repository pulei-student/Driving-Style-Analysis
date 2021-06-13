# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import numpy as np
from matplotlib import pyplot as plt
import os

from matplotlib import font_manager
import matplotlib as mpl
zhfont1 = font_manager.FontProperties(fname='SimHei.ttf')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd



# %%
def loading_data(path):
    import pandas as pd
    import os
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


dir='17103'
datas_orig = loading_data(os.path.join('car-following',dir))

iter = 100
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
for data in datas:
    range_datas,thre = get_seg_d(data)
    anylyze_data=get_seg_v_a(range_datas,thre)
    all_counts = all_counts+np.array(get_prob(anylyze_data))
    


# %%

def plotting_style(all_counts):
    Y = ['急减','缓减','匀速','缓加','急加']
    X = ['快近','渐近','维持','渐远','快远']
    fig = plt.figure(figsize=(7,23))
    axes = []
    img = []
    titles = ['远距离','中距离','短距离']
    for idx in range(len(all_counts)): 
        axes.append(fig.add_subplot(3,1,idx+1))
        axes[-1].set_ylabel('加速度'+r'$a_x$',size=14,fontproperties=zhfont1)
        axes[-1].set_xticks(np.linspace(0.5,4.5,5,endpoint=True))
        axes[-1].set_xticklabels(X,fontproperties=zhfont1,size=13)
        axes[-1].set_xlabel('相对速度'+r'$\Delta$v',size=14,fontproperties=zhfont1)
        axes[-1].set_yticks(np.linspace(0.5,4.5,5,endpoint=True))
        axes[-1].set_yticklabels(Y,fontproperties=zhfont1,size=13)
        axes[-1].set_title(titles[idx],fontproperties=zhfont1,size=16)
        img.append(axes[-1].pcolormesh(all_counts[idx],cmap = mpl.cm.Spectral_r))
        
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right",size='5%',pad=0.05)
        #print(np.linspace(0.5,4.5,5,endpoint=True))
        norm = mpl.colors.Normalize(vmin=0,vmax=all_counts[idx].max())
        cmap = mpl.cm.Spectral_r
        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax)

all_counts=all_counts/len(datas)
plotting_style(all_counts)
#plt.xla
plt.show()


# %%
if not os.path.isdir('no_seg'):
    os.makedirs('no_seg')
if not os.path.isdir(os.path.join('no_seg',dir)):
    os.makedirs(os.path.join('no_seg',dir))
for i in range(len(all_counts)):
    range_counts=all_counts[i]
    frame_counts=pd.DataFrame(range_counts)
    frame_counts.to_csv(os.path.join('no_seg',dir,str(i)+'.csv'))



# %%



