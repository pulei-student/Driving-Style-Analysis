# %%
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
zhfont1=font_manager.FontProperties(fname='SimHei.ttf',size=20)
# %%
root='all_counts'
root_='no_seg'
dir_list=os.listdir(root)
# %%
file=pd.read_csv(os.path.join(root,dir_list[0],'1.csv'),index_col=0)
print(file)
print(np.shape(file))
# %%
driver=list()
driver_seg=list()
for h in range(3):    
    driver_range=list()
    driver_seg_range=list()
    for idx in range(10):
        file=pd.read_csv(os.path.join(root,dir_list[idx],str(h)+'.csv'),index_col=0)
        file_=pd.read_csv(os.path.join(root_,dir_list[idx],str(h)+'.csv'),index_col=0)
        driver_range.append(np.array(file))
        driver_seg_range.append(np.array(file_))
    driver.append(driver_range)
    driver_seg.append(driver_seg_range)
#%%
print(file)
print(np.shape(driver))
# %%
differ_all=list()
self_differ_all=list()
fig=plt.figure(figsize=(8,5))
axes=[]
titles=['远距离','中距离','短距离']
img=[]
for h in range(3):
    differ=np.zeros((10,10))
    self_differ=np.zeros((10,1))
    for idx in range(10):
        for jdx in range(10):
            for i in range(5):
                for j in range(5):
                    differ[idx][jdx] = differ[idx][jdx]+driver_seg[h][idx][i][j]*\
                        np.log(driver_seg[h][idx][i][j]/driver[h][jdx][i][j])
                    self_differ[idx][0] = differ[idx][0]+driver[h][idx][i][j]*\
                        np.log(driver[h][idx][i][j]/driver_seg[h][idx][i][j])
    differ_all.append(differ)
    self_differ_all.append(self_differ)
    axes.append(fig.add_subplot(3,1,h+1))
    axes[-1].set_title(titles[h],FontProperties=zhfont1)
    #Raxes[-1].set_xlabel('驾驶员',FontProperties=zhfont1)
    #axes[-1].set_xlabel('驾驶员',FontProperties=zhfont1)
    axes[-1].set_yticks(np.linspace(0.5,1,1))
    axes[-1].set_yticklabels(['#'])
    axes[-1].set_xticks(np.linspace(0.5,9.5,10,endpoint=True))
    axes[-1].set_xticklabels(['#0','#1','#2','#3','#4','#5','#6','#7','#8','#9'])
    plt.tick_params(labelsize=15)
    img.append(axes[-1].pcolormesh(self_differ.T,cmap=mpl.cm.jet))
    divider = make_axes_locatable(axes[-1])
    cax=divider.append_axes("bottom",size='50%',pad=0.3)
    norm = mpl.colors.Normalize(vmin=0,vmax=1)
    cmap = mpl.cm.jet
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
plt.subplots_adjust(hspace=1,wspace=0.5)
plt.show()
# %%
