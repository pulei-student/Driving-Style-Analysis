# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import os
import shutil
import numpy as np
from matplotlib import font_manager
import matplotlib as mpl
import scipy
zhfont1 = font_manager.FontProperties(fname='SimHei.ttf',size=22)


# %%
size = 1e6
lables=['device','trip','time','range','rangerate','targettype','status','cipv','ax','speedwsu','obstacleid']
labless=['eventnum','device','trip','time','obstacleid','range','rangerate','targettype','status','cipv','ax','speedwsu']


# %%
ranges=list()
rangerate=list()
ax=list()
iters=0
path = 'car-following'
files = os.listdir(path)
for i in range(len(files)):
    dirs = os.listdir(os.path.join(path,files[i]))
    for j in range(len(dirs)):
        try:
            data=pd.read_csv(os.path.join(path,files[i],dirs[j]))
            ranges=ranges+list(data['range'])
            rangerate=rangerate+list(data['rangerate'])
            ax=ax+list(data['ax'])
            #iters=iters+1
        except StopIteration as e:
            print(iters)


# %%
# ranges=list()
# rangerate=list()
# ax=list()
# iters=0
# path = 'car-following/17103'
# files = os.listdir(path)
# chunk=pd.read_csv('event_18.csv',chunksize=size,names=labless)
# for i in range(12):
#     try:
#         data=chunk.get_chunk()
#         ranges=ranges+list(data['range'])
#         rangerate=rangerate+list(data['rangerate'])
#         ax=ax+list(data['ax'])
#         #iters=iters+1
#     except StopIteration as e:
#         print(iters)


# %%
data_test=data
print(data_test.head())


# %%
import matplotlib.pyplot as plt
from fitter import Fitter
f=Fitter(ranges,bins=200,distributions=['skewnorm','norm','gamma','laplace','t','beta'],xmin=5,xmax=120,timeout=600)
f.hist()
f.fit()


# %%
f.hist()
f.summary()


# %%

method="sumsquare_error"
Nbest=4
fs=27

fig=plt.figure(figsize=[20,5])

ax1=plt.subplot(121)
###drawing hist
plt.hist(ranges,bins=80,density=True,alpha=1)
###drawing fitted curve cdf
lable_list={'norm':'正态分布','gamma':'伽马分布','laplace':'拉普拉斯分布','beta':'贝塔分布','t':'学生t分布','skewnorm':'偏分布'}
colors={'norm':'olive','gamma':'r','beta':'b','t':'g','laplace':'m','skewnorm':'orange'}
try:
    names = f.df_errors.sort_values(
                    by=method).index[0:Nbest]
except Exception:
    names = f.df_errors.sort(method).index[0:Nbest]
flag=0
for name in names:
    if flag==0:
        ax1 = plt.plot(f.x,f.fitted_pdf[name],lw=3,label=lable_list[name],color=colors[name])
        flag=1
        pdf_list = list(f.fitted_pdf[name])
        x_list = list(f.x)
        y=max(pdf_list)
        x=round(x_list[pdf_list.index(y)],2)
        y=round(y,2)
    else:
        ax1 = plt.plot(f.x,f.fitted_pdf[name],lw=3,label=lable_list[name],color=colors[name],ls='--')
plt.xlim(0,120)
plt.xlabel('相对距离[m]',fontsize=fs,fontproperties=zhfont1)
plt.ylabel('概率密度',fontsize=fs,fontproperties=zhfont1)
plt.title('概率密度函数拟合图',fontsize=fs,fontproperties=zhfont1)
plt.tick_params(labelsize=fs)
plt.legend(fontsize=3,prop=zhfont1)
plt.grid(True)

plt.scatter(x,y,s=400,c='k',marker='+',linewidths=5)
plt.annotate((x,y),xy=(x,y),size=fs)

ax2=plt.subplot(122)
flag=0
for name in names:
    dist = eval('scipy.stats.'+name)
    cdf_fitted=dist.cdf(f.x,*f.fitted_param[name])
    if flag==0:
        ax2 = plt.plot(cdf_fitted,f.x,lw=4,label=lable_list[name],color=colors[name])
        flag=1
        params=f.get_best()[name]
        func=name
    else:
        plt.plot(cdf_fitted,f.x,lw=4,label=lable_list[name],color=colors[name],ls='--')
plt.xlim(0,1)
plt.ylim(0,100)
plt.xlabel('所占百分比',fontsize=fs,fontproperties=zhfont1)
plt.ylabel('相对距离[m]',fontsize=fs,fontproperties=zhfont1)
plt.title('累计分布函数图',fontsize=fs,fontproperties=zhfont1)
plt.tick_params(labelsize=fs)
plt.legend(prop=zhfont1,fontsize=3)
plt.grid(True)
print(func)
y_func=scipy.stats.gamma(*params)
x1 = 0.25
y1 = round(y_func.ppf(x1),2)
plt.scatter(x1,y1,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x1,y1),xy=(x1,y1),size=fs,xytext=(x1+0.05,y1))
x2 = 0.85
y2 = round(y_func.ppf(x2),2)
plt.scatter(x2,y2,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x2,y2),xy=(x2,y2),size=fs,xytext=(x2-0.35,y2))


# %%
g=Fitter(rangerate,bins=100,distributions=['gamma','t','beta','norm','laplace'],xmin=-8,xmax=8,timeout=600)
g.fit()


# %%
g.hist()
g.summary()


# %%
fig=plt.figure(figsize=[22,5])
method='sumsquare_error'
fs=27
import scipy
ax1=plt.subplot(121)
###drawing hist
plt.hist(rangerate,bins=400,density=True,alpha=1)
###drawing fitted curve cdf
lable_list={'norm':'正态分布','gamma':'伽马分布','laplace':'拉普拉斯分布','t':'学生t分布','beta':'贝塔分布'}
colors={'norm':'pink','gamma':'r','t':'g','laplace':'m','beta':'b'}
try:
    names = g.df_errors.sort_values(by=method).index[0:4]
except Exception:
    names = g.df_errors.sort_values(by=method).index[0:Nbest]
flag=0
for name in names:
    if flag==0:
        ax1 = plt.plot(g.x,g.fitted_pdf[name],lw=3,label=lable_list[name],color=colors[name])
        flag=1
        pdf_list = list(g.fitted_pdf[name])
        x_list = list(g.x)
        y=max(pdf_list)
        x=round(x_list[pdf_list.index(y)],3)
        y=round(y,3)
    else:
        ax1 = plt.plot(g.x,g.fitted_pdf[name],lw=3,label=lable_list[name],color=colors[name],ls='--')
    plt.xlim(-10,10)
    plt.xlabel('相对速度[m/s]',fontsize=fs,fontproperties=zhfont1)
    plt.ylabel('概率密度',fontsize=fs,fontproperties=zhfont1)
    plt.title('概率密度函数拟合图',fontsize=fs,fontproperties=zhfont1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=fs-5,prop=zhfont1)
    plt.grid(True)
   
#plt.scatter(x,y,s=400,c='k',marker='+',linewidths=5)
#plt.annotate((x,y),xy=(x,y),size=fs)

ax2=plt.subplot(122)
flag=0
for name in names:
    dist = eval('scipy.stats.'+name)
    cdf_fitted=dist.cdf(g.x,*g.fitted_param[name])
    if flag==0:
        ax2 = plt.plot(cdf_fitted,g.x,lw=4,label=lable_list[name],color=colors[name])
        flag=1
        params=g.get_best()[name]
        func=name
    else:
        plt.plot(cdf_fitted,g.x,lw=4,label=lable_list[name],color=colors[name],ls='--')
    plt.xlim(0,1)
    plt.ylim(-5,7)
    plt.xlabel('所占百分比',fontsize=fs,fontproperties=zhfont1)
    plt.ylabel('相对速度[m/s]',fontsize=fs,fontproperties=zhfont1)
    plt.title('累计分布函数图',fontsize=fs,fontproperties=zhfont1)
    plt.tick_params(labelsize=fs)
    plt.legend(fontsize=2,prop=zhfont1)
    plt.grid(True)
print(func)
y_func=scipy.stats.laplace(*params)
x1 = 0.15
y1 = round(y_func.ppf(x1),2)
plt.scatter(x1,y1,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x1,y1),xy=(x1,y1),size=fs,xytext=(x1-0.15,y1-1.5))
x2 = 0.40
y2 = round(y_func.ppf(x2),2)
plt.scatter(x2,y2,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x2,y2),xy=(x2,y2),size=fs,xytext=(x2-0.12,y2-1.3))
x3 = 0.60
y3 = round(y_func.ppf(x3),2)
plt.scatter(x3,y3,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x3,y3),xy=(x3,y3),size=fs,xytext=(x3+0.02,y3-0.5))
x4 = 0.85
y4 = round(y_func.ppf(x4),2)
plt.scatter(x4,y4,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x4,y4),xy=(x4,y4),size=fs,xytext=(x4-0.15,y4+0.5))


# %%
h=Fitter(ax,bins=100,xmin=-3,xmax=3,distributions=['gamma','t','norm','laplace'],timeout=600)
h.fit()


# %%
h.hist()


# %%
fig=plt.figure(figsize=[22,5])

ax1=plt.subplot(121)
###drawing hist
plt.hist(ax,bins=300,density=True,alpha=1)
###drawing fitted curve cdf
lable_list={'norm':'正态分布','gamma':'伽马分布','laplace':'拉普拉斯分布','beta':'贝塔分布','t':'学生t分布'}
colors={'norm':'pink','gamma':'r','beta':'b','t':'g','laplace':'m'}
Nbest=4
try:
    names = h.df_errors.sort_values(
                    by=method).index[0:Nbest]
except Exception:
    names = h.df_errors.sort_values(by=method).index[0:Nbest]
flag=0
for name in names:
    if flag==0:
        plt.plot(h.x,h.fitted_pdf[name],lw=3,label=lable_list[name],color=colors[name])
        flag=1
        pdf_list = list(h.fitted_pdf[name])
        x_list = list(h.x)
        y=max(pdf_list)
        x=round(x_list[pdf_list.index(y)],3)
        y=round(y,3)
    else:
        plt.plot(h.x,h.fitted_pdf[name],lw=3,label=lable_list[name],color=colors[name],ls='--')
    plt.xlim(-3,3)
    plt.xlabel('加速度'+r'$[m\/s^2]$',fontsize=fs,fontproperties=zhfont1)
    plt.ylabel('概率密度',fontsize=fs,fontproperties=zhfont1)
    plt.title('概率密度函数拟合图',fontsize=fs,fontproperties=zhfont1)
    plt.tick_params(labelsize=fs)
    ax1.legend(fontsize=fs-5,prop=zhfont1)
    plt.grid(True)
   
#plt.scatter(x,y,s=400,c='k',marker='+',linewidths=5)
#plt.annotate((x,y),xy=(x,y),size=fs)

ax2=plt.subplot(122)
flag=0
for name in names:
    dist = eval('scipy.stats.'+name)
    cdf_fitted=dist.cdf(h.x,*h.fitted_param[name])
    if flag==0:
        ax2 = plt.plot(cdf_fitted,h.x,lw=4,label=lable_list[name],color=colors[name])
        flag=1
        params=h.get_best()[name]
        func=name
    else:
        plt.plot(cdf_fitted,h.x,lw=4,label=lable_list[name],color=colors[name],ls='--')
    plt.xlim(0,1)
    plt.ylim(-2.5,2)
    plt.xlabel('所占百分比',fontsize=fs,fontproperties=zhfont1)
    plt.ylabel('加速度'+r'$[m/s^2]$',fontsize=fs,fontproperties=zhfont1)
    plt.title('累计分布函数图',fontsize=fs,fontproperties=zhfont1)
    plt.tick_params(labelsize=fs)
    plt.legend(loc='lower right',prop=zhfont1)
    plt.grid(True)
print(func)
y_func=scipy.stats.t(*params)
x1 = 0.20
y1 = round(y_func.ppf(x1),2)
plt.scatter(x1,y1,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x1,y1),xy=(x1,y1),size=fs,xytext=(x1-0.19,y1-0.55))
print(x1,y1)
x2 = 0.40
y2 = round(y_func.ppf(x2),2)
plt.scatter(x2,y2,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x2,y2),xy=(x2,y2),size=fs,xytext=(x2-0.1,y2-0.55))
x3 = 0.60
y3 = round(y_func.ppf(x3),2)
plt.scatter(x3,y3,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x3,y3),xy=(x3,y3),size=fs,xytext=(x3-0.15,y3+0.2))
x4 = 0.80
y4 = round(y_func.ppf(x4),2)
plt.scatter(x4,y4,s=400,c='black',marker='*',linewidths=5)
plt.annotate((x4,y4),xy=(x4,y4),size=fs,xytext=(x4+0.02,y4-0.2))


# %%



