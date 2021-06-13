# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import pandas as pd
import os
import shutil
import numpy as np


# %%
root = 'car-following'
if os.path.exists(root):
    shutil.rmtree(root)
os.makedirs(root)


# %%
size = 1e6
lables=['eventnum','device','trip','time','obstacleid','range','rangerate','targettype','status','cipv','ax','speedwsu']
data = pd.read_csv('event_18.csv',names=lables)


# %%
print(data.head())

# %% [markdown]
# #columns: eventnum,device,trip,time,range,rangerate,targettype,status,cipv,axwsu,speedwsu,obstacleid

# %%

event_num = data['eventnum']
change_list=[0]
for i in range(1,len(data)):
    if event_num[i]!=event_num[i-1]:
        change_list.append(i)
change_list.append(len(event_num))
###slips
print('there are '+str(len(change_list)+1)+' car following events. Now selecting envents time over 50s.')


# %%
number=0
for j in range(1,len(change_list)):
    if change_list[j]-change_list[j-1]>=500:
        event=data.iloc[change_list[j-1]:change_list[j],:]
        if not os.path.isdir(os.path.join(root,str(data.loc[change_list[j-1],'device']))):
                os.makedirs(os.path.join(root,str(data.loc[change_list[j-1],'device'])))
        number=number+1
        event.to_csv(os.path.join(root,str(data.loc[change_list[j-1],'device']),str(number)+'.csv'))
        


# %%
print('there are '+str(number)+' car following events in total.')


# %%



