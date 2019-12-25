# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 19:33:20 2019

@author: AFNAN
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
ddqn_r=[]
e=[]
dqn_r=[]
dddqn_r=[]
avg_dqn=[]
avg_ddqn=[]
avg_dddqn=[]

with open('doubledqn.txt') as f:
    ddqn_str = f.read().split(",")
    e.append(ddqn_str[0])
    e.append(ddqn_str[3])
    e.append(ddqn_str[6])
    e.append(ddqn_str[9])
    e.append(ddqn_str[12])
    #e.append(ddqn_str[12])
    ddqn_r.append(ddqn_str[1])
    ddqn_r.append(ddqn_str[4])
    ddqn_r.append(ddqn_str[7])
    ddqn_r.append(ddqn_str[10])
    ddqn_r.append(ddqn_str[13])
    avg_ddqn.append(ddqn_str[2])
    avg_ddqn.append(ddqn_str[5])
    avg_ddqn.append(ddqn_str[8])
    avg_ddqn.append(ddqn_str[11])
    avg_ddqn.append(ddqn_str[14])
with open('dqn.txt') as f:
    dqn_str = f.read().split(",")
    dqn_r.append(dqn_str[1])
    dqn_r.append(dqn_str[4])
    dqn_r.append(dqn_str[7])
    dqn_r.append(dqn_str[10])
    dqn_r.append(dqn_str[13])
    avg_dqn.append(dqn_str[2])
    avg_dqn.append(dqn_str[5])
    avg_dqn.append(dqn_str[8])
    avg_dqn.append(dqn_str[11])
    avg_dqn.append(dqn_str[14])
with open('dueling.txt') as f:
    dddqn_str = f.read().split(",")
    dddqn_r.append(dddqn_str[1])
    dddqn_r.append(dddqn_str[4])
    dddqn_r.append(dddqn_str[7])
    dddqn_r.append(dddqn_str[10])
    dddqn_r.append(dddqn_str[13])
    avg_dddqn.append(dddqn_str[2])
    avg_dddqn.append(dddqn_str[5])
    avg_dddqn.append(dddqn_str[8])
    avg_dddqn.append(dddqn_str[11])
    avg_dddqn.append(dddqn_str[14])
    


ddqn_r = list(map(float, ddqn_r))
dqn_r = list(map(float, dqn_r))
dddqn_r = list(map(float, dddqn_r))
avg_ddqn = list(map(float, avg_ddqn))
avg_dqn = list(map(float, avg_dqn))
avg_dddqn = list(map(float, avg_dddqn))
#dueling_dqn_r = list(map(float, avg_ddqn))
e=list(map(float,e))
plt.plot(e, dqn_r, 'r', linestyle=':', marker='*')
plt.plot(e, ddqn_r, 'b', linestyle=':', marker='*')
plt.plot(e, dddqn_r, 'g', linestyle=':', marker='*')
plt.xlabel('User Demand (Mbps)')
plt.ylabel('Total Power Consumption (Watts)')
plt.legend(["DQN","Double DQN","Dueling DQN"])
plt.grid(True)
plt.xlim(10, 60) 

plt.ylim(min(dddqn_r)-(min(dddqn_r)%5), 5-(max(dddqn_r)%5)+max(dddqn_r))

banner = 'Scenario:', 10 , ' RRHs and' , 8 , ' users'
plt.title(banner)

#plt.grid(True)
plt.show()
#power efficiency

time_slot = range(0, len(avg_ddqn))
plt.plot(time_slot, avg_dqn, linestyle='-', marker='o')
plt.plot(time_slot, avg_ddqn, linestyle='-', marker='o')
plt.plot(time_slot, avg_dddqn, linestyle='-', marker='o')
plt.xlabel('Time Slot')
plt.ylabel('Average Total Power Consumption')
plt.title('Average total power consumption in time varying user demand scenario')
plt.grid(True)
plt.xlim(0, 20)
plt.legend(["DQN","Double DQN","Dueling DQN"])
plt.ylim(min(avg_ddqn)-(min(avg_ddqn)%5), 5-(max(avg_ddqn)%5)+max(avg_ddqn))
plt.show()

























