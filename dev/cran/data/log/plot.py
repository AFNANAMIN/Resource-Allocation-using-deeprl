# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 07:24:52 2019

@author: hania
"""

from statistics import mean
import matplotlib.pyplot as plt

# power from 0-10
#f = open("11-05-11-39-27-dqn.log", "r") #scenario 1
#f = open("11-05-12-57-10-dqn.log", "r") #scenario 2
f = open("11-05-14-27-03-dqn.log", "r") #scenario 3
log_lines = f.read().split('\n')
f.close()
ep_power10 = []
str_to_start_find = "Ep-power: "
str_to_end_find = " | Num-rrh-on:"
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power10.append(float(s))
            
p10=mean(ep_power10)

# power from 10-20
#f = open("11-05-11-46-52-dqn.log", "r") #scenario 1
#f = open("11-05-14-18-59-dqn.log", "r") #scenario 2
f = open("11-05-14-28-45-dqn.log", "r") #scenario 3
log_lines = f.read().split('\n')
f.close()
ep_power20 = []
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power20.append(float(s))
            
p20=mean(ep_power20)

# power from 20-30
#f = open("11-05-11-49-03-dqn.log", "r") #scenario 1
#f = open("11-05-14-20-18-dqn.log", "r") #scenario 2
f = open("11-05-16-01-48-dqn.log", "r") #scenario 3
log_lines = f.read().split('\n')
f.close()
ep_power30 = []
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power30.append(float(s))
            
p30=mean(ep_power30)

# power from 30-40
#f = open("11-05-11-54-47-dqn.log", "r") #scenario 1
#f = open("11-05-14-23-11-dqn.log", "r") #scenario 2
f = open("11-05-16-03-29-dqn.log", "r") #scenario 3
log_lines = f.read().split('\n')
f.close()
ep_power40 = []
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power40.append(float(s))
            
p40=mean(ep_power40)


# power from 40-50
#this does not works in scenario 2 as all the power values aboce 40 are infinite - check log file 11-05-14-24-18-dnq.log
#comment this before running for scenario 2

#f = open("11-05-12-00-21-dqn.log", "r") #scenario 1
f = open("11-05-16-08-10-dqn.log", "r") #scenario 3
log_lines = f.read().split('\n')
f.close()
ep_power50 = []
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power50.append(float(s))
            
p50=mean(ep_power50)


# power from 50-60
#this does not works in scenario 2 as all the power values aboce 40 are infinite - check log file 11-05-14-24-18-dnq.log
#comment this before running for scenario 2

#f = open("11-05-12-03-15-dqn.log", "r") #scenario 1
f = open("11-05-16-12-37-dqn.log", "r") #scenario 3
log_lines = f.read().split('\n')
f.close()
ep_power60 = []
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power60.append(float(s))
            
p60=mean(ep_power60)

user_demand = [10, 20, 30, 40, 50, 60] #scenario 1, scenario 3
#user_demand = [10, 20, 30, 40] #scenario 2

total_power = [p10, p20, p30, p40, p50, p60] #scenario 1, scenario 3
#total_power = [p10, p20, p30, p40] #scenario 2

plt.plot(user_demand, total_power, linestyle='-', marker='o')

plt.xlabel('User Demand (Mbps)')
plt.ylabel('Total Power Consumption (Watts)')
plt.xlim(10, 60) #scenario 1, scenario 2
#plt.ylim(30, 55) #scenario 1, scenario 2
plt.ylim(40, 70) #scenario 3

#plt.title('Scenario 1: 6 RRHs and 3 Users') #scenario 1
#plt.title('Scenario 2: 6 RRHs and 4 Users') #scenario 2
plt.title('Scenario 3: 8 RRHs and 4 Users') #scenario 2
plt.grid(True)
plt
plt.show()
            
