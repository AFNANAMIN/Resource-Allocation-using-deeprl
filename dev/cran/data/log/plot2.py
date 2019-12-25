# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 17:21:46 2019

@author: hania
"""

from statistics import median
import matplotlib.pyplot as plt

# power from 0-10
f = open("11-05-11-39-27-dqn.log", "r") #scenario 1
log_lines = f.read().split('\n')
f.close()
ep_power = []
str_to_start_find = "Ep-power: "
str_to_end_find = " | Num-rrh-on:"
for i, log in enumerate(log_lines):
        if str_to_start_find in log:
            s = log[log.find(str_to_start_find)+len(str_to_start_find):log.rfind(str_to_end_find)]
            if s != 'nan':
                ep_power.append(float(s))
            
            
            
p = []
piece = ep_power[0:13]
p.append(median(piece))
piece = ep_power[13:24]
p.append(median(piece))
piece = ep_power[24:34]
p.append(median(piece))
piece = ep_power[34:45]
p.append(median(piece))
piece = ep_power[45:66]
p.append(median(piece))
piece = ep_power[66:76]
p.append(median(piece))
piece = ep_power[76:86]
p.append(median(piece))
piece = ep_power[86:97]
p.append(median(piece))
piece = ep_power[97:108]
p.append(median(piece))
piece = ep_power[108:119]
p.append(median(piece))
piece = ep_power[119:129]
p.append(median(piece))
piece = ep_power[129:140]
p.append(median(piece))
piece = ep_power[140:150]
p.append(median(piece))
piece = ep_power[150:160]
p.append(median(piece))
piece = ep_power[160:171]
p.append(median(piece))
piece = ep_power[171:181]
p.append(median(piece))
piece = ep_power[181:203]
p.append(median(piece))
piece = ep_power[203:213]
p.append(median(piece))
piece = ep_power[213:224]
p.append(median(piece))
piece = ep_power[224:235]
p.append(median(piece))

time_slot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
plt.plot(time_slot, p, linestyle='-', marker='o')

plt.xlabel('Time Slot')
plt.ylabel('Average Total Power Consumption')
plt.title('Average total power consumption in time varying user demand scenario')
plt.grid(True)
plt.xlim(0, 20)
plt.ylim(29.5, 33.5)
plt.show()
            