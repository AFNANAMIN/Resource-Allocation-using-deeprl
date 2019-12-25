#from Agent.dqn import DQNAgent, user_demand, total_power, average_slot_power
from Agent.duelingdqn import DuelingDQNAgent, user_demand, total_power, average_slot_power
#from Agent.doubledqn import DoubleDQNAgent, user_demand, total_power, average_slot_power
from Env.env import Env
from parsers import CRANParser
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def main(args):
    
    d_min, d_max, n_rrh, n_usr, n_epochs = args
    for i in range(len(d_min)):
        
        parser = CRANParser()
        parser.set_defaults(demand_min=d_min[i], demand_max=d_max[i], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
        config = parser.parse_args()
        print("Eenie")
        env = Env('master', config)
        print("Meenie")
        #gains = env._get_gains(n_rrh, n_usr) 
        #print (gains)                        
        #dqn_agent = DQNAgent(env, config) 
        dueling_agent =DuelingDQNAgent(env, config) 
        #agent = DoubleDQNAgent(env, config)
        print("Minie")                      
        tf.reset_default_graph()              
        dueling_agent.work()                        
    


if __name__ == '__main__':
    d_min = [0.e6, 10.e6, 20.e6, 30.e6, 40.e6, 50.e6]   
    d_max = [10.e6, 20.e6, 30.e6, 40.e6, 50.e6, 60.e6]  
    #d_min = [10.e6]                                      
    #d_max = [60.e6]                                      
    n_rrh = 10
    n_usr = 8
    n_epochs = 10
    args = d_min, d_max, n_rrh, n_usr, n_epochs
    
    main(args)
    

    
#    time_slot = range(0, len(average_slot_power))
#    plt.plot(time_slot, average_slot_power, linestyle='-', marker='o')
#    plt.xlabel('Time Slot')
#    plt.ylabel('Average Total Power Consumption')
#    plt.title('Average total power consumption in time varying user demand scenario')
#    plt.grid(True)
#    plt.xlim(0, 20)
#    plt.ylim(min(average_slot_power)-(min(average_slot_power)%5), 5-(max(average_slot_power)%5)+max(average_slot_power))
#    plt.show()
    
    plt.plot(user_demand, total_power, 'r', linestyle=':', marker='*')
    plt.xlabel('User Demand (Mbps)')
    plt.ylabel('Total Power Consumption (Watts)')
    plt.xlim(10, 60) 
    plt.ylim(min(total_power)-(min(total_power)%5), 5-(max(total_power)%5)+max(total_power))
    banner = 'Scenario:', n_rrh , ' RRHs and' , n_usr , ' users'
    plt.title(banner)
    plt.grid(True)
    plt
    plt.show()
    print(total_power)
    
    parser = CRANParser()
    parser.set_defaults(demand_min=d_min[0], demand_max=d_max[0], num_rrh=n_rrh, num_usr=n_usr, epochs=n_epochs)
    config = parser.parse_args()
    print("Eenie")
    env = Env('master', config)
    print("Meenie")
    ch_gain = env._get_gains(n_rrh, n_usr)
    print ('channel gain  = ', ch_gain)