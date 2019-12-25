import argparse
import datetime
import os
from os.path import join as pjoin


class CRANParser(argparse.ArgumentParser):
    TIME_STAMP = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')

    @staticmethod
    def __home_out(path):
        full_path = pjoin('./', 'dev', 'cran', 'data', path)
        print("Aaaaa")
        if not os.path.exists(full_path):
            print("Makedir")
            os.makedirs(full_path)
        return full_path

    def __init__(self):
        super(CRANParser, self).__init__()

        self.__init_cran()
        self.__init_dqn()
        self.__init_agent()
        self.__init_env()

    def __init_cran(self):
        self.add_argument('--num_rrh', type=int, default=6, help='number of RRH per cell')
        self.add_argument('--num_usr', type=int, default=3, help='number of usr per cell')

        self.add_argument('--demand_min', type=float, default=10.e6, help='minimal user demand bps')
        self.add_argument('--demand_max', type=float, default=60.e6, help='maximal user demand bps')

        self.add_argument('--band', type=float, default=10.e6, help='the bandwidth Hz')
        self.add_argument('--tm', type=float, default=1, help='tm ??')
        self.add_argument('--eta', type=float, default=0.25, help='power amplifier efficiency')
        self.add_argument('--theta_2', type=float, default=6.3095734448e-14, help='theta^2 ??')

        self.add_argument('--pow_on', type=float, default=6.8, help='active power for RRH Watts')
        self.add_argument('--pow_slp', type=float, default=4.3, help='sleep power for RRH Watts')
        self.add_argument('--pow_gap', type=float, default=3.0, help='transition power for RRH Watts')
        self.add_argument('--pow_tsm', type=float, default=1.0, help='maximal transmit power for RRH Watts')

    def __init_dqn(self):
        self.add_argument('--lr', type=float, default=1.e-3, help='learning rate for dqn')

    def __init_agent(self):
        self.add_argument('--observations', type=int, default=100, help='observations steps')
        self.add_argument('--update', type=int, default=8, help='n step q learning')
        self.add_argument('--tests', type=int, default=10, help='testing episode')
        self.add_argument('--episodes', type=int, default=100, help='training episode')
        self.add_argument('--epsilon_steps', type=int, default=4000, help='episodes for epsilon greedy explore')
        self.add_argument('--epochs', type=int, default=10, help='training epochs for each episode')

        self.add_argument('--save_ep', type=int, default=20, help='save model every n episodes')
        self.add_argument('--load_id', type=str, default=None, help='the model id to restore')

        self.add_argument('--gamma', type=float, default=0.99, help='reward discount rate')

        self.add_argument('--buffer_size', type=int, default=100000, help='size of replay buffer')
        self.add_argument('--mini_batch', type=int, default=64, help='size of mini batch')

        self.add_argument('--epsilon_init', type=float, default=0.5, help='initial value of explorer epsilon')
        self.add_argument('--epsilon_final', type=float, default=0.01, help='final value of explorer epsilon')

    def __init_env(self):
        self.add_argument('--random_seed', type=int, default=10000, help='seed of random generation')
        self.add_argument('--dir_sum', type=str, default=self.__home_out('sum'), help='the path of tf summary')
        self.add_argument('--dir_mod', type=str, default=self.__home_out('mod'), help='the path of tf module')
        self.add_argument('--dir_log', type=str, default=self.__home_out('log'), help='the path of tf log')
        self.add_argument('--run_id', type=str, default=self.TIME_STAMP)
