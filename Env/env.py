import numpy as np
import cvxopt as cvx
import math

from Env.opter import CvxOpt


class Env:
    REWARD_NEG = 0
    STATE_ON = 1
    STATE_OFF = 0

    def __init__(self, name, configure):

        self.name = name

        if configure.random_seed >= 0:
            np.random.seed(configure.random_seed)

        self._num_rrh = configure.num_rrh
        self._num_usr = configure.num_usr

        self._DM_MAX = configure.demand_max
        self._DM_MIN = configure.demand_min

        self._pow_on = configure.pow_on
        self._pow_slp = configure.pow_slp
        self._pow_gap = configure.pow_gap
        self._pow_tsm = configure.pow_tsm

        self._dm = self._generate_demand()

        self.MAX_EP = configure.episodes
        self.MAX_EXP_EP = configure.epsilon_steps
        self.MAX_TEST_EP = configure.tests

        self._dm_index = 0

        self._BAND = configure.band   #######bandwidth
        self._ETA = configure.eta     
        self._THETA_2 = configure.theta_2
        self._TM = configure.tm
        # todo replace const with dynamical variable
        self._CONST = 1.345522816371604e-06

        self._P_MIN, self._P_MAX = self._get_power_bound()

        all_off = np.zeros(self._num_rrh) + self.STATE_OFF
        self._state_rrh_min = all_off.copy()
        self._state_rrh_min_last = all_off.copy()
        self._state_rrh_max = all_off.copy()
        self._state_rrh_last = self._state_rrh = all_off.copy()
        self._state_rrh_rd_last = self._state_rrh_rd = all_off.copy()

        self.reset()

    @property           ### state space is the user demand plus the number of rrh
    def state(self):
        dm = (self._demand - self._DM_MIN) / (self._DM_MAX - self._DM_MIN)
        print("state",self._state_rrh)
        print("dm", dm)
        return np.concatenate([self._state_rrh, dm])                    ####Concatenation refers to joining. This function is used to join two or more arrays of the same shape along a specified axis

    @property
    def demand(self):
        return np.around(self._demand / 10e6, decimals=3)

    @property
    def dim_state(self):
        return len(self.state)

    @property
    def dim_action(self):
        return self._num_rrh * 2 + 1
        # return self._num_rrh + 1

    @property
    def num_rrh(self):
        return self._num_rrh

    @property
    def num_rrh_on(self):
        return len((np.where(self._state_rrh == self.STATE_ON))[0])

    @property
    def max_rrh_reward(self):
        return self.on_max, self.power_max, self.reward_max

    @property
    def min_rrh_reward(self):
        return self.on_min, self.power_min, self.reward_min

    @property
    def rnd_rrh_reward(self):
        return self.on_rnd, self.power_rnd, self.reward_rnd

    def run_fix_solution(self):
        self._get_max_rrh_solution()
        self._get_min_rrh_solution()
        self._get_rnd_rrh_solution()
        self.on_max, self.power_max, self.reward_max = self._get_max_rrh_reward()
        self.on_min, self.power_min, self.reward_min = self._get_min_rrh_reward()
        self.on_rnd, self.power_rnd, self.reward_rnd = self._get_rnd_rrh_reward()

    def reward_to_power(self, reward):
        return (1.0 - reward) * (self._P_MAX - self._P_MIN) + self._P_MIN

    def reset(self):
        self.reset_channel()
        self.reset_demand()
        self.run_fix_solution()
        s = self.reset_state()
        return s

    def reset_channel(self):
        self._paras = self._init_channel()
        self._opter = CvxOpt()

    def reset_demand(self):
        self._demand = self._get_demand()
        self._paras['cof'] = self._get_factor(rk_demand=self._demand)

    def reset_state(self):
        self._state_rrh = np.zeros(self._num_rrh) + self.STATE_ON
        self._state_rrh_last = self._state_rrh.copy()

        return self.state

    def step(self, action):
        _, _, _ = self.sub_step(action)
        power, reward, done = self.perform()
        # done = True if stop else done
        return self.state, power, reward, done

    def sub_step(self, action):
        action_index = np.argmax(action)

        if action_index == self.dim_action - 1:
            # stop=True
            return self.state, 0, True

        s_rrh_old = self._state_rrh[int(action_index / 2)]
        if action_index % 2 == 0:
            if s_rrh_old == 1:
                pass
            else:
                self._state_rrh[int(action_index / 2)] = 1
        else:
            if s_rrh_old == 0:
                pass
            else:
                self._state_rrh[int(action_index / 2)] = 0

        return self.state, 0, False

    def perform(self):
        power, reward, done = self._get_power_reward_done(self._state_rrh, self._state_rrh_last)
        self._state_rrh_last = self._state_rrh.copy()
        return power, reward, done

    def _get_power_reward_done(self, state_rrh, state_last):
        done = False
        solution = self._get_solution(state_rrh)
        if solution:
            power, reward = self._get_reward(solution, state_rrh, state_last)
        else:
            # todo: replace power with a reasonable value, can not be 0
            power = reward = self.REWARD_NEG
            done = True
        return power, reward, done

    def _get_solution(self, state_rrh):
        on_index = np.where(state_rrh == self.STATE_ON)[0].tolist()
        num_on = len(on_index)

        # No active RRH
        if num_on == 0:
            return None

        self._opter.feed(
            h=self._paras['h'][on_index, :],
            cof=self._paras['cof'],
            p=self._paras['pl'][on_index],
            theta=self._paras['theta'],
            num_rrh=num_on,
            num_usr=self._num_usr
        )

        solution = self._opter.solve()

        if solution['x'] is None:
            return None
        else:
            return solution
 
    def _get_reward(self, solution, state_rrh, state_rrh_last):
        num_on = len((np.where(state_rrh == self.STATE_ON))[0])
        num_on_last = len((np.where(state_rrh_last == self.STATE_ON))[0])

        num_off = len(np.where(state_rrh == self.STATE_OFF)[0])

        # transition power
        diff = num_on - num_on_last
        power = self._pow_gap * diff if diff > 0 else 0
        # print('trP:', power)

        # on and sleep power
        p = (num_on * self._pow_on + num_off * self._pow_slp)
        power += p
        # print('ooP:', p, 'On:', num_on)

        # transmit power
        p = sum(solution['x'][1:] ** 2) * (1.0 / self._ETA)
        power += p
        # print('tmP:', p)

        # normalized power
        reward_norm = (power - self._P_MIN) / (self._P_MAX - self._P_MIN)

        # power to reward
        reward_norm = 1 - reward_norm

        # power, reward, done
        return power, reward_norm

    def _get_max_rrh_reward(self):
        power, reward, _ = self._get_power_reward_done(self._state_rrh_max, self._state_rrh_max)
        return self._num_rrh, power, reward

    def _get_min_rrh_reward(self):
        power, reward, _ = self._get_power_reward_done(self._state_rrh_min, self._state_rrh_min_last)
        return self._num_usr, power, reward

    def _get_rnd_rrh_reward(self):
        num_on = len((np.where(self._state_rrh_rd == self.STATE_ON))[0])
        power, reward, _ = self._get_power_reward_done(self._state_rrh_rd, self._state_rrh_rd_last)
        return num_on, power, reward

    def _get_max_rrh_solution(self):
        self._state_rrh_max = np.zeros(self._num_rrh) + self.STATE_ON

    def _get_min_rrh_solution(self):
        # todo: get uniform initializer
        self._state_rrh_min_last = self._state_rrh_min.copy()

        rd_num_on = range(self._num_rrh)
        rd_num_on = np.random.choice(rd_num_on, self._num_usr, replace=False)
        self._state_rrh_min = np.zeros(self._num_rrh)
        self._state_rrh_min[rd_num_on] = self.STATE_ON

    def _get_rnd_rrh_solution(self):
        state_rrh = np.zeros(self._num_rrh)
        for i in range(1, self._num_rrh + 1):
            state_rrh[:i] = self.STATE_ON
            _, _, done = self._get_power_reward_done(state_rrh, self._state_rrh_rd_last)
            if not done:
                break

        self._state_rrh_rd_last = self._state_rrh_rd.copy()
        self._state_rrh_rd = state_rrh.copy()

    def _get_gains(self, num_rrh=0, num_usr=0):
#        d = np.random.uniform(0, 800, size = (num_rrh, num_usr))
#        L = 14.81+3.76* np.log2(d)
#        c = -1 * L / 20
#        antenna_gain = 0.9
#        s = 0.8
#        channel_gains = pow(10, c) * math.sqrt((antenna_gain*s)) * np.random.rayleigh(scale=1.0, size=(num_rrh, num_usr))
        channel_gains = np.random.rayleigh(scale=1.0, size=(num_rrh, num_usr))
        channel_gains = cvx.matrix(channel_gains) * self._CONST  # * 1.345522816371604e-06
        return channel_gains

    def _get_factor(self, rk_demand):
        mu = np.array([self._TM * (2 ** (i / self._BAND) - 1) for i in rk_demand])
        factor = cvx.matrix(np.sqrt(1. + (1. / mu)))
        return factor

    def _get_demand(self):
        rk_demand = self._dm[self._dm_index]
        self._dm_index += 1
        return rk_demand

    def _generate_demand(self):
        rd = np.random.uniform(self._DM_MIN, self._DM_MAX, size=(20000, self._num_usr))
        return rd

    def _get_power_bound(self):
        pow_min = 1 * self._pow_on + (self._num_rrh - 1) * self._pow_slp
        pow_max = self._num_rrh * self._pow_on
        pow_max += self._num_rrh * (1.0 / self._ETA) * self._pow_tsm
        pow_max += self._pow_gap
        return pow_min, pow_max

    def _init_channel(self):
        self._demand = self._get_demand()
        p_max = np.zeros(self._num_rrh) + self._pow_tsm
        theta = np.zeros(self._num_usr) + self._THETA_2

        def _get_pl(p_max):
            pl = cvx.matrix(np.sqrt(p_max), size=(1, len(p_max)))
            return pl

        def _get_theta(theta):
            theta = cvx.matrix(np.sqrt(theta), size=(1, len(theta)))
            return theta

        return {
            'h': self._get_gains(num_rrh=self._num_rrh, num_usr=self._num_usr),
            'cof': self._get_factor(rk_demand=self._demand),
            'pl': _get_pl(p_max=p_max),
            'theta': _get_theta(theta=theta)
        }
