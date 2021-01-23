from collections import deque
import numpy as np


class GeneralizedABCDHebbian:
    def __init__(self):
        self.learning_rate = 1e-4
        self.Aw = 1.0
        self.Bw = 0.0
        self.Cw = 0.0
        self.Dw = 0.0

    
    def step(self, inputs, activities, reward=None):
        # print(reward)
        act_inpt_cat = np.r_[inputs, activities]
        weight_update = self.learning_rate * (self.Aw * np.outer(activities, act_inpt_cat) \
                        + self.Bw * np.outer(activities, np.ones_like(act_inpt_cat))\
                        + self.Bw * np.outer(np.ones_like(activities), act_inpt_cat) + self.Dw)
        return weight_update * reward if reward is not None else weight_update

    def reset(self):
        pass

    def get_params(self):
        return np.r_[self.Aw, self.Bw, self.Cw, self.Dw]
    
    def set_params(self, data):
        self.Aw, self.Bw, self.Cw, self.Dw = data.copy()
    
    def len_params(self):
        return self.get_params().shape[0]
    
    def init_params(self, min_val, max_val):
        params_len = self.len_params()
        random_params = np.random.uniform(low=min_val, high=max_val, size=params_len)
        self.set_params(random_params)


class BufferedHebb(GeneralizedABCDHebbian):
    def __init__(self, *args, **kwargs):
        super(BufferedHebb, self).__init__(*args, **kwargs)
        self.buffer_len = 30
        self.gamma = 0.9
        self.buffer = {'in' : deque([]),'activ': deque([]), 'R': deque([])}
        self.t = 0

    def step(self, inputs, activities, reward=None):
        weight_update = 0.0
        if self.t >= self.buffer_len:
            cumm_reward = np.sum([rew*self.gamma**k for k, rew in enumerate(self.buffer['R'])])
            weight_update = super().step(self.buffer['in'][0].copy(), self.buffer['activ'][0].copy(), reward=cumm_reward)
        self.buffer['in'].append(inputs)
        self.buffer['activ'].append(activities)
        self.buffer['R'].append(reward)
        if self.t >= self.buffer_len:
            self.buffer['in'].popleft()
            self.buffer['activ'].popleft()
            self.buffer['R'].popleft()
        self.t += 1
        return weight_update

    def reset(self):
        self.t = 0
        self.buffer = {'in' : deque([]), 'activ': deque([]), 'R': deque([])}

# class HebbA2C(GeneralizedABCDHebbian):
#     def __init__(self, *args, **kwargs):
#         super(HebbA2C, self).__init__(*args, **kwargs)
#         self.critic_lr = 1e-3
#         self.gamma = 0.9
#         self.prev_state = None
#         self.prev_action = None
#         self.critic_weights = None

    
#     def step(self, outputs, state, actions, reward=None):
#         #* Compute value estimation
#         st_ac_vec = np.r_[state, actions]
#         estim_value = self.critic_weights.dot(st_ac_vec)
#         self.super().step(reward=estim_value)

#         # #* Update weights of critic
#         # td_err = reward + self.gamma * estim_value
#         # self.critic_weights += self.critic_lr * 

#     def reset(self):
#         pass
