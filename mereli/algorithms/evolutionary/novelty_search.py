from collections import deque
import numpy as np

class NoveltySearch:
    def __init__(self, k=20, max_buffer_size=20000, variable='eval_time', weight=.5):
        self.k = k
        self.max_buffer_size = max_buffer_size
        self.behavior_var = variable
        self.weight = weight
        self.buffer = deque([])
    
    def update(self, new_value):
        if self.buffer_size + 1 >= self.max_buffer_size:
            self.buffer.popleft()
        self.buffer.append(new_value[self.behavior_var])

    def novelty_metric(self, value):
        val = value[self.behavior_var]
        k_nearest = sorted([np.linalg.norm(val - x) for x in self.buffer])[:self.k]
        if self.behavior_var == 'eval_time':
            return np.mean(k_nearest) / max(self.buffer)
        else:# position var
            return np.exp(np.mean(k_nearest))

    def reset(self):
        self.buffer = deque([])

    @property
    def buffer_size(self):
        return len(self.buffer)