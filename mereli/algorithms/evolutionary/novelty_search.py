from collections import deque
import numpy as np

class NoveltySearch:
    def __init__(self, k=10, max_buffer_size=10000):
        self.k = k
        self.max_buffer_size = max_buffer_size
        self.buffer = deque([])
    
    def update(self, new_value):
        if self.buffer_size + 1 >= self.max_buffer_size:
            self.buffer.popleft()
        self.buffer.append(new_value)

    def novelty_metric(self, value):
        k_nearest = sorted([np.abs(value - x) for x in self.buffer])[:self.k]
        return np.mean(k_nearest) / max(self.buffer)

    def reset(self):
        self.buffer = deque([])

    @property
    def buffer_size(self):
        return len(self.buffer)