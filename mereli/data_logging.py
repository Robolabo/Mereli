import os
import numpy as np


class BaseLogger:

    def __init__(self, path, filename):
        self.path = path 
        self.data = None
        self.filename = filename

    def add(self):
        pass

    def empty(self):
        pass

class PickleLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super(PickleLogger, self).__init__(*args, **kwargs)
        self.data = {}

    def add(self, data_item):
        for k, item in data_item.items():
            if not k in self.data:
                self.data[k] = [item] # May switch to deque or better struct.
            else:
                self.data[k].append(item)

    def save(self):
        pass

    def load(self):
        pass

    def empty(self):
        self.data = {} 


class CSVLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        super(CSVLogger, self).__init__(*args, **kwargs)
        self.data = []
        self.labels = []

        # if not os.path.isdir(self.path):
        #     os.mkdir(self.path)

    def set_labels(self, labels):
        self.labels = labels

    def add(self, data_item):
        self.data.append(data_item)

    def save(self):
        np.savetxt(os.path.join(self.path, self.filename), self.data)

    def empty(self):
        pass


