import os




class BaseLogger:

    def __init__(self, filename, log_dir):
        self.filename = filename
        self.log_dir = log_dir
        self.path = None
        self.data = None

    def add(self):
        pass

    def empty(self):
        pass

class CSVLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
       pass 

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
