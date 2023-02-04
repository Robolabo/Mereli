import os
import pickle
import numpy as np
from mereli.globals import global_states

class DataLogger:
    def __init__(self):
        self.path = global_states.log_info['path'] 
        self.data = {}
        self.info = []
        self.target_object = None 

    def configure(self, target_object, info):
        self.target_object = target_object
        self.info = info
        
        
        for key in info:
            path, asset, time = self.decode_variable(key) 
            if path[0] in self.target_object.groups:
                new_path_items = self.target_object.groups[path[0]]
                for item  in new_path_items:
                    new_path = [item]
                    if len(path) > 1:
                        new_path += path[1:]
                    entry = ':'.join(new_path) + '@' + asset 
                    self.data[entry] = []
            else: 
                self.data[key] = []

    def decode_variable(self, query):
        path, asset = tuple(query.split('@'))
        path_items = path.split(':')
        time = None
        if '?t=' in asset:
            asset, time = asset.split('?t=')
        return path_items, asset, time
        
        
    def update(self):
        for variable in self.data:
            path_items, asset, time = self.decode_variable(variable)
            aux_pointer = self.target_object
            if path_items[0] in self.target_object.hierarchy:
                aux_pointer = aux_pointer.hierarchy[path_items[0]]
                path_items.pop(0)
            
            for path_item in path_items:
                if path_item in self.target_object.hierarchy:
                    aux_pointer = getattr(aux_pointer, path_item)
                else:
                    aux_pointer = getattr(aux_pointer, path_item)
            data = getattr(aux_pointer, asset)
            try:
                if len(self.data[variable]) == 0:
                    self.data[variable] = data if isinstance(data, np.ndarray) else np.array([data])
                else:
                    self.data[variable] = np.vstack((self.data[variable], data)) 
            except:
                __import__('pdb').set_trace()

    def save_pickle(self):
        save_path = self.path + '.pickle'
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)


    def save_csv(self):
        pass

    def reset(self):
        for key in self.data:
            self.data[key] = []

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


