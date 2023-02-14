import sys
import zmq
import json


class DashboardConnection:

    def __init__(self):
        self.paused = True 
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

    def process(self, t, n_robots, data_logger):
        message = self.socket.recv()
        if message:
            if message == b'pause':
                res = 'Simulation stopped t={}'.format(t)
                self.socket.send(bytes(res, 'utf-8'))#'Simulation stopped t={}'.format(self.t))
                self.paused = True
            elif message == b'resume':
                self.paused = False
                self.socket.send(b'Simulation Resumed')
            elif message == b'end':
                self.socket.send(b'Simulation Ended')
                sys.exit(0)
            elif message == b'step':
                res = 'Simulation stopped t={}'.format(t)
                data_dict = data_logger.get_last_row()
                data_dict = {**data_dict, **{'t' : t, 'n' : n_robots}}
                data = json.dumps(data_dict)
                self.socket.send(bytes(data, 'utf-8'))#'Simulation stopped t={}'.format(self.t))
            elif message == b'data':
                self.socket.send(b'Data requested.')
                # self.socket.send_json(self.data_logger.data)
            else:
                self.socket.send(b'Message not understood.')
        return self.paused
