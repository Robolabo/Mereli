import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import threading
import json
import zmq
import holoviews as hv
from holoviews.streams import Stream, param
# import param
import panel as pn

css = '''
.widget-button .bk-btn-group button {
  font-size: 30pt;
}
'''
pn.extension(sizing_mode = 'stretch_width',raw_css=[css])



class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

context = zmq.Context()
#  Socket to talk to server
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

def onclick_pause(event):
    socket.send(b"pause")
    message = socket.recv()
    print(message)
    print('PAUSED')

def onclick_resume(event):
    socket.send(b"resume")
    message = socket.recv()
    print(message)
    print('RESUME')

def onclick_end(event):
    socket.send(b"end")
    message = socket.recv()
    sys.exit(0)

freq = pn.widgets.FloatSlider(name="Frequency", start=0, end=10, value=2)
phase = pn.widgets.FloatSlider(name="Phase", start=0, end=np.pi)
robot_selector = pn.widgets.Select(name='Select', options={'Robot 1': 0})
template = pn.template.FastGridTemplate(site="Mereli", title="Mereli Dashboard", 
                                        sidebar=[pn.pane.Markdown("## Settings"), freq, phase,
                                                 robot_selector])



class BaseClass(param.Parameterized):
    t = param.Integer(0, bounds=(0, 100000))
    n = param.Integer(0, bounds=(0, 100))
    st = param.Number(0.2, bounds=(-2, 2))
    lmarks_x = param.List([0.], item_type=float)
    lmarks_y = param.List([0.], item_type=float)
    positions_x = param.List([0.], item_type=float)
    positions_y = param.List([0.], item_type=float)
    robots = {}
    time_series = param.List([0.], item_type=float)
    neighbors = []

    def decode(self, msg):
        data_dict = json.loads(msg)
        self.t = data_dict['t']
        self.n = data_dict['n']
        if len(robot_selector.options) != self.n:
            robot_selector.options = {f'Robot {ii}': ii for ii in range(self.n)}
        if len(data_dict) > 2:
            if len(self.lmarks_x) < 2:
                lmarks = np.stack(data_dict['virtual_space@landmarks']) 
                self.lmarks_x = lmarks[:, 0].tolist()
                self.lmarks_y = lmarks[:, 1].tolist()
    
            neighbors = data_dict[f'robotA_{robot_selector.value}@neighbor_names']
            neighbors = neighbors[0] if isinstance(neighbors[0], list) else neighbors
            try:
                self.neighbors = [*map(lambda x: int(x.split('_')[1]), neighbors)]
            except:
                __import__('pdb').set_trace()
            self.positions_x = []
            self.positions_y = []
            for i in range(data_dict['n']):
               pos_v = data_dict[f'robotA_{i}@position']
               self.positions_x.append(pos_v[0])
               self.positions_y.append(pos_v[1])
               if str(i) not in self.robots:
                    self.robots[str(i)] = {'x' :param.Number(0, bounds=(-1,1)), 'y': param.Number(0, bounds=(-1,1))} 
               state = data_dict[f'robotA_{i}:virtual_particle@state']
               self.robots[str(i)]['x'] = state[0]
               self.robots[str(i)]['y'] = state[1]


class TimeSeries(param.Parameterized):
    time_series = None #param.Array(np.array([0.])) 
     
    def decode(self, msg):
        data_dict = json.loads(msg)
        if len(data_dict) > 2:
            focus = robot_selector.value
            data = data_dict[f'robotA_{focus}:virtual_particle:controller@voltages']
            if self.time_series is None:
                self.time_series = np.array(data).reshape(1, -1)
            else:
                self.time_series = np.vstack((self.time_series, data))
            if self.time_series.shape[0] > 200:
                self.time_series = self.time_series[1:]


data = BaseClass()
ann_voltage_ts = TimeSeries()

def timer_callback(socket, data):
    socket.send(b"step")
    msg= socket.recv().decode("utf-8")
    data.decode(msg)
    ann_voltage_ts.decode(msg)



header  = pn.pane.Markdown("""
# Welcome to Mereli
---
""", width=500)


# BUTTONS
pause_btn = pn.widgets.Button(name='\u23f8',  width=70, height=70, button_type='primary', css_classes=["widget-button"])
resume_btn = pn.widgets.Button(name='\u25b6', width=70,height=70,button_type='primary', css_classes=["widget-button"])
end_btn = pn.widgets.Button(name='\u23f9', width=70,height=70,button_type='primary', css_classes=["widget-button"])
step_btn = pn.widgets.Button(name='\u23ed', width=70,height=70,button_type='primary', css_classes=["widget-button"])
step_back_btn = pn.widgets.Button(name='\u23ee', width=70,height=70,button_type='primary', css_classes=["widget-button"])
row = pn.Row(step_back_btn, resume_btn, pause_btn, end_btn, step_btn)
resume_btn.on_click(onclick_resume)
pause_btn.on_click(onclick_pause)
end_btn.on_click(onclick_end)




progress = pn.indicators.Progress(name='Progress', value=0,bar_color='warning', width=700, height=20)
spacer = pn.Spacer(background='#f7f7f7',    margin=0)




def update_bar(event):
    progress.value = int(100 * data.t / 3000)

data.param.watch(update_bar, 't')



def comm_space(state_x, state_y, lmarks_x, lmarks_y):
    focus_bot = robot_selector.value
    st_X = np.array(state_x) 
    st_Y = np.array(state_y)
    non_neighs = [ii for ii in range(data.n) if ii != focus_bot and ii not in data.neighbors]
    h1 = hv.Points((st_X[non_neighs], st_Y[non_neighs])).opts(color='k', marker='o',  size=10)
    h0 = hv.Points((state_x[focus_bot],state_y[focus_bot])).opts(color='b', marker='o',size=10)
    h2 = hv.Points((lmarks_x, lmarks_y)).opts(color='r', marker='diamond', size=20)
    h3 = hv.Points((st_X[data.neighbors],st_Y[data.neighbors])).opts(color='orange', marker='o', size=10,)
    return (h2* h1*h0*h3).opts(xlim=(-1, 1), ylim=(-1,1), shared_axes=False, toolbar=None)

def phy_space(pos_x, pos_y):
    focus_bot = robot_selector.value
    H, W = 4, 4
    walls = hv.Rectangles([(-W/2, -H/2, W/2-0.1, -H/2+0.1),
                          (-W/2, -H/2+0.1, -W/2+0.1, H/2-0.1),
                          (-W/2, H/2, W/2-0.1, H/2-0.1),
                          (W/2-.1, H/2, W/2, -H/2),])
    h1 = hv.Points((pos_x, pos_y)).opts(xlim=(-W/2, W/2), ylim=(-H/2, H/2), color='k', marker='o', size=10, toolbar=None)
    h0 = hv.Points((pos_x[focus_bot], pos_y[focus_bot])).opts(xlim=(-W/2, W/2), ylim=(-H/2, H/2), color='b', marker='o', size=10, toolbar=None)
    return hv.Overlay([walls, h1, h0])

def plot_ori(oris):
    return hv.Curve(oris).opts(color='k', xlim=(0, 200), ylim=(0, 2*np.pi), axiswise=True, toolbar=None) 


def plot_time_series(time_series):
        return hv.NdOverlay({f'Neuron {i}' : hv.Curve(ts).opts(xlim=(0, 200), ylim=(0, 2*np.pi), axiswise=True, toolbar=None, shared_axes=False) 
                 for i, ts in enumerate(time_series.T)})



st_stream = Stream.define('states', 
                state_x=param.List([0.], item_type=float), 
                state_y=param.List([0.], item_type=float),
                lmarks_x=param.List([0.], item_type=float),
                lmarks_y=param.List([0.], item_type=float))
comm_dmap = hv.DynamicMap(comm_space, streams=[st_stream()])
ppp = pn.panel(comm_dmap, width=400, height=400)

phy_stream = Stream.define('postions',pos_x=param.List([0.], item_type=float),pos_y=param.List([0.], item_type=float))
phy_plot = pn.panel(hv.DynamicMap(phy_space, streams=[phy_stream()]), width=400, height=400)

ts_stream = Stream.define('time_series', time_series=param.Array(np.array([[0.]])))
# ts_plot = pn.panel(hv.DynamicMap(plot_ori, streams=[ts_stream()]), width=400, height=400)
ts_plot = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=400, height=400)

ts1 = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=800, height=400)
# ts2 = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=400, height=400)
# ts3 = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=400, height=400)
# ts4 = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=400, height=400)
# ts5 = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=400, height=400)
# ts6 = pn.panel(hv.DynamicMap(plot_time_series, streams=[ts_stream()]), width=400, height=400)

radio_group = pn.widgets.RadioButtonGroup(name='Monitored Signals', options=['Sensors', 'Neurons', 'Communication'], 
                                          button_type='primary', width=899, css_classes=["widget-button"])

def update_plot(event):
    if len(data.robots) == 0:
        return
    new_data = np.stack([np.r_[val['x'], val['y']] for val in data.robots.values()])
    ppp.object.event(state_x=new_data[:,0].tolist(), state_y=new_data[:,1].tolist(), lmarks_x=data.lmarks_x, lmarks_y=data.lmarks_y)
    phy_plot.object.event(pos_x=data.positions_x, pos_y=data.positions_y) 
    if radio_group.value == ' Sensors':
        pass
    # if data.t % 5  == 0:
    ts1.object.event(time_series=ann_voltage_ts.time_series)
    #     ts2.object.event(time_series=ann_voltage_ts.time_series)
        # ts3.object.event(time_series=ann_voltage_ts.time_series)
        # ts4.object.event(time_series=ann_voltage_ts.time_series)
        # ts5.object.event(time_series=ann_voltage_ts.time_series)
        # ts6.object.event(time_series=ann_voltage_ts.time_series)

data.param.watch(update_plot, 't')



template.main[:6, :] = pn.Column(pn.Row(spacer, header, spacer), 
                                 pn.Row(spacer,pn.pane.Markdown("## **Simulation Progress**", width=220),spacer),
                                 pn.Row(spacer,progress,spacer),
                                 pn.Row(spacer,row,spacer),
                                 pn.Row(spacer, ppp, phy_plot, spacer),
                                 spacer,
                                 pn.Row(spacer,radio_group,spacer),
                                 spacer,
                                 ts1
                                 # pn.Row(spacer, ts1, ts2, ts3, spacer),
                                 # pn.Row(spacer, ts4, ts5, ts6, spacer),
                                 ) 


template.servable()




tmr = RepeatTimer(0.05, timer_callback, [socket, data])
tmr.daemon = True
tmr.start()
