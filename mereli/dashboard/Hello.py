import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import zmq



# def plot_graph( key=None):
#     robot_positions = np.array([
#          [0,0],
#          [1, 0],
#          [0,1],
#          [1, 1],
#      ])
#      fig, axes = plt.subplots(1, 2)
#      axes[0].hist(np.random.normal(1, 1, size=100), bins=20)
#      axes[1].scatter(robot_positions[:,0],robot_positions[:,1])

#      # columns = st.columns(3)
#      # with columns[1]:

#      return  st.pyplot(fig)

# Create socket and save it in session state
if 'socket' not in st.session_state.keys():
    st.session_state['socket'] = None 
if st.session_state['socket'] is None:
    context = zmq.Context()
    #  Socket to talk to server
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    st.session_state['socket'] = socket

def onclick_pause():
    socket = st.session_state['socket']
    socket.send(b"pause")
    message = socket.recv()
    print(message)


tmri = 0
def onclick_resume():
    socket = st.session_state['socket']
    socket.send(b"resume")
    message = socket.recv()
    print(message)
    tim = st.session_state['t']
    print(f'TMR={tim}')

class RepeatTimer(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


if 'data' not in st.session_state.keys():
    st.session_state['data'] = [] 
graph = st.line_chart(st.session_state['data'])
bar = st.progress(0)

def timer_callback(socket, st_sess):
    socket.send(b"step")
    message = socket.recv()
    # t = message.decode("utf-8").split('t=')[1]

    if 'data' in st_sess:
        st_sess['data'].append(np.random.randn(1))


if 't' not in st.session_state.keys():
    st.session_state['t'] = None 




st.markdown('''
# Welcome to mereli
---

            ''')

pause_button = st.button("⏸️ Pause Simulation", key='a', on_click=onclick_pause)
resume_button = st.button("⏸️ Resume Simulation", key='b', on_click=onclick_resume)
# if st.session_state.get('t') is not None:
#     print(st.session_state['t'])
#     bar.progress(st.session_state['t']/3000)

if st.session_state['socket'] is not None:
    socket = st.session_state['socket']
    tmr = RepeatTimer(1, timer_callback, [socket, st.session_state])
    tmr.daemon = True
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    add_script_run_ctx(tmr)
    tmr.start()

from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=1000)

