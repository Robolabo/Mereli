import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt

def plot_graph(key=None):
    robot_positions = np.array([
         [0,0],
         [1, 0],
         [0,1],
         [1, 1],
     ])

     arr = np.random.normal(1, 1, size=100)
     fig, axes = plt.subplots(1, 2)
     axes[0].hist(arr, bins=20)
     axes[1].scatter(robot_positions[:,0],robot_positions[:,1])

     # columns = st.columns(3)
     # with columns[1]:

     return  st.pyplot(fig)

st.set_page_config(page_title="Sensors", page_icon="📈")

st.markdown("# Plotting Demo")

pause_button = st.button("⏸️ Pause Simulation")
resume_button = st.button("⏸️ Resume Simulation")
sim_bar = st.progress(10)

st.sidebar.header("Plotting Demo")
plot_sensors = plot_graph()

if pause_button:
    plot_sensors.empty()

if resume_button:
     plot_sensors = plot_graph()
