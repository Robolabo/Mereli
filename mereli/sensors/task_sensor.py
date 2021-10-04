import numpy as np
from mereli.sensors import Sensor
from mereli.register import sensor_registry


@sensor_registry(name='task_sensor')
class TaskSensor(Sensor):
    """ 
    """
    def __init__(self, *args, **kwargs):
        super(TaskSensor, self).__init__(*args, **kwargs)

    def step(self, hierarchy):
        task_scheduler = [obj for obj in hierarchy if type(obj).__name__ == 'TaskScheduler'][0]
        tsk = task_scheduler.current_task / (task_scheduler.num_tasks - 1)
        return np.array([tsk])