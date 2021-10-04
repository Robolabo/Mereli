import numpy as np

from mereli.objects import Robot
from mereli.sensors import Sensor
from mereli.register import sensor_registry


@sensor_registry(name='group_reward_sensor')
class GroupRewardSensor(Sensor):
    """ Sensor that measures the mean reward obtained by all the robots in the 
    vicinity of the sensor owner. The vicinity is established as a sphere of a radius 
    determinied by the ``range`` argument. The sensor simplifies and abstracts the underlying 
    use of communication technologies such as Bluetooth or Zigbee. An important note is that 
    the receiving robot is not aware of the neighboring robot that sent the reward because there is 
    no directionality awareness and rewards are averaged.

    :param float range: radius of the vicinity. Only rewards within this range can be perceived by 
        the robot. 
    
    """
    def __init__(self, *args, range=2.0, **kwargs):
        super(GroupRewardSensor, self).__init__(*args, **kwargs)
        self.range = range

    def step(self, neighborhood):
        """ Step method of th Group Reward Sensor. It collects the rewards of any robot within an 
        Euclidean distance lower that ``range`` and returns the mean reward of the vicinity.
        
        :param list neighborhood: list of world entities. This parameter is not used at all in this sensor, but it is 
            kept as a parameter because other sensors may need to use it.
        """
        rewards = []
        for obj in filter(lambda x: issubclass(type(x), Robot), neighborhood):
            if np.linspace(self.sensor_owner.position - obj.position) < self.range:
                if obj.reward is not None:
                    rewards.append(obj.reward)
        return np.mean(rewards)