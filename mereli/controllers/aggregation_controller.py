import numpy as np
from mereli.controllers import RobotController
from mereli.register import controller_registry

import random

class Perlin:
    def __init__(self):
        self.gradients = []
        self.lowerBound = 0


    def valueAt(self, t):
        if(t<self.lowerBound):
            print("ERROR: Input parameter out of bounds!")
            return
        # Add to gradients until it covers t
        while t >= len(self.gradients)-1+self.lowerBound:
            self.gradients.append(random.uniform(-1, 1))

        discarded = int(self.lowerBound) # getting number of gradients that have been discarded
        # Compute products between surrounding gradients and distances from them
        d1 = (t-t//1)
        d2 = d1-1
        a1 = self.gradients[(int)(t//1)-discarded]*d1
        a2 = self.gradients[(int)(t//1+1)-discarded]*d2

        amt = self.__ease(d1)

        return self.__lerp(a1,a2,amt)

    def discard(self, amount):
        gradientsToDiscard = int(amount+self.lowerBound%1)
        self.gradients = self.gradients[gradientsToDiscard:]
        self.lowerBound += amount

    def __ease(self, x):
        return 6*x**5-15*x**4+10*x**3


    def __lerp(self, start, stop, amt):
        return amt*(stop-start)+start

# import matplotlib.pyplot as plt 
# per = Perlin()
# vals = [per.valueAt(t) for t in np.linspace(0,10,10000)]
# plt.plot(vals)
# plt.show()
# import pdb; pdb.set_trace()




@controller_registry(name='perlin_walk')
class PerlinWalk(RobotController):
    """
    """
    def __init__(self, *args, **kwargs):
        super(PerlinWalk, self).__init__(*args, **kwargs)
        self.perlin_generator = Perlin()
        self.sensitivity = 0.19
        self.t = 0

    def step(self, state, reward=0.0):
        """
        """
        st_ds = state['distance_sensor']
        if not any(st_ds > self.sensitivity):
            delta_ori = self.perlin_generator.valueAt(self.t)
            wh_speed = 0.2 * (np.abs(delta_ori) * 2 - 1)
            if delta_ori < 0:
                action = np.array([1., wh_speed])
            else: 
                action = np.array([wh_speed, 1.])
        else:
            if any(st_ds[[0,1]] > self.sensitivity):
                # print('Turn Left')
                action = np.array([1., -1])
            elif any(st_ds[[6,7]] > self.sensitivity):
                # print('Turn Right')
                action = np.array([-1, 1.])
            else:
                # print('GO straight over')
                action = np.array([1., 1.])
        self.t += 0.1
        return {'joint_velocity_actuator' : action}

@controller_registry(name='follow_robot')
class FollowRobot(RobotController):
    """
    """
    def __init__(self, *args, **kwargs):
        super(FollowRobot, self).__init__(*args, **kwargs)
        self.perlin_walk = PerlinWalk(*args, **kwargs)
        self.aux = np.random.random() < 0.3
        print(self.aux)

    def step(self, state, reward=0.0):
        """
        """
        st_ds = state['distance_sensor']
        ir_rx = state['IR_receiver']['msg']
        if not any(ir_rx != 0.0) or self.aux:
            return {**self.perlin_walk.step(state), 'IR_transmitter' : np.array([1.])}
        elif any(st_ds[[0,1]] > 0.6):
            return {'joint_velocity_actuator' : np.array([0., 0.]), 'IR_transmitter' : np.array([self.aux])}
        else:
            if any(ir_rx[[0,7]] != 0.0):
                action = np.array([1., 1.])
            elif any(ir_rx[[1,2,3]] != 0.0):
                action = np.array([-1., 1.])
            elif any(ir_rx[[6,5,4]] != 0.0):
                action = np.array([1., -1.])
            else:
                action = np.array([0., 0.])
            return {'joint_velocity_actuator' : action, 'IR_transmitter' : np.array([self.aux])}




@controller_registry(name='stay')
class StayController(RobotController):
    def __init__(self, *args, **kwargs):
        super(StayController, self).__init__(*args, **kwargs)
    def step(*args, **kwargs):
        return {'joint_velocity_actuator' : np.array([0.,0.]), 'IR_transmitter' : np.array([1.])}

@controller_registry(name='aggregation')
class AggregationController(RobotController):
    """ Controller devoted to the obstacle avoidance task. This means that the controller 
    will read from the distance sensor, process the measurements and return joint velocity 
    actions required to avoid colliding with any other tangible entity. This controller is 
    currently hard coded for the Epuck robot.

    :param float sensitivity: value in [0, 1] that defines the threshold in the distance sensor reading 
        to interpret an obstacle detection. 
    """
    def __init__(self, *args, sensitivity=0.1, avoid_collide=True, **kwargs):
        super(AggregationController, self).__init__(*args, **kwargs)
        self.perlin_walk = PerlinWalk(*args, **kwargs)
        self.sensitivity = sensitivity
        self.avoid_collide = avoid_collide

    def step(self, state, reward=0.0):
        """ Method to execute once the controller program. It reads the current distance sensor measurement, 
        and plans the action as follows:

        .. code-block:: 
        
            IF DS[0] > SENSITIVITY OR DS[1] > SENSITIVITY THEN
                TURN LEFT
            ELSE IF DS[6] > SENSITIVITY OR DS[7] > SENSITIVITY THEN
                TURN RIGHT
            ELSE THEN
                GO STRAIGHT OVER
        
        :param dict state: state with the sensor reading. The dict maps the reference name of the sensor to 
            the sensor np.ndarray reading.
        :param float reward: reward (if any). Not used in this controller.
        """
        st_ds = state['distance_sensor']
        ir_rx = state['IR_receiver']['msg']
        max_val = np.max(st_ds)
        max_dir = np.argmax(st_ds)
        min_dir = np.argmin(st_ds)
        if not any(ir_rx != 0.0) or ir_rx[max_dir] == 0.:
            return self.perlin_walk.step(state)
        else:
            num_ok = np.logical_and(0.2 <= st_ds, st_ds <= 0.7)
            if np.sum(num_ok) >= 2: 
                action = np.array([0,0])
            else:
                return self.perlin_walk.step(state)
            # import pdb; pdb.set_trace()
            # if max_val > 0.6: # Too close, avoid
            #     if min_dir in [7,6,5,4]:
            #         action = np.array([-1, 1.]) # Turn right
            #     else:
            #         action = np.array([1., -1]) # Turn left
            # else: # Aggregate
            #     if 0.4 < max_val < 0.6:
            #         action = np.array([0,0])
            #     else:
            #         if max_dir in [7,0]:
            #             action = np.array([1, 1.]) # Straight
            #         elif max_dir in [1,2,3]:
            #             action = np.array([1., -1]) # Turn left
            #         else:
            #             action = np.array([-1, 1.]) # Turn right
        return {'joint_velocity_actuator' : action, 'IR_transmitter' : np.array([1.])}