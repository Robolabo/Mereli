import numpy as np
from shapely.geometry import Point
from spike_swarm_sim.objects import WorldObject
from spike_swarm_sim.register import sensors, actuators, world_object_registry

@world_object_registry(name='robot')
class Robot(WorldObject):
    """
    Base class for the robot world object.
    """
    def __init__(self, pos, *args, orientation=0, **kwargs):
        super(Robot, self).__init__(pos=pos, static=False, luminous=False,\
                        tangible=True, shape='circular', *args, **kwargs)
        self._init_theta = orientation
        # self.theta = 0 # initialized in reset      
        self.theta = orientation

        self.radius = 11
        self._food = False

        #* Initialize sensors and actuators according to controller requirements
        self.sensors = {k : s(self, **self.controller.enabled_sensors[k])\
                            for k, s in sensors.items()\
                            if k in self.controller.enabled_sensors.keys()}
        self.actuators = {k : k == 'wheel_actuator' and a(self.radius, **self.controller.enabled_actuators[k])\
                            or a(**self.controller.enabled_actuators[k])\
                            for k, a in actuators.items()\
                            if k in self.controller.enabled_actuators.keys()}

        #* Storage for actions selected by the controllers to be fed to actuators
        self.planned_actions = {k : [None] for k in actuators.keys()}

        #* Rendering colors (TO BE MOVED TO RENDER FILE IN THE FUTURE)
        self.colorA = 'black'
        self.colorB = 'black'
        self.color2 = ('skyblue3', 'green')[self.trainable]
        self.reset()

    
    def step(self, neighborhood, reward=None, perturbations=None):
        """
        Firstly steps all the sensors in order to perceive the environment.
        Secondly, the robot executes its controller in order to compute the
        actions based on the sensory information.
        Lasty, the actionas are stored as planned actions to be eventually executed.
        =====================
        - Args:
            neighborhood [list] -> list filled with the neighboring world objects.
            reward [float] -> reward to be fed to the controller update rules, if any.
            perturbations [list of PostProcessingPerturbation or None] -> perturbation to apply 
                        to the stimuli before controller step.
        - Returns:
            State and action tuple of the current timestep. Both of them are expressed as 
            a dict with the sensor/actuator name and the corresponding stimuli/action.
        =====================
        """
        #* Sense environment surroundings.
        state = self.perceive(neighborhood)
        #* Apply perturbations to stimuli 
        for pert in perturbations:
            state = pert(state, self)

        #* Obtain actions using controller.
        actions = self.controller.step(state, reward=reward)
        #* Plan actions for future execution
        self.plan_actions(actions)

        # #* Handle robot food pickup
        # if 'food_area_sensor' in state.keys() and bool(state['food_area_sensor'][0]):
        #     self.food = True
        # if 'nest_sensor' in state.keys() and bool(state['nest_sensor'][0]):
        #     self.food = False

        #* Render robot LED
        self.update_colors(state, actions)
        return state, actions

    def update_colors(self, state, action):
        colors = ['black', 'red', 'yellow', 'blue']
        if 'wireless_transmitter' in self.actuators.keys():
            for k, msg in enumerate(action['wireless_transmitter']['msg']):
                symbol = np.argmin([np.abs(sym - msg) for sym in [0, 0.33, 0.66, 1]])
                if k == 0:
                    self.colorA = colors[symbol]
                if k == 1:
                    self.colorB = colors[symbol]
    
        if 'led_actuator' in self.actuators.keys():
            self.color2 = ('green', 'white', 'red')[action['led_actuator']] #[actions['wireless_transmitter']['state']]#
      
    def plan_actions(self, actions):
        for actuator, action in actions.items():
            self.planned_actions[actuator] = (actuator == 'wheel_actuator')\
                    and [action, self.pos, self.theta]  or [action]

    def actuate(self):
        """
        Executes the previously planned actions in order to be processed in the world.
        =====================
        - Args: None
        - Returns: None
        =====================
        """
        for actuator_name, actuator in self.actuators.items():
            # if actuator_name not in self.planned_actions:
            #     raise Exception('Error: Actuator does not have corresponding planned action.')
            actuator.step(*iter(self.planned_actions[actuator_name]))
        if 'wheel_actuator' in self.controller.enabled_actuators.keys() or 'target_pos_actuator' in self.controller.enabled_actuators.keys():
            self._move(validated=True)

    def perceive(self, neighborhood):
        """
        Computes the observed stimuli by steping each of the active sensors.
        =====================
        - Args:
            neighborhood [list] -> list filled with the neighboring world objects.
        -Returns:
            A dict with each sensor name as key and the sensor readings as value.
        =====================
        """
        return {sensor_name : sensor.step(neighborhood)\
                for sensor_name, sensor in self.sensors.items()}

    def _move(self, validated=False):
        """
        
        =====================
        - Args:
            validated [bool] -> flag indicating if the planned movement is valid 
                        (for example with no collisions). 
        - Returns: None
        =====================
        """
        self.pos += self.actuators['wheel_actuator'].delta_pos.astype(float) * float(validated)
        self.theta += self.actuators['wheel_actuator'].delta_theta * float(validated)
        # control angle range in (-pi,pi]
        self.theta = self.theta % (2*np.pi) #(self.theta, self.theta + 2*np.pi)[self.theta < 0]
        self.actuators['wheel_actuator'].delta_pos = np.zeros(2)
        self.actuators['wheel_actuator'].delta_theta = 0.0


    def reset(self):
        """
        Resets the robot dynamics, sensors, actuators and controller. Position and orientation 
        can be randomly initialized or fixed. In the former case a seed can be specified.
        =====================
        - Args:
            seed [int] -> seed for random intialization.
        - Returns: None
        =====================
        """
        self.delta_pos = np.zeros(2)
        self.delta_theta = 0.0
        self._food = False
        #* Reset Controller
        if self.controller is not None:
            self.controller.reset()
        #* Reset Actuators
        for actuator in self.actuators.values():
            if hasattr(actuator, 'reset'):
                actuator.reset()
        #* Reset Sensors
        for sensor in self.sensors.values():
            if hasattr(sensor, 'reset'):
                sensor.reset()

    @property
    def food(self):
        """Getter for the food attribute. It is a boolean attribute active if the robot stores food.
        """
        return self._food

    @food.setter
    def food(self, hasfood):
        """Setter for the food attribute. It is a boolean attribute active if the robot stores food.
        """
        self._food = hasfood

    @property
    def bounding_box(self):
        return Point(self.pos[0], self.pos[1]).buffer(self.radius).boundary
    
    def intersect(self, g):
        inters = self.bounding_box.intersection(g)
        if not inters: return []
        if isinstance(inters, Point):
            return np.array(inters.coords)
        else:
            return [np.array(v.coords[0]) for v in inters.geoms]
        
    def initialize_render(self, canvas):
        x, y = tuple(self.pos)
        contour_id = canvas.create_oval(x-self.radius-2, y-self.radius-2,\
                x + self.radius+2, y + self.radius+2, fill=self.color2)
        # body_id = canvas.create_oval(x-self.radius, y-self.radius,\
        #         x + self.radius, y + self.radius, fill=self.color)
        bodyA_id = canvas.create_arc(x-self.radius, y-self.radius,\
                x + self.radius, y + self.radius, start=np.degrees(self.theta), extent=180, fill="black")
        bodyB_id = canvas.create_arc(x-self.radius, y-self.radius,\
                x + self.radius, y + self.radius, start=np.degrees(self.theta)+180, extent=180, fill="black")
        orient_id = canvas.create_line(x, y,\
                x + self.radius * 2 * np.cos(self.theta),\
                y + self.radius * 2 * np.sin(self.theta),\
                fill='black', width=2)
        self.render_dict = {
            'contour' : contour_id,
            'bodyA' : bodyA_id,
            'bodyB' : bodyB_id,
            'orient' : orient_id,
        }
        return canvas
    
    def render(self, canvas):
        """
        Renders the robot in a 2D tkinter canvas.
        """
        x, y = tuple(self.pos)
        canvas.coords(self.render_dict['contour'],\
                x-self.radius, y-self.radius,\
                x + self.radius, y + self.radius)
        canvas.coords(self.render_dict['bodyA'],\
                x-self.radius+3, y-self.radius+3,\
                x + self.radius-3, y + self.radius-3)
        canvas.coords(self.render_dict['bodyB'],\
                x-self.radius+3, y-self.radius+3,\
                x + self.radius-3, y + self.radius-3)
        canvas.itemconfig(self.render_dict['contour'], fill=self.color2)
        canvas.itemconfig(self.render_dict['bodyA'], start=0, extent=180, fill=self.colorA)
        canvas.itemconfig(self.render_dict['bodyB'], start=180, extent=180, fill=self.colorB)
        canvas.coords(self.render_dict['orient'], x, y,\
                x + self.radius * 2 * np.cos(self.theta),\
                y + self.radius * 2 * np.sin(self.theta))
        return canvas