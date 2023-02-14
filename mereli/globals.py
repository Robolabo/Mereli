import logging


sim_status = 'idle'

class Globals:
    def __init__(self):
        self._EVAL = False
        self._DEBUG = False
        self._RENDER = True
        self._INFO = True
        self._LOG = False
        self._INTERACTIVE = False
        self.log_info = {} 

    def set_eval_state(self, new_state):
        self._EVAL = new_state
    
    def set_render_state(self, new_state):
        self._RENDER = new_state

    def set_log_state(self, new_state):
        self._LOG = new_state

    def set_debug_state(self, new_state):
        if new_state:
            # logging.basicConfig(level=logging.DEBUG)
            logging.getLogger().level = logging.DEBUG
            logging.getLogger().debug('Executing in DEBUG mode.')
        self._DEBUG = new_state

    def set_data_logging(self, path):
        self.log_info = {'path' : path}

    def set_states(self, render=True, eval=False, debug=False, log=False, info=False, interactive=False):
        self._EVAL = eval
        self._RENDER = render
        self._DEBUG = debug
        self._LOG = log
        self._INFO = info
        self._INTERACTIVE = interactive
        if debug:
            # logging.basicConfig(level=logging.DEBUG)
            logging.getLogger().level = logging.DEBUG
            logging.getLogger().debug('Executing in DEBUG mode.')
        elif info:
            logging.getLogger().level = logging.INFO
            logging.getLogger().info('Executing in VERBOSE mode.')

    @property
    def EVAL(self):
        return self._EVAL
    
    @property
    def RENDER(self):
        return self._RENDER

    @property
    def DEBUG(self):
        return self._DEBUG

    @property
    def INFO(self):
        return self._INFO or self._DEBUG

    @property
    def LOG(self):
        return self._LOG

    @property
    def INTERACTIVE(self):
        return self._INTERACTIVE


    @property
    def LOG_PATH(self):
        return self.log_info['path']

global_states = Globals()
