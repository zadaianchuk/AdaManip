from abc import abstractclassmethod
from envs.base_env import BaseEnv
from logging import Logger

class BaseManipulation :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        self.env = env
        self.cfg = cfg
        self.logger = logger

    @abstractclassmethod
    def collect_data(self, obs, eval=False) :

        pass