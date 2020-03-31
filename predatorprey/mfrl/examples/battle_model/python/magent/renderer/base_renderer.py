# Original Work Copyright © 2018-present, MAgents (https://github.com/geek-ai/MAgent). 
# All rights reserved.

from abc import ABCMeta, abstractmethod


class BaseRenderer:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def start(self, *args, **kwargs):
        pass
