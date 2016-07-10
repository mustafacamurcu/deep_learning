import numpy as np
import tensorflow as tf

class ReplayMemory(object):

    def __init__(self, length, memory):
        self.length = length
        self.memory = memory

    def append(self, transition):
        self.memory.append(transition)
        assert len( self.memory ) <= MAX_SIZE
        else:
