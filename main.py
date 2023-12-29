import SafeTraversal
import numpy as np

env = SafeTraversal.LavaPitRoom()
env.train(1000)
env.aiPlay(10)