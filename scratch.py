import numpy as np
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# program setup
X, y = vertical_data(samples=100, classes=3)
# program loop
