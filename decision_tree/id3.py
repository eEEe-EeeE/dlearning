import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def info_entropy(P):
    return -np.dot(P, np.log2(P))


