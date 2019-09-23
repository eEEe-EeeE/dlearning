import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import signal

def c_gen(Z, leads_num):
    for row in range(leads_num):
        Zd = np.delete(Z, row, 0)
        z = Z[row]
        yield np.linalg.inv(Zd.dot(Zd.T)).dot(Zd).dot(z)

def my_plot(y, Z):
    plt.subplot(2, 1, 1)
    plt.plot(Z[0][:2000])
    plt.subplot(2, 1, 2)
    plt.plot(y[0][:2000])
    plt.show()

def main():
    df = pd.read_csv("dataset/train/100.txt", sep=" ")
    leads_num = df.columns.size
    Z = df.values
    Z = Z.T
    C = c_gen(Z, leads_num)
    C = np.column_stack(tuple(C))
    C = np.row_stack((C, np.zeros(leads_num) - 1))
    U = Z + C.dot(Z)
    b, a = signal.butter(8, [2 * 40 / 500, 2 * 125 / 500], "bandpass")
    filtedZ = signal.filtfilt(b, a, Z)
    filtedU = signal.filtfilt(b, a, U)
    P = 2 * filtedU * filtedZ / (np.square(filtedU) + np.square(filtedZ))
    y = Z - (1 - P) * filtedZ
    my_plot(y, Z)


if __name__ == '__main__':
    main()
