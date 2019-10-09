import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import os
from multiprocessing import Process, Pool


def c_gen(Z, leads_num):
    for row in range(leads_num):
        Zd = np.delete(Z, row, 0)
        z = Z[row]
        yield np.linalg.inv(Zd.dot(Zd.T)).dot(Zd).dot(z)


# def my_plot(y, Z):
#     plt.subplot(2, 1, 1)
#     plt.plot(Z[0][:2000])
#     plt.subplot(2, 1, 2)
#     plt.plot(y[0][:2000])
#     plt.show()


def uncertainty_filter():
    df = pd.read_csv("dataset/train/100.txt", sep=" ")
    leads_num = df.columns.size
    # original data
    Z = df.values
    Z = Z.T
    # coefficient matrix
    C = c_gen(Z, leads_num)
    C = np.column_stack(tuple(C))
    C = np.row_stack((C, np.zeros(leads_num) - 1))
    # predict data
    U = Z + C.dot(Z)
    b, a = signal.butter(8, [2 * 40 / 500, 2 * 125 / 500], "bandpass")
    filtedZ = signal.filtfilt(b, a, Z)
    filtedU = signal.filtfilt(b, a, U)
    P = 2 * filtedU * filtedZ / (np.square(filtedU) + np.square(filtedZ))
    y = Z - (1 - P) * filtedZ
    # my_plot(y, Z)


# n : 降采样数据间隔
def signal_matrix(n, in_path, out_file):
    path = Path(in_path)
    txt_list = sorted(path.iterdir(), key=lambda e: int(e.name.strip(".txt")))
    id_list = []
    data_list = []
    for txt in txt_list:
        id_list.append(txt.name)
        data = np.loadtxt(txt.absolute(), skiprows=1)
        data = data[::n].T
        data = data.reshape(data.size)
        data_list.append(data)
    df = np.array(data_list)
    df = pd.DataFrame(df, index=id_list)
    df.to_csv(out_file, header=False)


def main():
    signal_matrix(4, "dataset/testB_noDup_rename", "output/signal_matrix_test_B.csv")


if __name__ == '__main__':
    main()
