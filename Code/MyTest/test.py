# 代码语法测试文件
import torch
import numpy as np
import matplotlib.pyplot as plt

import time
from tqdm import *

def test_tdqm():
    for i in tqdm(range(1000)):
        time.sleep(.01)    #进度条每0.1s前进一次，总时间为1000*0.1=100


def Test():
    flag = False
    out = 1 if flag else 0
    print(out)


def CurveTest():
    X = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    Y = np.sin(X)
    plt.plot(X, Y)
    plt.show()


if __name__ == "__main__":
    # Test()
    # CurveTest()
    test_tdqm()
