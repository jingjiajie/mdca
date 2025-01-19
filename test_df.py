import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import pandas as pd

print(__name__)

def apply_test(shm):
    try:
        print("I'm child process!!!")
        ndarr = np.ndarray((1000, 31), dtype='object', buffer=shm.buf)
        print(ndarr[0])
    except Exception as e:
        print("ERROR!!!", e)


if __name__ == '__main__':
    data_df: pd.DataFrame = pd.read_csv('data/flights/flights.csv')
    data_df = data_df.head(1000)
    data = data_df.values
    l = []
    for i in range(0, len(data)):
        r = []
        for j in range(0, len(data[i])):
            r.append(data[i][j])
        l.append(r)
    shared_list = shared_memory.ShareableList(sequence=l)
    pool = multiprocessing.Pool(1)
    print('开始子进程。。。')
    for i in range(1):
        pool.apply_async(func=apply_test, args=(shared_list,))
    pool.close()
    pool.join()

    pass
