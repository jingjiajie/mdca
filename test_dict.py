import time
import multiprocessing
from multiprocessing import shared_memory

import numpy as np

print(__name__)

class Wrapper:

    def __init__(self,shm):
        self.shm = shm

def apply_test(dict):
    try:
        print("I'm child process!!!")
        # local_dict = {}
        # for key in dict.keys():
        #     local_dict[key] = dict[key]
        # for i in range(0, 100000):
        #     local_dict['foo'] = 123
        # for key in local_dict.keys():
        #     dict[key] = local_dict[key]
        # shm = shared_memory.SharedMemory(name="parameter_data", create=True, size=1000000)
        # array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        for i in range(0, 5000):
            dict[f'foo{i}'] = {'bar': 123}
    except Exception as e:
        print("ERROR!!!", e)


if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        shared_dict = manager.dict()

        print('开始主进程。。。')
        start = time.time()
        # 使用线程池建立子进程
        pool = multiprocessing.Pool(1)
        print('开始子进程。。。')
        for i in range(1):
            pool.apply_async(func=apply_test, args=(shared_dict,))
        pool.close()
        pool.join()
        print('主进程结束，耗时 %s' % (time.time() - start))
        print('shared_dict: ', type(shared_dict['foo1']))
        # print('local_dict: ', local_dict)
