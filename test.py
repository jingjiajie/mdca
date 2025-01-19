import time
import multiprocessing
from multiprocessing import shared_memory

import numpy as np

print(__name__)

def apply_test(arr):
    try:
        print("I'm child process!!!")
        # arr = np.ndarray(5, dtype=np.int64, buffer=shm.buf)
        for i in range(0, 100000000):
            arr[1] = 123
    except Exception as e:
        print("ERROR!!!", e)


if __name__ == '__main__':
    data = np.array([1, 2, 3, 4, 5])
    shm = shared_memory.SharedMemory(name="parameter_data", create=True, size=data.nbytes)
    array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    array[:] = data

    local_array = np.ndarray(5, dtype=np.int64)

    print('开始主进程。。。')
    start = time.time()
    # 使用线程池建立3个子进程
    pool = multiprocessing.Pool(1)
    print('开始子进程。。。')
    for i in range(1):
        pool.apply_async(func=apply_test, args=(array,))
    pool.close()
    pool.join()
    print('主进程结束，耗时 %s' % (time.time() - start))
    print('array: ', array)
    print('local_array: ', local_array)