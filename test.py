import threading
import time

import numpy as np

a: np.ndarray = np.array(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]), dtype=bool)
b: np.ndarray = np.array(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]), dtype=bool)

m: np.ndarray = np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1])

n: np.ndarray = np.zeros(5000000, dtype=bool)

def calc():
    for i in range(0, 1000):
        # c = a[m]
        # nonzero_count = np.count_nonzero(c)

        # sample_times = 100
        # length: int = len(a)
        # total_sample_count: int = int(length / 100)
        # step: int = int(length/sample_times)
        # sample_per_step: int = int(total_sample_count / sample_times)
        # nonzero_count: int = 0
        # for i in range(0, sample_times):
        #     low: int = int(i * step)
        #     high: int = low + sample_per_step
        #     c = a[low: high]
        #     nonzero_count += np.count_nonzero(c)

        # windows = np.lib.stride_tricks.as_strided(a, shape=(500000,), strides=(10 * 1,), writeable=False)
        # np.count_nonzero(windows)

        # c = a & b
        # d = c & a

        global n
        n[:] = 0
        n &= a
        n &= b

        x: threading.local = threading.local()


if __name__ == '__main__':
    start = time.time()
    all_threads = []
    for i in range(0, 1):
        t = threading.Thread(target=calc)
        t.start()
        all_threads.append(t)

    for t in all_threads:
        t.join()

    print("cost: ", (time.time() - start))