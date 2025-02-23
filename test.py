import threading
import time

import numpy as np
from bitarray import bitarray, util

a: np.ndarray = np.packbits(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]))
b: np.ndarray = np.packbits(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]))

c: np.ndarray = np.array(np.random.choice(5000000, size=50000, replace=False), dtype=np.uint32)
d: np.ndarray = np.array(np.random.choice(5000000, size=50000, replace=False), dtype=np.uint32)

g = np.array(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]), dtype=np.uint8)
h = np.array(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]), dtype=np.uint8)

c_list = c.tolist()
d_list = d.tolist()

e = bitarray(5000000)
# f = bitarray(5000000)

l = list(range(0, 100000))

def calc():
    for i in range(0, 1000):
        e1 = bitarray.copy(e)
        # e1 = bitarray(5000000)
        # e1.setall(1)
        pass
        # e[0] = 1
        # pass
        # e = bitarray(5000000)
        # s = e[::100]

        # s1 & s2
        # e & f

        # e = bitarray(a_list)
        # global e,f
        # e[c_list]=1
        # e[d_list]=1
        # e = e & f
        # np.intersect1d(c, d, assume_unique=True)
        # np.union1d(c,d)

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