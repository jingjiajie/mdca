import threading
import time

import numpy as np
from bitarray import bitarray

a: np.ndarray = np.array(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]), dtype=np.uint8)
b: np.ndarray = np.array(np.random.choice([0, 1], size=5000000, p=[1 - 0.1, 0.1]), dtype=np.uint8)

c: np.ndarray = np.array(np.random.choice(5000000, size=1000000, replace=False), dtype=np.uint32)
d: np.ndarray = np.array(np.random.choice(5000000, size=1000000, replace=False), dtype=np.uint32)

e = bitarray(a.__iter__())
f = bitarray(b.__iter__())

def calc():
    for i in range(0, 1000):
        # np.count_nonzero(a)
        print(e.nbytes)

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