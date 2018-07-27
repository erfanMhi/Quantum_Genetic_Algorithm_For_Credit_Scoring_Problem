from multiprocessing import Pool, TimeoutError
import time
import os
import contextlib

def f(x,y=3):
    print('hi')
    for i in range(10) :
        x=x+1
    print('bye')
    return (x*x,)

if __name__ == '__main__':
    # start 4 worker processes
    with contextlib.closing(Pool(processes=10)) as pool:
        res = pool.map_async(f, (i for i in range(10))) 
                    # runs in *only* one process
                # print(res.get())             # prints "400"
        results_list = res.get()
        print(results_list)           

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")