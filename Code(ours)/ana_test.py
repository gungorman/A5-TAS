import numpy as np
import math 

def fun(x):
    return (x-4)**2+2

a, b = -1, 9
alpha, beta=0.3, 0.7
N_max=100
for i in range(N_max):
    x_l=a+alpha*(b-a)
    x_r=a+beta*(b-a)
    if fun(x_l)<fun(x_r):
        b=x_r
    else:
        a=x_l

print(a, b)