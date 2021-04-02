import heapq as hq
from collections import defaultdict 
import numpy as np


aa = np.arange(-10,10)
test = np.argmax(aa>5)
print(test)

test1 = np.array([[0,0], [0,0], [0,0]])
test2 = np.array([[4328.05, 3554.37], [3583.6, 2600.05], [4208.99, 3465.83]])

def angle_rowwise_v2(A, B):
    p1 = np.einsum('ij,ij->i',A,B)
    p2 = np.einsum('ij,ij->i',A,A)
    p3 = np.einsum('ij,ij->i',B,B)
    p4 = p1 / np.sqrt(p2*p3)
    return np.arccos(np.clip(p4,-1.0,1.0))

result = angle_rowwise_v2(test1,test2)

print(result*180/np.pi)

#i = np.where(result==np.min(result[np.nonzero(result)]))[0][0]
#print(i)


print("output ", np.linalg.norm(test2- test1, axis=1))