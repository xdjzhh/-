from sklearn import preprocessing
import numpy as np
data = [[0, 0,3], [0, 0,1], [1, 1,2], [1, 1,0]]
data1 = [[0, 0,2], [0, 0,0], [1, 1,1], [1, 1,0]]
# 1. 基于mean和std的标准化
scaler1 = preprocessing.OneHotEncoder().fit(data1)
scaler = preprocessing.OneHotEncoder().fit(data)
print(scaler)
print(scaler.transform(data).toarray())
print(scaler.transform(data1).toarray())
print(scaler1.transform(data1).toarray())

a = [[[1,3],[2,4]]]
a = np.asarray(a)
print(a.shape)


a=[1,2,3,4,5]
b = [4,5,6,7,8]
for i, j in zip(a,b):
    print(i,j)