import pickle
import sys
import time

import matplotlib
import numpy as np
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt
#import matplotlib

from predict import (countDifferences, predict, hamming_distance, sort_train_labels_knn, model_selection_knn, classification_error,printt,erease,printt2828)


def load_data():
    PICKLE_FILE_PATH = 'train.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pickle.load(f)


def save_data(x):
    output = open('zapisuje_pociete1.pkl', 'wb')
    pickle.dump(x, output)
    output.close()

def save():


     with open('train.pkl', 'rb') as file:
         x, y = pickle.load(file)
         x, y = x[:40000], y[:40000]
     x = x >= 0.35
     np.savez_compressed('data_bool1',x=x,y=y)


def get_data():
    loaded = np.load('data_bool1.npz')
    x = loaded['x']
    y = loaded['y']
    x = x[:14000]
    y = y[:14000]
    return (x, y)

#(x_train, y_train) = get_data()
#print(y_train)
#save()


# validacyjne dostajesz na serwerze

print("hello")

data = load_data()
zz=data[1]
# print(data)
(x, y) = data
# print(x)
# print(y)
rep = (x, y)
# print(rep)

# liczba testów 60000

# print(data[1][:1000])
datatrainx = data[0][:1000]
#datatrainx= datatrainx.astype(float)
datavalx = data[0][57500:60000]
datatrainy = data[1][:1000]
datavaly = data[1][57500:60000]
#print(datatrainx)
"""
for i in range(0,800):
    for j in range(0,datatrainx.shape[1]):
        if datatrainx[i][j]<=0.1:
            datatrainx[i][j] = 0
print(datatrainx)
"""
#datatrainxInt16=(datatrainx*100).astype(int16)
datatrainxInt16=np.int16(datatrainx*100)
datatrainyInt16=np.int16(datatrainy)
#datatrainy=(datatrainy*100).astype()
#save_data((np.float16(datatrainx), np.float16(datatrainy)))

# print(data2[0])
k_values = range(1, 100, 2)
#print(np.argmin(datatrainx[0]))
# vall=model_selection_knn(data1[0], data1[1], data2[0], data2[1], k_values)
aa=list(map(lambda k:erease(k),datatrainx))
ab=list(map(lambda k:erease(k),datavalx))

l = [1, 2, 4, 8.4]
#print(np.argmax(l[::1]))
#vall = model_selection_knn(np.array(ab), np.array(aa), datavaly, datatrainy, k_values)
#print(vall)

ac=list(map(lambda k:erease(k),datatrainxInt16))
acc=np.array(ac)
#save_data((acc, datatrainyInt16))
p = predict(datavalx)
#z=p.reshape(5,500)
print(p)
#print(z[1][0])
print(datavaly)


#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=sys.maxsize)
# print(data[0].shape[0])
# np.reshape(data[0][0], (36, 36))
# print((np.random.rand(300)*10).astype(int).shape[0])
# print(predict(data[0]))
#datatrainx32=np.float32(datatrainx)
#print(datatrainx32[55].reshape((36, 36)))
#pyplot.imshow(datatrainx[0].reshape((36, 36)), cmap=matplotlib.cm.Greys_r)
#pyplot.show()
#print(datatrainx[0])
#aa=(datatrainx[0]*1000).astype(int)
#print(aa)
#a=((datatrainx[0].transpose())*100).astype(int)
#print(a)
#print(countDifferences(a))
#lum_img = np.reshape(data[0][0], (36, 36))
#plt.imshow(lum_img)
'''
printt(datatrainx[2])
print(" ")
printt(datatrainx[32])
print(" ")
printt(datatrainx[43])
print(" ")
printt(datatrainx[266])
print(" ")
printt(datatrainx[87])
'''
'''
test=datatrainx[33]
testt=test.reshape(36,36)
testtt=testt.transpose()
printt(test)
print(" ")
printt(testtt)
print(" ")
printt2828(erease(datatrainx[33]))

zz=erease(datatrainx[33])

print(zz[1]==test[6*36+6])
aa=list(map(lambda k:erease(k),datatrainx))
print(aa[0].shape[0])
'''
#plt.imshow(datatrainx[0].reshape((36, 36)), cmap=cm.Greys_r)
#plt.show()