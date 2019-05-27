import pickle
import sys

import numpy as np
import predict
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from predict import (hamming_distance, sort_train_labels_knn, model_selection_knn,classification_error)

def load_data():
    PICKLE_FILE_PATH = 'train.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pickle.load(f)



#validacyjne dostajesz na serwerze

print("hello")


data = load_data()
print(data)
(x,y)=data
#print(x)
#print(y)
rep=(x,y)
#print(rep)

#liczba testów 60000

datatrainx=data[0][:800]
datavalx=data[0][1000:2000]
datatrainy=data[1][:800]
datavaly=data[1][1000:2000]
data1 = np.vsplit(data[0],30)
data2 = np.split(data[1],30)
#print(data2[0])
k_values = range(1, 201, 2)
#vall=model_selection_knn(data1[0], data1[1], data2[0], data2[1], k_values)

#vall=model_selection_knn(datavalx, datatrainx, datavaly, datatrainy, k_values)
#print(vall)

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=sys.maxsize)
#print(data[0].shape[0])
#np.reshape(data[0][0], (36, 36))
#print((np.random.rand(300)*10).astype(int).shape[0])
#print(predict(data[0]))
#plt.imshow(data[0][55].reshape((36, 36)), cmap=cm.Greys_r)
#plt.show()
#lum_img = np.reshape(data[0][0], (36, 36))
#plt.imshow(lum_img)