import numpy
numpy.random.seed(1337)
from keras.layers import Dense,Activation,Input,Dropout,normalization, Flatten
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D
from keras.models import Sequential,Model
from keras.utils import np_utils, generic_utils
from keras.layers.core import Dense, Activation
import scipy.io as scio
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import SGD
import h5py
from sklearn import metrics
from keras.layers.normalization import BatchNormalization
from keras import initializations
from keras.callbacks import EarlyStopping

import random
#1022
dataset=numpy.empty([11689,1,129,155])
k1=0
for i in range(1,3415):
    #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
    dataFile = '/home/serverar/data/1022pen/yuputu/' + str(i) + '.mat'
    data = h5py.File(dataFile)
    #data = scio.loadmat(dataFile)
    d=data['ZZ'][:]
    d=d.transpose()
    dataset[k1,0,:,:]=d
    k1=k1+1


label=numpy.zeros([11689])
a=numpy.zeros((804))+1
b=numpy.zeros((1746))
c=numpy.zeros((864))+2
label[0:804]=a
label[804:2550]=b
label[2550:3414]=c


#1018
for i in range(1,2349):
    #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
    dataFile = '/home/serverar/data/1018pen/yuputu/' + str(i) + '.mat'
    data = h5py.File(dataFile)
    #data = scio.loadmat(dataFile)
    d=data['ZZ'][:]
    d=d.transpose()
    dataset[k1,0,:,:]=d
    k1=k1+1



a=numpy.zeros((732))+1
b=numpy.zeros((768))
c=numpy.zeros((848))+2
label[0+3414:732+3414]=a
label[732+3414:1500+3414]=b
label[1500+3414:2348+3414]=c

#1003
for i in range(1,2545):
    #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
    dataFile = 'E:/myself download/1003PEN/yuputu/' + str(i) + '.mat'
    #data = h5py.File(dataFile)
    data = scio.loadmat(dataFile)
    d=data['ZZ'][:]
    #d=d.transpose()
    dataset[k1,0,:,:]=d
    k1=k1+1



a=numpy.zeros((960))+1
b=numpy.zeros((684))+2
c=numpy.zeros((900))
label[0+5762:960+5762]=a
label[960+5762:1644+5762]=b
label[1644+5762:2544+5762]=c

#1119
for i in range(1,3384):
    #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
    dataFile = 'E:/myself download/11yue19rizhangyuxin/yuputu/' + str(i) + '.mat'
    #data = h5py.File(dataFile)
    data = scio.loadmat(dataFile)
    d=data['ZZ'][:]
    #d=d.transpose()
    dataset[k1,0,:,:]=d
    k1=k1+1


a=numpy.zeros((935))+1
b=numpy.zeros((1020))+2
c=numpy.zeros((1428))
label[0+8306:935+8306]=a
label[935+8306:1955+8306]=b
label[1955+8306:3383+8306]=c


#np.savetxt('/Users/gryffindor/Paper/1.csv', abel, delimiter = ',')

listA = [i for i in range(11689)]
random.shuffle(listA)
listB = [i for i in range(11689)]

dataset1=numpy.zeros([11689,1,129,155])
label1=numpy.zeros([11689])
for i,j in zip(listA,listB):
    dataset1[j,:,:,:]=dataset[i,:,:,:]
    label1[j] = label[i]

label1=np_utils.to_categorical(label1,3)

#input_shape = (1, 16, 15)
nb_filters = 64
# size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (3, 3)


model = Sequential()
'''
model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(1, 129, 155),dim_ordering='th',init='glorot_uniform'))
model.add(Activation('relu'))


model.add(Convolution2D(32, kernel_size[0], kernel_size[1],border_mode='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(64, kernel_size[0], kernel_size[1],border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(64, kernel_size[0], kernel_size[1],border_mode='valid'))
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))


model.add(Dense(3))
model.add(Activation('softmax'))
'''
#input_shape = (1, 16, 15)
nb_filters = [64,96,256,256,96]
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size1 = (11, 11)
kernel_size2 = (5, 5)
kernel_size3 = (3, 3)

model.add(Convolution2D(nb_filter[0], kernel_size1[0], kernel_size1[1],
                        border_mode='valid',
                        input_shape=(1, 129, 155),dim_ordering='th',init='glorot_uniform'))
model.add(Activation('relu'))

model.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filter[1], kernel_size2[0], kernel_size2[1],border_mode='valid'))
model.add(Activation('relu'))

model.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')

model.add(MaxPooling2D(pool_size=pool_size))


model.add(Convolution2D(nb_filter[2], kernel_size3[0], kernel_size3[1],border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter[3], kernel_size3[0], kernel_size3[1],border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filter[4], kernel_size3[0], kernel_size3[1],border_mode='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))


#sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True);
model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
              optimizer='rmsprop',

#model.load_weights('/Users/gryffindor/Paper/accelerate CNN/weight3.h5' )

X_train, X_test, Y_train, Y_test = train_test_split(dataset1, label1, test_size=0.1, random_state=100)


mean=numpy.zeros([129,155])
for k in range(10520):
    mean=mean+X_train[k,0,:,:]
mean=mean/10520.0
for k in range(10520):
    X_train[k, 0, :, :]=X_train[k,0,:,:]-mean

for k in range(1169):
    X_test[k, 0, :, :] = X_test[k, 0, :, :] - mean

model.fit(X_train, Y_train, batch_size=256, nb_epoch=100,verbose=1,validation_data=(X_test, Y_test))
model.save_weights('/home/serverar/data/fourdays.h5')

score = model.evaluate(X_test, Y_test, verbose=1)

print('Test error score:', score[0])
print('Test accuracy:', score[1])


predict_label1=model.predict(X_test, batch_size=128, verbose=1)
numpy.savetxt('/home/serverar/data/predict4day1.csv', predict_label1, delimiter = ',')
predict_label2=model.predict_classes(X_test, batch_size=128, verbose=1)
numpy.savetxt('/home/serverar/data/predict4day2.csv', predict_label2, delimiter = ',')


Y1=[]
for i in Y_test:
    i = list(i)
    m = max(i)
    Y1.append(i.index(m))


print(metrics.accuracy_score(Y1, predict_label2))




