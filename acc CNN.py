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

dataset=numpy.empty([9120,3,16,15])
k1=0
for i in range(1,153):
    for j in range(1,61):
        #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
        dataFile = '/Users/gryffindor/Paper/accelerate CNN/channel1/' + str(i) + str(j) + '.mat'
        data = h5py.File(dataFile)
        #data = scio.loadmat(dataFile)
        d=data['Z'][:]   #(15,16)的数组，应该转置，h5py不知道怎么转置,应该是（16，15）
        d=d.transpose()
        dataset[k1,0,:,:]=d
        k1=k1+1



k2=0
for i in range(1, 153):
    for j in range(1, 61):
        # dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
        dataFile = '/Users/gryffindor/Paper/accelerate CNN/channel2/' + str(i) + str(j) + '.mat'
        data = h5py.File(dataFile)
        # data = scio.loadmat(dataFile)
        d = data['Z'][:]  # (15,16)的数组，应该转置，h5py不知道怎么转置,应该是（16，15）
        d = d.transpose()
        dataset[k2, 1, :, :] = d
        k2 = k2 + 1
k3 = 0
for i in range(1, 153):
    for j in range(1, 61):
        # dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
        dataFile = '/Users/gryffindor/Paper/accelerate CNN/channel3/' + str(i) + str(j) + '.mat'
        data = h5py.File(dataFile)
        # data = scio.loadmat(dataFile)
        d = data['Z'][:]  # (15,16)的数组，应该转置，h5py不知道怎么转置,应该是（16，15）
        d = d.transpose()
        dataset[k3, 2, :, :] = d
        k3 = k3 + 1


#label 60个1，60个2，...，60个19，60个1，...，60个19，（八次，代表八个人）
label=numpy.zeros([9120])
a=numpy.zeros((60))
for i in range(8):
    for j in range(19):
        label[0+60*j+1140*i:60+60*j+1140*i]=a+j
#np.savetxt('/Users/gryffindor/Paper/1.csv', abel, delimiter = ',')



listA = [i for i in range(9120)]
random.shuffle(listA)
listB = [i for i in range(9120)]

dataset1=numpy.zeros([9120,3,16,15])
label1=numpy.zeros([9120])
for i,j in zip(listA,listB):
    dataset1[j,:,:,:]=dataset[i,:,:,:]
    label1[j] = label[i]

label1=np_utils.to_categorical(label1,19)
'''
X_train=dataset[912:,:,:,:]
Y_train=label[912:]
X_test=dataset[:912,:,:,:]
Y_test=label[:912]

X_train=dataset[912:,:,:,:]
Y_train=label[912:]
X_test=dataset[912:1824,:,:,:]
Y_test=label[912:1824]
'''
'''
X_train, X_test, Y_train, Y_test = train_test_split(dataset1, label1, test_size=0.1, random_state=15)

for l in range(3):
    #这里做归一化
    mean=numpy.zeros([16,15])
    for k in range(8208):
        mean=mean+X_train[k,l,:,:]
    mean=mean/8208.0
    for k in range(8208):
        X_train[k, 0, :, :]=X_train[k,l,:,:]-mean

    for k in range(912):
        X_test[k, 0, :, :] = X_test[k, l, :, :] - mean
'''
#input_shape = (1, 16, 15)
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)




model = Sequential()

model.add(Convolution2D(32, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(3, 16, 15),dim_ordering='th',init='glorot_uniform'))
model.add(Activation('relu'))



model.add(Convolution2D(32, kernel_size[0], kernel_size[1],border_mode='valid'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))
#model.add(AveragePooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))


model.add(Dense(19))
model.add(Activation('softmax'))


#sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True);
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.load_weights('/Users/gryffindor/Paper/accelerate CNN/weight3.h5' )

sum1=0
for i in [3,30,56,77,24,120,349,70,96,500]:
    X_train, X_test, Y_train, Y_test = train_test_split(dataset1, label1, test_size=0.1, random_state=i)

    for l in range(3):
        #这里做归一化
        mean=numpy.zeros([16,15])
        for k in range(8208):
            mean=mean+X_train[k,l,:,:]
        mean=mean/8208.0
        for k in range(8208):
            X_train[k, l, :, :]=X_train[k,l,:,:]-mean

        for k in range(912):
            X_test[k, l, :, :] = X_test[k, l, :, :] - mean

    model.fit(X_train, Y_train, batch_size=128, nb_epoch=1,verbose=1,validation_data=(X_test, Y_test))
#model.save_weights('/Users/gryffindor/Paper/accelerate CNN/weight.h5')  #存三通道
#model.save_weights('/Users/gryffindor/Paper/accelerate CNN/weight1.h5')  #存一通道，归一化后
#model.save_weights('/Users/gryffindor/Paper/accelerate CNN/weight2.h5')  #存三通道，归一化后
#model.save_weights('/Users/gryffindor/Paper/accelerate CNN/weight3.h5')  #存三通道，归一化后,再把它打乱计算的权重
    score = model.evaluate(X_test, Y_test, verbose=1)

    print('Test error score:', score[0])#误差率,就是在迭代中的那个loss，所有样本预测值与实际值的损失综合
    print('Test accuracy:', score[1])#准确率
    sum1=sum1+score[1]


print('10-fold:',sum1/10.0)

#0.919956140351+0.915570175439+0.912280701754+0.910087719298+0.910087719298+0.911184210526+ 0.906798245614+0.901315789474+0.904605263158+0.883771929825
