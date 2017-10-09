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
'''
#10月22日
dataset=numpy.empty([3414,1,129,155])
k1=0
for i in range(1,3414):
    #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
    dataFile = '/Users/gryffindor/Paper/10yue22ri zhangyuxin/pen/yuputu/' + str(i) + '.mat'
    data = h5py.File(dataFile)
    #data = scio.loadmat(dataFile)
    d=data['ZZ'][:]
    d=d.transpose()
    dataset[k1,0,:,:]=d
    k1=k1+1


label=numpy.zeros([3414])
a=numpy.zeros((804))+1
b=numpy.zeros((1746))
c=numpy.zeros((864))+2
label[0:804]=a
label[804:2550]=b
label[2550:]=c


#np.savetxt('/Users/gryffindor/Paper/1.csv', abel, delimiter = ',')
'''
#10月18日
dataset=numpy.empty([2348,1,129,155])
k1=0
for i in range(1,2349):
    #dataFile = '/Users/gryffindor/Paper/accelerate CNN/'+str(i)+str(j)+'.mat'
    dataFile = '/Users/gryffindor/Paper/1018penclass/yuputu/' + str(i) + '.mat'
    data = h5py.File(dataFile)
    #data = scio.loadmat(dataFile)
    d=data['ZZ'][:]
    d=d.transpose()
    dataset[k1,0,:,:]=d
    k1=k1+1


label=numpy.zeros([2348])
a=numpy.zeros((732))+1
b=numpy.zeros((768))
c=numpy.zeros((848))+2
label[0:732]=a
label[732:1500]=b
label[1500:]=c


#np.savetxt('/Users/gryffindor/Paper/1.csv', abel, delimiter = ',')

listA = [i for i in range(2348)]
random.shuffle(listA)
listB = [i for i in range(2348)]

dataset1=numpy.zeros([2348,1,129,155])
label1=numpy.zeros([2348])
for i,j in zip(listA,listB):
    dataset1[j,:,:,:]=dataset[i,:,:,:]
    label1[j] = label[i]

label1=np_utils.to_categorical(label1,3)
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
        X_train[k, l, :, :]=X_train[k,l,:,:]-mean

    for k in range(912):
        X_test[k, l, :, :] = X_test[k, l, :, :] - mean
'''
#input_shape = (1, 16, 15)
nb_filters = 64
# size of pooling area for max pooling
pool_size = (3, 3)
# convolution kernel size
kernel_size = (3, 3)


##网络结构和数据平衡

model = Sequential()

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


#sgd = SGD(lr=0.1, momentum=0.8, decay=0.0, nesterov=True);
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#model.load_weights('/Users/gryffindor/Paper/accelerate CNN/weight3.h5' )

X_train, X_test, Y_train, Y_test = train_test_split(dataset1, label1, test_size=0.1, random_state=100)

#这里做归一化
mean=numpy.zeros([129,155])
for k in range(2113):
    mean=mean+X_train[k,0,:,:]
mean=mean/2113.0
for k in range(2113):
    X_train[k, 0, :, :]=X_train[k,0,:,:]-mean

for k in range(235):
    X_test[k, 0, :, :] = X_test[k, 0, :, :] - mean

model.fit(X_train, Y_train, batch_size=256, nb_epoch=3,verbose=1,validation_data=(X_test, Y_test))
#model.save_weights('/Users/gryffindor/Paper/1018penclass/yuputu/weight.h5')

score = model.evaluate(X_test, Y_test, verbose=1)

print('Test error score:', score[0])#误差率,就是在迭代中的那个loss，所有样本预测值与实际值的损失综合
print('Test accuracy:', score[1])#准确率


predict_label1=model.predict(X_test, batch_size=128, verbose=1)  #预测出类别数组，分三类的话每个预测出来的类别是一个三元数组
numpy.savetxt('/Users/gryffindor/Paper/1018penclass/CNNpredict1.csv', predict_label1, delimiter = ',')
predict_label2=model.predict_classes(X_test, batch_size=128, verbose=1)#预测出来的是最终的类别，是一个数，可以直接和最原始的类别进行比较
numpy.savetxt('/Users/gryffindor/Paper/1018penclass/CNNpredict2.csv', predict_label2, delimiter = ',')


Y1=[]
for i in Y_test:
    i = list(i)
    m = max(i)
    Y1.append(i.index(m))


print(metrics.accuracy_score(Y1, predict_label2))




