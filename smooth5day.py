
from sklearn import metrics
import numpy
my_matrix = numpy.loadtxt(open("E:\myself download\sleep experiment\predict4day_smooth2.csv","rb"),delimiter=",",skiprows=0)
i=0
while i<1140:
    a=my_matrix[i:i+60]
    n1 = numpy.count_nonzero(a == 0)
    n2 = numpy.count_nonzero(a == 1)
    n3 = numpy.count_nonzero(a == 2)
    b=[n1,n2,n3]
    j=b.index(max(b))
    my_matrix[i:i + 60]=j
    i=i+60


label_test=numpy.zeros([1200])
a=numpy.zeros((120))+1   #deep
b=numpy.zeros((120))+2   #rem
c=numpy.zeros((120))     #light
a1=numpy.zeros((60))+1   #deep
b1=numpy.zeros((60))+2   #rem
c1=numpy.zeros((60))     #light
label_test[0:120]=a
label_test[120:240]=c
label_test[240:360]=b
label_test[360:420]=a1
label_test[420:540]=c
label_test[540:660]=b
label_test[660:720]=a1
label_test[720:780]=b1
label_test[780:840]=c1
label_test[840:960]=a
label_test[960:960+120]=b
label_test[960+120:960+240]=c

print(metrics.accuracy_score(label_test, my_matrix))
print(metrics.classification_report(label_test, my_matrix))
print(metrics.confusion_matrix(label_test, my_matrix))
my_matrix2 = numpy.loadtxt(open("E:\myself download\sleep experiment\predict4day_smooth2.csv","rb"),delimiter=",",skiprows=0)

print('other')
print(metrics.accuracy_score(label_test, my_matrix2))
print(metrics.classification_report(label_test, my_matrix2))
print(metrics.confusion_matrix(label_test, my_matrix2))