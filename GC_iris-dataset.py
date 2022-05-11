import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from mlxtend.plotting import plot_decision_regions

iris=load_iris()

def Train_test_split(data,labels,test_ratio=0.3):
    data=np.array(data)
    labels=np.array(labels)
    data=np.hstack((data, np.atleast_2d(labels).T))
    np.random.shuffle(data)
    test_data = []
    test_size=int(test_ratio*len(data))
    train_data=data
    while len(test_data)<test_size:
        index=0 
        test_data.append(train_data[index])
        train_data=np.delete(train_data,index,axis=0)
    test_data=np.array(test_data)
    train_labels=train_data[:,-1]
    test_labels=test_data[:,-1]
    train_data=np.delete(train_data,4,axis=1)
    test_data=np.delete(test_data,4,axis=1)
    return train_data,test_data,train_labels,test_labels

def calculate_accuracy(predicted_y,test_y):
    i=0
    count=0
    while i<len(test_y):
        if predicted_y[i]==test_y[i]:
            count=count+1
        i=i+1
    return (count/len(test_y))*100

#spliting data

x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=Train_test_split(x, y)

#model training on 4 features

model=GaussianNB()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

#model training on 2 features to draw decision boundaries

m=GaussianNB()
m=SVC(C=0.5, kernel='linear')
m.fit(x_train[:,[0,2]],y_train)
y_predicted=m.predict(x_test[:,[0,2]])
plot_decision_regions(x_train[:,[0,2]],np.array(list(map(np.int64, y_train))), clf=m, legend=2)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('SVM on Iris')
plt.show()
print(calculate_accuracy(y_predicted, y_test))
print(calculate_accuracy(y_predict, y_test))