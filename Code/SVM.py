#import the required libraries
import pandas as pd
import numpy
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.svm import SVC #support vector classifier class
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler


#reading in data from excel
data = pd.read_excel (r'C:\Users\lauren\Documents\1st year Grad\ECE 9603\project\heart_edited.xlsx')
data = data.drop(['Age'], axis = 1)
data = data.drop(['fbs'], axis = 1)
data = data.drop(['restecg'], axis = 1)
data = data.drop(['chol'], axis = 1)
data = data.drop(['trestbps'], axis = 1)
#assign x and y
y = data.target.values.ravel()
x = data.drop(['target'], axis = 1)
#make array from df
x=np.array(x)
y=np.array(y)

SVM = ['SVM1',  'SVM2',  'SVM3',  'SVM4',  'SVM5']
cmResults = pd.DataFrame(columns  = ['SVM1',  'SVM2',  'SVM3',  'SVM4',  'SVM5'])
clResults = pd.DataFrame(columns  = ['SVM1',  'SVM2',  'SVM3',  'SVM4',  'SVM5'])

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

#k folds split
kf = KFold(5, True)
kf.get_n_splits(x)
for a in range (0,4):
    ker = kernels[a]#set solver type
    print("kernel type is: ", ker)
    b=0
    # Enumerate splits
    for train_index, test_index in kf.split(x):
        s=SVM[b]
        
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]      
        #normailize train data 
        norm = MinMaxScaler(feature_range=(0, 1))
        X_train = norm.fit_transform(X_train)
        Y_train = norm.fit_transform(Y_train.reshape(-1, 1))
        X_test = norm.fit_transform(X_test)
        Y_test = norm.fit_transform(Y_test.reshape(-1, 1))
              
        model = SVC(kernel= ker,  gamma='scale') #kernal: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        model.fit(X_train, Y_train.ravel())
        Y_pred = model.predict(X_test)
        
        cm_SVM= confusion_matrix(Y_test,Y_pred)
        #seperate to be stored in individual rows
        SVM_cm = (cm_SVM[1][0], cm_SVM[0][0], cm_SVM[0][1], cm_SVM[1][1]) #false negs, true neg, false pos, true pos
        cmResults[s] = SVM_cm #store CM results 
        
        #record the average classification report info for each trial
        result_SVM = (accuracy_score(Y_test,Y_pred), precision_score(Y_test,Y_pred), recall_score(Y_test,Y_pred), f1_score(Y_test,Y_pred))
        clResults[s] = result_SVM
        b=b+1

    
    #calculate average over 5 trials
    cmResults['AVG'] = cmResults.mean(axis=1)
    clResults['AVG'] = clResults.mean(axis=1)
    
    #print average results of Confusion matrix and Classification report
    print ("Average Confustion Matrix:")
    print(cmResults['AVG'])
    print ("average Classification results:")
    print(clResults['AVG'])
        
    #plot average CM    
    ax= plt.subplot()
    cm_average = [[round(cmResults['AVG'][1]), round(cmResults['AVG'][2])],[round(cmResults['AVG'][0]),round(cmResults['AVG'][3])]]
    sn.heatmap(cm_average, annot=True, cmap="Blues")
    
    # labels, title and ticks
    ax.set_xlabel('Classifier Prediction')
    ax.set_ylabel('True Value')
    ax.set_title('Average Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1']) 
    ax.yaxis.set_ticklabels(['0', '1'])
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() 



