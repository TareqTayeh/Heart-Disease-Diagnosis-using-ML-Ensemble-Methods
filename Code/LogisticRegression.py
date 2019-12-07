#import the required libraries
import pandas as pd
import numpy
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as logr
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#***** assign values after train test split from average of coresponding heart disease true or false ********8

#reading in data from excel
data = pd.read_excel (r'C:\Users\lauren\Documents\1st year Grad\ECE 9603\project\heart_edited.xlsx')
data = data.drop(['Age'], axis = 1)
data = data.drop(['fbs'], axis = 1)
data = data.drop(['restecg'], axis = 1)
data = data.drop(['chol'], axis = 1)
data = data.drop(['trestbps'], axis = 1)

#assign x and y
y = data.target.values
x = data.drop(['target'], axis = 1)

ls = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5', 'LR6',  'LR7',  'LR8',  'LR9',  'LR10', 'LR11', 'LR12', 'LR13', 'LR14', 'LR15', 'LR16', 'LR17',  'LR18',  'LR19', 'LR20']
cmResults = pd.DataFrame(columns  = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5', 'LR6',  'LR7',  'LR8',  'LR9',  'LR10', 'LR11', 'LR12', 'LR13', 'LR14', 'LR15', 'LR16', 'LR17',  'LR18',  'LR19', 'LR20', 'AVG'])
clResults = pd.DataFrame(columns  = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5', 'LR6',  'LR7',  'LR8',  'LR9',  'LR10', 'LR11', 'LR12', 'LR13', 'LR14', 'LR15', 'LR16', 'LR17',  'LR18',  'LR19', 'LR20', 'AVG'])

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

for a in range (0,5):
    sol = solvers[a]#set solver type
    print("solver type is: ", sol)
    
    for b in range(0,20): #run 20 times for each solver
        l=ls[b]
           
        #train test split, random 20% test and 80% train becasue its not a time series
        X_train, _ , Y_train, _ = train_test_split(x,y, test_size = 0.2)
        Y_train = Y_train.reshape(-1, 1)
        
        #normailize train data 
        norm = MinMaxScaler(feature_range=(0, 1))
        X_train = norm.fit_transform(X_train)
        Y_train = norm.fit_transform(Y_train)
        
        #normalize data for test split 
        test = norm.fit_transform(data)
        Y_test = np.delete(test,[0,1,2,3,4,5,6,7], 1)
        X_test = np.delete(test, 8, 1)
        
        #get test split
        _, X_test, _, Y_test = train_test_split(X_test, Y_test, test_size = 0.2)
        
        #build model
        model = logr(solver = sol) #solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
        model.fit(X_train, Y_train.ravel())
        Y_pred = model.predict(X_test)
        
        cm_LR= confusion_matrix(Y_test,Y_pred)
        #seperate to be stored in individual rows
        LR_cm = (cm_LR[1][0], cm_LR[0][0], cm_LR[0][1], cm_LR[1][1]) #false negs, true neg, false pos, true pos
        cmResults[l] = LR_cm #store CM results 
        
        #record the average classification report info for each trial
        result_LR = (accuracy_score(Y_test,Y_pred), precision_score(Y_test,Y_pred), recall_score(Y_test,Y_pred), f1_score(Y_test,Y_pred))
        clResults[l] = result_LR

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