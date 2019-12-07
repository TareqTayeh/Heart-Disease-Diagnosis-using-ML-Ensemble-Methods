#import the required libraries
import pandas as pd
import numpy
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import eli5

from sklearn.linear_model import LogisticRegression as logr
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from eli5.sklearn import PermutationImportance

#***** assign values after train test split from average of coresponding heart disease true or false ********8

#reading in data from excel
data = pd.read_excel(r'../Dataset/heart_edited.xlsx')
#Drop specific columns
data = data.drop(["age","fbs","trestbps","chol","restecg"],axis=1)

# Split the target from the rest of the data set, assign x and y
y_df = data.target.values.ravel()
x_df = data.drop(['target'], axis = 1)
# Make array from df
x = np.array(x_df)
y = np.array(y_df)

ls = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5']
cmResults = pd.DataFrame(columns  = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5'])
clResults = pd.DataFrame(columns  = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5'])

solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

#k folds split
kf = KFold(5, True)
kf.get_n_splits(x)
for a in range (0,5):
    sol = solvers[a]#set solver type
    print("solver type is: ", sol)
    b=0
    l=ls[b]
    # Enumerate splits
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        
        #normailize train data 
        norm = MinMaxScaler(feature_range=(0, 1))
        X_train = norm.fit_transform(X_train)
        Y_train = norm.fit_transform(Y_train.reshape(-1, 1))
        
        X_test = norm.fit_transform(X_test)
        Y_test = norm.fit_transform(Y_test.reshape(-1, 1))
        
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
        b=b+1
        
        #Permutation Importance
        perm = PermutationImportance(model, random_state = 1).fit(X_test, Y_test)
        eli5.show_weights(perm, feature_names = x_df.columns.tolist())

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