#ensemble with no weights
#import the required libraries
import pandas as pd
import numpy
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression as logr
from sklearn.ensemble import VotingClassifier as VotingClassification
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam 


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

cmResultsSVM = pd.DataFrame(columns = ['SVM1', 'SVM2', 'SVM3', 'SVM4', 'SVM5', 'AVG'])
cmResultsLR = pd.DataFrame(columns  = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5', 'AVG'])
cmResultsGNB = pd.DataFrame(columns = ['GNB1', 'GNB2', 'GNB3', 'GNB4', 'GNB5', 'AVG'])
cmResultsNN = pd.DataFrame(columns  = ['NN1',  'NN2',  'NN3',  'NN4',  'NN5', 'AVG'])
cmResultsENS = pd.DataFrame(columns = ['ENS1', 'ENS2', 'ENS3', 'ENS4', 'ENS5', 'AVG'])

clResultsSVM = pd.DataFrame(columns = ['SVM1', 'SVM2', 'SVM3', 'SVM4', 'SVM5', 'AVG'])
clResultsLR = pd.DataFrame(columns  = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5', 'AVG'])
clResultsGNB = pd.DataFrame(columns = ['GNB1', 'GNB2', 'GNB3', 'GNB4', 'GNB5', 'AVG'])
clResultsNN = pd.DataFrame(columns  = ['NN1',  'NN2',  'NN3',  'NN4',  'NN5', 'AVG'])
clResultsENS = pd.DataFrame(columns = ['ENS1', 'ENS2', 'ENS3', 'ENS4', 'ENS5', 'AVG'])


ss = ['SVM1', 'SVM2', 'SVM3', 'SVM4', 'SVM5']
ls = ['LR1',  'LR2',  'LR3',  'LR4',  'LR5']
gs = ['GNB1', 'GNB2', 'GNB3', 'GNB4', 'GNB5']
ns = ['NN1',  'NN2',  'NN3',  'NN4',  'NN5']
es = ['ENS1', 'ENS2', 'ENS3', 'ENS4', 'ENS5']

a=0
#k folds split
kf = KFold(5, True)
kf.get_n_splits(x) 
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
    
    s = ss[a]
    l = ls[a]
    g = gs[a]
    n = ns[a]
    e = es[a]
    
    #build models
    #SVM model
    modelSVM = SVC(kernel='linear', gamma = 'scale') #kernal: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    modelSVM.fit(X_train, Y_train.ravel())
    Y_pred_SVM = modelSVM.predict(X_test)
    
    #LR model
    modelLR = logr(solver = 'newton-cg') #solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    modelLR.fit(X_train, Y_train.ravel())
    Y_pred_LR = modelLR.predict(X_test)
    
    #GNB Model
    modelGNB = GaussianNB()
    modelGNB.fit(X_train, Y_train.ravel())
    Y_pred_GNB = modelGNB.predict(X_test)
    
    #NN Model
    model = Sequential()
    model.add(Dense(16, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=300, batch_size=10, verbose = 10)
    Y_pred_NN = np.round(model.predict(X_test)).astype(int)
    
    #Combine in ensemble 
    Y_pred = VotingClassification(estimators=[('svm', modelSVM), ('lr',modelLR), ('GNB', modelGNB), ('NN', model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy']))], voting='hard', weights=None, n_jobs=None, flatten_transform=True)
    Y_pred = Y_pred.fit(X_train, Y_train)
    pred = Y_pred.predict(X_test)
    
    #get results
    cm_SVM= confusion_matrix(Y_test,Y_pred_SVM)
    cm_LR= confusion_matrix(Y_test,Y_pred_LR)
    cm_GNB = confusion_matrix(Y_test,Y_pred_GNB)
    cm_NN = confusion_matrix(Y_test,Y_pred_NN)
    cm_en = confusion_matrix(Y_test, pred)
    
    #false negs, true neg, false pos, true pos
    SVM_cm = ((cm_SVM[1][0]),(cm_SVM[0][0]),(cm_SVM[0][1]),(cm_SVM[1][1]))
    LR_cm = (cm_LR[1][0], cm_LR[0][0], cm_LR[0][1], cm_LR[1][1])
    GNB_cm = (cm_GNB[1][0], cm_GNB[0][0], cm_GNB[0][1], cm_GNB[1][1])
    NN_cm = (cm_NN[1][0], cm_NN[0][0], cm_NN[0][1], cm_NN[1][1])
    en_cm = (cm_en[1][0], cm_en[0][0], cm_en[0][1], cm_en[1][1])
 
    #accuracy, precision, recall, f1
    result_SVM = (accuracy_score(Y_test,Y_pred_SVM), precision_score(Y_test,Y_pred_SVM), recall_score(Y_test,Y_pred_SVM), f1_score(Y_test,Y_pred_SVM))
    result_LR = (accuracy_score(Y_test,Y_pred_LR), precision_score(Y_test,Y_pred_LR), recall_score(Y_test,Y_pred_LR), f1_score(Y_test,Y_pred_LR))
    result_GNB = (accuracy_score(Y_test,Y_pred_GNB), precision_score(Y_test,Y_pred_GNB), recall_score(Y_test,Y_pred_GNB), f1_score(Y_test,Y_pred_GNB))
    result_NN = (accuracy_score(Y_test,Y_pred_NN), precision_score(Y_test,Y_pred_NN), recall_score(Y_test,Y_pred_NN), f1_score(Y_test,Y_pred_NN))
    result_en = (accuracy_score(Y_test,pred), precision_score(Y_test,pred), recall_score(Y_test,pred), f1_score(Y_test,pred))
    
    #save results
    cmResultsSVM[s] = SVM_cm
    cmResultsLR[l] = LR_cm
    cmResultsGNB[g] =GNB_cm
    cmResultsNN[n] = NN_cm
    cmResultsENS[e] = en_cm
    
    clResultsSVM[s] = result_SVM
    clResultsLR[l] = result_LR
    clResultsGNB[g] =result_GNB
    clResultsNN[n] = result_NN
    clResultsENS[e] = result_en
    a=a+1

#get average results
cmResultsSVM['AVG'] = cmResultsSVM.mean(axis=1)
cmResultsLR['AVG'] = cmResultsLR.mean(axis=1)
cmResultsGNB['AVG'] = cmResultsGNB.mean(axis=1) 
cmResultsNN['AVG'] = cmResultsNN.mean(axis=1)
cmResultsENS['AVG'] = cmResultsENS.mean(axis=1)

clResultsSVM['AVG'] = clResultsSVM.mean(axis=1)
clResultsLR['AVG'] = clResultsLR.mean(axis=1)
clResultsGNB['AVG'] = clResultsGNB.mean(axis=1) 
clResultsNN['AVG'] = clResultsNN.mean(axis=1)
clResultsENS['AVG'] = clResultsENS.mean(axis=1)

print ("average SVM:")
print(cmResultsSVM['AVG'])
print(clResultsSVM['AVG'])
print ("average LR:")
print(cmResultsLR['AVG'])
print(clResultsLR['AVG'])
print ("average GNB")
print(cmResultsGNB['AVG'])
print(clResultsGNB['AVG'])
print ("average NN")
print(cmResultsNN['AVG'])
print(clResultsNN['AVG'])
print ("average ENS:")
print(cmResultsENS['AVG'])
print(clResultsENS['AVG'])
