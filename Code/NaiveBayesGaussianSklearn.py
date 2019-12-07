#!/usr/bin/env python
# coding: utf-8

# In[43]:


from __future__ import division
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp


# In[44]:


#Reading csv values
df = pd.read_excel(r'../Dataset/heart_dataset_complete.xlsx')
df.head()


# In[45]:


#Convert question marks to mean values of the corresponding attribute
# df['ca'] = df['ca'].replace('?', 1)
# df['thal'] = df['thal'].replace('?', 2)


# In[46]:


df


# In[47]:


#Split dataframe into x (independant variables) and y (dependant variable)
x_df=df.iloc[1:df.shape[0],0:13]
print(x_df)

y_df=df.iloc[1:df.shape[0],13:14]
print(y_df)


# In[48]:


#Function for entire calculations and model fitting
def model_calculations(x_df, y_df):
    # Converting dataframe into arrays
    x=np.array(x_df)
    y=np.array(y_df)
    # Prepare cross validation
    accuracy_scores = []
    kf = KFold(5, True)
    kf.get_n_splits(x)
    print(kf)
    # Enumerate splits
    for train_index, test_index in kf.split(x):
    #   print("TRAIN:", train_index, "TEST:", test_index)
        print("\nTEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Gaussian function of sklearn
        gnb = GaussianNB()
        gnb.fit(x_train, y_train.ravel())
        y_pred = gnb.predict(x_test)
        accuracy_scores.append(metrics.accuracy_score(y_test, y_pred)*100)
        print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
        # convert 2D array to 1D array
        y1=y_test.ravel()
        y_pred1=y_pred.ravel()
        #A confusion matrix is a summary of prediction results on a classification problem.
        #The number of correct and incorrect predictions are summarized with count values and broken down by each class.
        #TP - FN
        #FP - TN
        print("Confusion Matrix")
        cf_matrix=confusion_matrix(y1,y_pred1)
        print(cf_matrix)
        #F1 score = 2*((precision*recall)/(precision+recall))
        #Precision = Number of True Positives divided by the number of True Positives and False Positives.
        #Recall = Number of True Positives divided by the number of True Positives and the number of False Negatives
        print("Precision")
        precision=precision_score(y1,y_pred1,average='weighted')
        print(precision)
        print("Recall")
        recall=recall_score(y1,y_pred1,average='weighted')
        print(recall)
        print("F1 Score")
        f_score=f1_score(y1,y_pred1,average='weighted')
        print(f_score)
        # Matrix from 1D array
        y2=np.zeros(shape=(len(y1),5))
        y3=np.zeros(shape=(len(y_pred1),5))
        for i in range(len(y1)):
            y2[i][int(y1[i])]=1
        for i in range(len(y_pred1)):
            y3[i][int(y_pred1[i])]=1
        # ROC Curve generation
        n_classes = 2

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y2[:, i], y3[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y2.ravel(), y3.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        print("ROC Curve")
        # First aggregate all false positive rates
        lw=2
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','black'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for Multi-Class')
        plt.legend(loc="lower right")
        plt.show()

    print(accuracy_scores)
    max_value = max(accuracy_scores)
    min_value = min(accuracy_scores)
    avg_value = sum(accuracy_scores)/len(accuracy_scores)
    print('Max:', max_value)
    print('Min:',min_value)
    print('Avg:',avg_value)


# In[49]:


model_calculations(x_df, y_df)


# In[50]:


# Converting dataframe into arrays
x=np.array(x_df)
y=np.array(y_df)
# Prepare cross validation
accuracy_scores = []
kf = KFold(5, True)
kf.get_n_splits(x)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Gaussian function of sklearn
gnb = GaussianNB()
gnb.fit(x_train, y_train.ravel())


# In[51]:


#eli5 provides a way to compute feature importances for any black-box estimator by measuring how score decreases 
#when a feature is not available;

#Permutation importance is calculated after a model has been fitted. So we won't change the model or change what predictions 
#we'd get for a given value. It randomly re-orders a single column of the validation data, leaving the target and all 
#other columns in place, and calculates the prediction accuracy of the now-shuffle data. The process is repeated with 
#multiple shuffles to measure the amount of randomness in the calculation

import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(gnb, random_state = 1).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_df.columns.tolist())


# In[52]:


#On average, it looks like the most important factors in terms of permutation are:
# 1) Number of major vessels coloured by fluoroscopy. 
# 2) Slope of peak exercise ST segment
# 3) Exercise induced angina
# 4) Sex
# 5) Chest Pain Type


# In[53]:


#Drop age, serum cholesterol, fasting blood sugar, resting blood pressure and resting electrocardiographic results
df = df.drop(columns=['age', 'trestbps', 'chol', 'fbs', 'restecg'])


# In[54]:


#Split dataframe into x (independant variables) and y (dependant variable)
x_df=df.iloc[1:df.shape[0],0:8]
print(x_df)

y_df=df.iloc[1:df.shape[0],8:9]
print(y_df)


# In[55]:


model_calculations(x_df, y_df)


# In[56]:


# converting dataframe into arrays
x=np.array(x_df)
y=np.array(y_df)
# prepare cross validation
accuracy_scores = []
kf = KFold(5, True)
kf.get_n_splits(x)
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Gaussian function of sklearn
gnb = GaussianNB()
gnb.fit(x_train, y_train.ravel())
#Permutation Importance
perm = PermutationImportance(gnb, random_state = 1).fit(x_test, y_test)
eli5.show_weights(perm, feature_names = x_df.columns.tolist())

