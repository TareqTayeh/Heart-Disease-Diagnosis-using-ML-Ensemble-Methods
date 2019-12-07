#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from random import seed, randrange
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt, pi, exp
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading csv values
df = pd.read_excel(r'../Dataset/heart_dataset_complete.xlsx')
#df.head()
#print(df.to_string())


# In[3]:


#Convert question marks to mean of the attribute
# df['ca'] = df['ca'].replace('?', 1) #Calculated Mean
# df['thal'] = df['thal'].replace('?', 2) #Calculated Mean


# In[4]:


#Converting all values from strings to floats
df['age'] = df['age'].astype(float)
df['sex'] = df['sex'].astype(float)
df['cp'] = df['cp'].astype(float)
df['trestbps'] = df['trestbps'].astype(float)
df['chol'] = df['chol'].astype(float)
df['fbs'] = df['fbs'].astype(float)
df['restecg'] = df['restecg'].astype(float)
df['thalach'] = df['thalach'].astype(float)
df['exang'] = df['exang'].astype(float)
df['oldpeak'] = df['oldpeak'].astype(float)
df['slope'] = df['slope'].astype(float)
df['ca'] = df['ca'].astype(float)
df['thal'] = df['thal'].astype(float)
df['target'] = df['target'].astype(float)


# In[5]:


df


# In[6]:


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated
 
# Test separating data by class
dataset = [[3.393533211,2.331273381,0],
    [3.110073483,1.781539638,0],
    [1.343808831,3.368360954,0],
    [3.582294042,4.67917911,0],
    [2.280362439,2.866990263,0],
    [7.423436942,4.696522875,1],
    [5.745051997,3.533989803,1],
    [9.172168622,2.511101045,1],
    [7.792783481,3.424088941,1],
    [7.939820817,0.791637231,1]]
separated = separate_by_class(dataset)
for label in separated:
    print(label)
    for row in separated[label]:
        print(row)


# In[7]:


heart_disease_dataset = np.array(df)


# In[8]:


separated = separate_by_class(heart_disease_dataset)
for label in separated:
    print(label)
    for row in separated[label]:
        print(row)


# In[9]:


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))


# In[10]:


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)


# In[11]:


#The first trick is the use of the zip() function that will aggregate elements from each provided argument. 
#We pass in the dataset to the zip() function with the * operator that separates the dataset (that is a list of lists) 
#into separate lists for each row. The zip() function then iterates over each element of each row and returns a column 
#from the dataset as a list of numbers. A clever little trick.
#--------
#We then calculate the mean, standard deviation and count of rows in each column. A tuple is created from these 3 numbers 
#and a list of these tuples is stored. We then remove the statistics for the class variable as we will not need these 
#statistics.
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries


# In[12]:


summary = summarize_dataset(dataset)
print(summary)


# In[13]:


heart_summary = summarize_dataset(heart_disease_dataset)
print(heart_summary)
#age -- sex -- cp -- trestbps -- chol -- fbs -- restecg -- thalach -- exang -- oldpeak -- slope -- ca -- thal -- target


# In[14]:


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# In[15]:


summary = summarize_by_class(dataset)
for label in summary:
    print(label)
    for row in summary[label]:
        print(row)


# In[16]:


heart_summary = summarize_by_class(heart_disease_dataset)
for label in heart_summary:
    print(label)
    for row in heart_summary[label]:
        print(row)
#age -- sex -- cp -- trestbps -- chol -- fbs -- restecg -- thalach -- exang -- oldpeak -- slope -- ca -- thal -- target


# In[17]:


#Calculating the probability or likelihood of observing a given real-value like X1 is difficult.
#One way we can do this is to assume that X1 values are drawn from a distribution, such as a Gaussian distribution.

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# In[18]:


# Test Gaussian PDF
print(calculate_probability(1.0, 1.0, 1.0))
print(calculate_probability(2.0, 1.0, 1.0))
print(calculate_probability(0.0, 1.0, 1.0))


# In[19]:


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


# In[20]:


# Test calculating class probabilities
dataset = [[3.393533211,2.331273381,0],
    [3.110073483,1.781539638,0],
    [1.343808831,3.368360954,0],
    [3.582294042,4.67917911,0],
    [2.280362439,2.866990263,0],
    [7.423436942,4.696522875,1],
    [5.745051997,3.533989803,1],
    [9.172168622,2.511101045,1],
    [7.792783481,3.424088941,1],
    [7.939820817,0.791637231,1]]
summaries = summarize_by_class(dataset)
probabilities = calculate_class_probabilities(summaries, dataset[6])
print(probabilities)


# In[21]:


probabilities = calculate_class_probabilities(heart_summary, heart_disease_dataset[1])
print(probabilities)


# In[22]:


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label
 
# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


# In[ ]:


#Evaluate Algorithm
seed(1)
n_folds = 5
scores = evaluate_algorithm(heart_disease_dataset, naive_bayes, n_folds)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

