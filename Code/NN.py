#python 3.7
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import max_norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ########################################################################### #
#######                        Data Exploration                         #######
# ########################################################################### #

# Load the data
dataframe = pd.read_csv("heart_edited.csv")
print (dataframe.head()) # Observe the first 5 entries to the data

# Look at distribution of values and general info of the data set
countNoDisease = len(dataframe[dataframe.target == 0])
countHaveDisease = len(dataframe[dataframe.target == 1])
print("Percentage of Patients Without Heart Disease: {:.2f}%".format((countNoDisease / (len(dataframe.target))*100)))
print("Percentage of Patients With Heart Disease: {:.2f}%".format((countHaveDisease / (len(dataframe.target))*100)))

dataframe.info()

# look for null values
print (dataframe.isnull().any().describe())
# fill in null values
null_counts = dataframe.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)

#Look at correlation between the values of the data set
plt.figure(figsize=(14,10))
sns.heatmap(dataframe.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()

# Split the target from the rest of the data set
predictors = dataframe.drop("target",axis=1)
target = dataframe["target"]
# Split into training and test set 70:30
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.30, random_state=1)
# Split into test and validation set 67:33 so total ratio is 70:20:10
X_test, X_val, Y_test,Y_val = train_test_split(X_test, Y_test, test_size=0.33, random_state=1)
    
# replace nan values from the test and training sets
X_train = X_train.fillna(X_train.mode().iloc[0])
X_test = X_test.fillna(X_test.mode().iloc[0])
X_val = X_val.fillna(X_val.mode().iloc[0])

# #############################################################################
#######                          Model Tuning                           #######
# #############################################################################
### do not recomment ever uncommenting this, you do not want this to run, just here for evidence of tuning ##

#def create_model(optimizer='adam', learn_rate=0.01, momentum=0, init_mode='uniform', activation='relu', 
#                    dropout_rate=0.0, weight_constraint=0, neurons1=1, neurons2=1):
## create model
#    model = Sequential()
#    model.add(Dense(neurons1, input_dim=13, kernel_initializer=init_mode, activation=activation, kernel_constraint=max_norm(weight_constraint)))
#    model.add(Dense(neurons2, kernel_initializer=init_mode, activation=activation, kernel_constraint=max_norm(weight_constraint)))
#    model.add(Dropout(dropout_rate))
#    model.add(Dense(1, activation='sigmoid'))
#    
#    # Compile model
#    #adam = Adam(lr=learn_rate, momentum=momentum)
#    #optimizer = SGD(lr=learn_rate, momentum=momentum)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#    return model
#
#model = KerasClassifier(build_fn=create_model, verbose=0)
#
## define the grid search parameters
#batch_size = [10, 20, 40, 60, 80, 100]
#epochs = [10, 50, 100, 300]
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#weight_constraint = [1, 2, 3, 4, 5]
#dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#neurons1 = [1, 4, 8, 12, 16, 20, 24, 28]
#neurons2 = [1, 2, 4, 6, 8]
#param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, momentum=momentum, 
#                    init_mode=init_mode, activation=activation, dropout_rate=dropout_rate, weight_constraint=weight_constraint, 
#                    neurons1=neurons1, neurons2=neurons2)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#grid_result = grid.fit(X_train, Y_train)
#
## summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#means = grid_result.cv_results_['mean_test_score']
#stds = grid_result.cv_results_['std_test_score']
#params = grid_result.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))


# #############################################################################
#######                      Build Final Model                          #######
# #############################################################################

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())

history=model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=10, verbose = 10)


# #############################################################################
#######                   Evaluate Final Model                          #######
# #############################################################################
#print(history.history.keys())

# Model accuracy over time
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

# Model Loss over time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

Y_pred = np.round(model.predict(X_test)).astype(int)

print('Results for Model')
print(accuracy_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Not Sick', 'Sick']); ax.yaxis.set_ticklabels(['Not Sick', 'Sick']);

b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!
