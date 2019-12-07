#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import seaborn as sns
from math import sqrt, pi, exp
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading csv values
df = pd.read_excel(r'../Dataset/heart_edited.xlsx')


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


#Displaying statistics of the dataset
df.describe()


# In[7]:


#Calculating the mean of every group
df.groupby('target').mean()


# In[8]:


# Graphs to check and analyze the data
fig, axes = plt.subplots( nrows=10, ncols=3, figsize=(15,40) )
plt.subplots_adjust( wspace=0.20, hspace=0.20, top=0.97 )
plt.suptitle("Heart Disease Data", fontsize=20)
axes[0,0].hist(df.age)
axes[0,0].set_xlabel("Age (years)")
axes[0,0].set_ylabel("Number of Patients")
axes[0,1].hist(df.sex)
axes[0,1].set_xlabel("Sex (0=female,1=male)")
axes[0,1].set_ylabel("Number of Patients")
axes[0,2].hist(df.cp,bins=4,range=(0.5,4.5),rwidth=0.80)
axes[0,2].set_xlim(0.0,5.0)
axes[0,2].set_xlabel("Type of Chest Pain [cp] \n(1 = Typical, 2 = Atypical, 3 = Non-anginal, 4 = Asymptomatic)")
axes[0,2].set_ylabel("Number of Patients")
axes[1,0].hist(df.trestbps)
axes[1,0].set_xlabel("Resting Blood Pressure [restbp]")
axes[1,0].set_ylabel("Number of Patients")
axes[1,1].hist(df.chol)
axes[1,1].set_xlabel("Serum Cholesterol [chol]")
axes[1,1].set_ylabel("Number of Patients")
axes[1,2].hist(df.fbs)
axes[1,2].set_xlabel("Fasting Blood Sugar [fbs] (0 = < 120mg/dl, 1 = > 120mg/dl)")
axes[1,2].set_ylabel("Number of Patients")
axes[2,0].hist(df.restecg)
axes[2,0].set_xlabel("Resting Electrocardiography [restecg] \n(0 =  Normal, 1 =  ST-T wave abnormality, 2= Left ventricular hypertrophy)")
axes[2,0].set_ylabel("Number of Patients")
axes[2,1].hist(df.thalach)
axes[2,1].set_xlabel("Maximum Heart Rate Achieved [thalach]")
axes[2,1].set_ylabel("Number of Patients")
axes[2,2].hist(df.exang)
axes[2,2].set_xlabel("Exercise Induced Angina [exang] (0 = No, 1 = Yes)")
axes[2,2].set_ylabel("Number of Patients")
axes[3,0].hist(df.oldpeak)
axes[3,0].set_xlabel("Exercise Induced ST Depression [oldpeak]")
axes[3,0].set_ylabel("Number of Patients")
axes[3,1].hist(df.slope)
axes[3,1].set_xlabel("Slope of Peak Exercise ST Segment [slope] \n(0 = Downsloping, 1 = Flat, 2 = Upsloping)")
axes[3,1].set_ylabel("Number of Patients")
axes[3,2].hist(df.ca,bins=4,range=(-0.5,3.5),rwidth=0.8)
axes[3,2].set_xlim(-0.7,3.7)
axes[3,2].set_xlabel("Major Vessels colored by Fluoroscopy [ca]")
axes[3,2].set_ylabel("Number of Patients")
axes[4,0].hist(df.thal)
axes[4,0].set_xlabel("Thal \n(1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")
axes[4,0].set_ylabel("Number of Patients")
axes[4,1].hist(df.target,bins=5,range=(-0.5,4.5),rwidth=0.8)
axes[4,1].set_xlim(-0.7,4.7)
axes[4,1].set_xlabel("Heart Disease [num] \n(0 = No Heart Disease, 1 = Heart Disease)")
axes[4,1].set_ylabel("Number of Patients")
axes[4,2].axis("off")

# Marginal feature distributions compared for disease and no-disease (likelihoods)
bins = np.linspace(20, 80, 15)
axes[5,0].hist(df[df.target>0].age.tolist(),bins,color=["crimson"],histtype="step",label="disease",density=True)
axes[5,0].hist(df[df.target==0].age,bins,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[5,0].set_xlabel("Age (years)")
axes[5,0].set_ylim(0.0,0.070)
axes[5,0].legend(prop={'size': 10},loc="upper left")
axes[5,1].hist(df[df.target>0].sex.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[5,1].hist(df[df.target==0].sex,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[5,1].set_xlabel("Sex (0=female,1=male)")
axes[5,1].legend(prop={'size': 10},loc="upper left")
axes[5,2].hist(df[df.target>0].cp.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[5,2].hist(df[df.target==0].cp,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[5,2].set_xlabel("Type of Chest Pain [cp] \n(1 = Typical, 2 = Atypical, 3 = Non-anginal, 4 = Asymptomatic)")
axes[5,2].legend(prop={'size': 10},loc="upper left")
bins = np.linspace(80, 200, 15)
axes[6,0].hist(df[df.target>0].trestbps.tolist(),bins,color=["crimson"],histtype="step",label="disease",density=True)
axes[6,0].hist(df[df.target==0].trestbps,bins,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[6,0].set_xlabel("Resting Blood Pressure [restbp]")
axes[6,0].legend(prop={'size': 10},loc="upper right")
axes[6,1].hist(df[df.target>0].chol.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[6,1].hist(df[df.target==0].chol,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[6,1].set_xlabel("Serum Cholesterol [chol]")
axes[6,1].legend(prop={'size': 10},loc="upper right")
axes[6,2].hist(df[df.target>0].fbs.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[6,2].hist(df[df.target==0].fbs,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[6,2].set_xlabel("Fasting Blood Sugar [fbs] (0 = < 120mg/dl, 1 = > 120mg/dl)")
axes[6,2].legend(prop={'size': 10},loc="upper right")
axes[7,0].hist(df[df.target>0].restecg.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[7,0].hist(df[df.target==0].restecg,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[7,0].set_xlabel("Resting Electrocardiography [restecg] \n(0 =  Normal, 1 =  ST-T wave abnormality, 2= Left ventricular hypertrophy)")
axes[7,0].set_ylim(0.0,4.0)
axes[7,0].legend(prop={'size': 10},loc="upper right")
axes[7,1].hist(df[df.target>0].thalach.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[7,1].hist(df[df.target==0].thalach,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[7,1].set_xlabel("thalach")
axes[7,1].legend(prop={'size': 10},loc="upper left")
axes[7,2].hist(df[df.target>0].exang.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[7,2].hist(df[df.target==0].exang,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[7,2].set_xlabel("exang (0 = No, 1 = Yes)")
axes[7,2].legend(prop={'size': 10},loc="upper right")
axes[8,0].hist(df[df.target>0].oldpeak.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[8,0].hist(df[df.target==0].oldpeak,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[8,0].set_xlabel("oldpeak")
axes[8,0].legend(prop={'size': 10},loc="upper right")
axes[8,1].hist(df[df.target>0].slope.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[8,1].hist(df[df.target==0].slope,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[8,1].set_xlabel("slope \n(0 = Downsloping, 1 = Flat, 2 = Upsloping)")
axes[8,1].legend(prop={'size': 10},loc="upper right")
axes[8,2].hist(df[df.target>0].ca.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[8,2].hist(df[df.target==0].ca,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[8,2].set_xlabel("ca")
axes[8,2].legend(prop={'size': 10},loc="upper right")
axes[9,0].hist(df[df.target>0].thal.tolist(),color=["crimson"],histtype="step",label="disease",density=True)
axes[9,0].hist(df[df.target==0].thal,color=["chartreuse"],histtype="step",label="no disease",density=True)
axes[9,0].set_xlabel("thal \n(1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")
axes[9,0].set_ylim(0.0,2.5)
axes[9,0].legend(prop={'size': 10},loc="upper right")
axes[9,1].axis("off")
axes[9,2].axis("off")
plt.show()


# In[11]:


#Correlation Graphs
sns.pairplot(df)
plt.show()


# In[12]:


#Age v Target Analysis
young_ages=df[(df.age>=29)&(df.age<40)]
middle_ages=df[(df.age>=40)&(df.age<55)]
elderly_ages=df[(df.age>55)]
print('Young Ages :',len(young_ages))
print('Middle Ages :',len(middle_ages))
print('Elderly Ages :',len(elderly_ages))
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[13]:


#Sex v Target Analysis
sns.countplot(df.sex)
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.title('Sex vs Count State')
plt.show()

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Disease"])
plt.ylabel('Frequency')
plt.show()


# In[14]:


#Chest Pain vs Target Analysis
sns.countplot(df.cp)
plt.xlabel('Chest Type (1 = Typical angina, 2 = Atypical angina, 3 =  Non-anginal pain, 4 =  Asymptomatic)')
plt.ylabel('Count')
plt.title('Chest Type vs Count State')
plt.show()

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Chest Pain')
plt.xlabel('cp (1 = Typical angina, 2 = Atypical angina, 3 =  Non-anginal pain, 4 =  Asymptomatic)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Disease"])
plt.ylabel('Frequency')
plt.show()


# In[15]:


#Resting Blood Pressure (trestbps) vs Target
ideal_blood_pressure=df[(df.trestbps>=90)&(df.trestbps<120)]
pre_high_blood_pressure=df[(df.trestbps>=120)&(df.trestbps<140)]
high_blood_pressure=df[(df.trestbps>140)]
print('Ideal Blood Pressure :',len(ideal_blood_pressure))
print('Pre-High Blood Pressure :',len(pre_high_blood_pressure))
print('High Blood Pressure :',len(high_blood_pressure))
sns.barplot(x=['Ideal','Pre-High', 'High'],y=[len(ideal_blood_pressure),len(pre_high_blood_pressure),len(high_blood_pressure)])
plt.xlabel('Blood Pressure Range')
plt.ylabel('Blood Pressure Counts')
plt.title('Blood Pressure State in Dataset')
plt.show()

pd.crosstab(df.trestbps,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Resting Blood Pressure')
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Frequency')
plt.show()

#Resting Blood Pressure (trestbps) vs Age for Target
plt.scatter(x=df.trestbps[df.target==1], y=df.age[(df.target==1)], c="red")
plt.scatter(x=df.trestbps[df.target==0], y=df.age[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Resting Blood Pressure")
plt.ylabel("Age")
plt.show()


# In[16]:


#Serum Cholestrol (chol) vs Target
pd.crosstab(df.chol,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Serum Cholestrol')
plt.xlabel('Serum Cholestrol')
plt.ylabel('Frequency')
plt.show()

#Serum Cholestrol (chol) vs Age for Target
plt.scatter(x=df.chol[df.target==1], y=df.age[(df.target==1)], c="red")
plt.scatter(x=df.chol[df.target==0], y=df.age[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Serum Cholestrol")
plt.ylabel("Age")
plt.show()


# In[17]:


#Fasting Blood Sugar (fbs) vs Target
pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Fasting Blood Sugar')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Frequency')
plt.show()

#Fasting Blood Sugar (fbs) vs Age for Target
plt.scatter(x=df.fbs[df.target==1], y=df.age[(df.target==1)], c="red")
plt.scatter(x=df.fbs[df.target==0], y=df.age[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Fasting Blood Sugar")
plt.ylabel("Age")
plt.show()


# In[18]:


#Maximum heart rate achieved during thallium stress test (Thalach) vs Age for Target
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[19]:


#Thallium stress test result (Thal) vs Target
sns.countplot(df.thal)
plt.xlabel('thal (1 = Fixed Defect, 2 = Normal, 3 = Reversible Defect)')
plt.ylabel('Count')
plt.title('Thal vs Count State')
plt.show()

pd.crosstab(df.thal,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Thallium Stress Test Result')
plt.xlabel('thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)')
plt.xticks(rotation=0)
plt.legend(["No Disease", "Disease"])
plt.ylabel('Frequency')
plt.show()


# In[20]:


#Maximum Heart Rate during Thallium Stress Test (thalach) vs Age for Target
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[21]:


#Old peak vs Slope for Target
plt.scatter(x=df.slope[df.target==1], y=df.oldpeak[(df.target==1)], c="red")
plt.scatter(x=df.slope[df.target==0], y=df.oldpeak[(df.target==0)])
plt.legend(["Disease", "No Disease"])
plt.xlabel("Slope")
plt.ylabel("Old Peak")
plt.show()


# In[22]:


def intrinsic_discrepancy(x,y):
    assert len(x)==len(y)
    sumx = sum(xval for xval in x)
    sumy = sum(yval for yval in y)
    id1  = 0.0
    id2  = 0.0
    for (xval,yval) in zip(x,y):
        if (xval>0) and (yval>0):
            id1 += (float(xval)/sumx) * np.log((float(xval)/sumx)/(float(yval)/sumy))
            id2 += (float(yval)/sumy) * np.log((float(yval)/sumy)/(float(xval)/sumx))
    return min(id1,id2)


# In[23]:


from tabulate import tabulate
from collections import Counter

#Calculating instrinsic values and histograms for each
intrinsic_values = []

hist,bin_edges   = np.histogram(df.age,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].age,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].age,bins=bin_edges,density=False)
intrinsic_values.append(('age', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.sex,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].sex,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].sex,bins=bin_edges,density=False)
intrinsic_values.append(('sex', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.cp,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].cp,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].cp,bins=bin_edges,density=False)
intrinsic_values.append(('cp', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.trestbps,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].trestbps,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].trestbps,bins=bin_edges,density=False)
intrinsic_values.append(('trestbps', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.chol,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].chol,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].chol,bins=bin_edges,density=False)
intrinsic_values.append(('chol', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.fbs,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].fbs,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].fbs,bins=bin_edges,density=False)
intrinsic_values.append(('fbs', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.restecg,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].restecg,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].restecg,bins=bin_edges,density=False)
intrinsic_values.append(('restecg', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.thalach,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].thalach,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].thalach,bins=bin_edges,density=False)
intrinsic_values.append(('thalach', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.exang,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].exang,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].exang,bins=bin_edges,density=False)
intrinsic_values.append(('exang', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.oldpeak,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].oldpeak,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].oldpeak,bins=bin_edges,density=False)
intrinsic_values.append(('oldpeak', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.slope,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].slope,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].slope,bins=bin_edges,density=False)
intrinsic_values.append(('slope', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.ca,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].ca,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].ca,bins=bin_edges,density=False)
intrinsic_values.append(('ca', intrinsic_discrepancy(hist1,hist2)))
hist,bin_edges   = np.histogram(df.thal,density=False)
hist1,bin_edges1 = np.histogram(df[df.target>0].thal,bins=bin_edges,density=False)
hist2,bin_edges2 = np.histogram(df[df.target==0].thal,bins=bin_edges,density=False)
intrinsic_values.append(('thal', intrinsic_discrepancy(hist1,hist2)))

intrinsic_values = sorted(intrinsic_values, key=lambda value: value[1], reverse=True)   # sort by intrinsic discrepancy

#Displaying results in a sorted manner
print(tabulate([[intrinsic_values[0][0], intrinsic_values[0][1]], 
                [intrinsic_values[1][0], intrinsic_values[1][1]], 
                [intrinsic_values[2][0], intrinsic_values[2][1]],
                [intrinsic_values[3][0], intrinsic_values[3][1]],
                [intrinsic_values[4][0], intrinsic_values[4][1]],
                [intrinsic_values[5][0], intrinsic_values[5][1]],
                [intrinsic_values[6][0], intrinsic_values[6][1]],
                [intrinsic_values[7][0], intrinsic_values[7][1]],
                [intrinsic_values[8][0], intrinsic_values[8][1]],
                [intrinsic_values[9][0], intrinsic_values[9][1]],
                [intrinsic_values[10][0], intrinsic_values[10][1]],
                [intrinsic_values[11][0], intrinsic_values[11][1]],
                [intrinsic_values[12][0], intrinsic_values[12][1]]], 
               headers=['Attribute', 'Intrinsic Discrepancy'], 
               tablefmt='orgtbl'
              ))


# In[24]:


#Correlation Heatmaps / Matrix
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!

