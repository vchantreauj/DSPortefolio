#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:25:01 2019

@author: Vanessa Chantreau

this file is an example of the use of SVC model to predict the cancer from dataset "breast cancer wisconsin"
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling

# IMPORT THE DATASET
#-------------------
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

cancer.keys()

# info on the dataset
print(cancer['DESCR'])

# here is the target y
print(cancer['target'])

# here are the feature names and values
columns = cancer['feature_names']
print(columns)
print(cancer['data'])

# create a pandas dataframe from the dataset so the analyse will be easier
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns = np.append(columns,['target']))

# first look to the data
pd.options.display.max_columns = None
df_cancer.describe()
df_cancer.info()
df_cancer.isnull().sum()
# the data are already clean: right type and no missing values

# what about the target balance
df_cancer['target'].value_counts()
# the target seams balanced enought

# DATA VISUALIZATION
#-------------------
df_copy = df_cancer.copy()
# export file required if running code in spyder
# direct use of ProfileReport otherwise
plts = pandas_profiling.ProfileReport(df_copy)
plts.to_file('profile_report.html')

# there are some highly correlated features
# let's drop them according to the profileReport
df_cancer_ind = df_cancer.drop(['mean concavity','mean perimeter','mean radius','perimeter error','radius error','worst area','worst concave points','worst perimeter','worst radius','worst texture'],axis=1)

# check the correlation of the features left with the target
def my_corr_mat(df,target):
    nb_var = df.shape[1]
    corrmat = df.corr()
    cols = corrmat.nlargest(nb_var, target)[target].index
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    # increase the default heatmap size
    fig, ax = plt.subplots(figsize=(10,10))  
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values, ax=ax)
    plt.show()

my_corr_mat(df_cancer_ind,'target')

def display_corr_with_col(df,col):
    corr_matrix = df.corr()
    corr_type = corr_matrix[col].copy()
    abs_corr_type = corr_type.apply(lambda x:abs(x))
    desc_corr_values = abs_corr_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig,ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values,y_values)
    ax.set_title('the correlation of all features with {}'.format(col),fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values,xlabels,rotation='vertical')
    plt.show()

display_corr_with_col(df_cancer_ind,'target')

# check the correlation of the features between one another
sns.pairplot(df_cancer_ind,hue = 'target',vars=['mean texture', 'mean area', 'mean smoothness', 'mean compactness','mean concave points', 'mean symmetry', 'mean fractal dimension'])
# no highly correlation among these ones
sns.pairplot(df_cancer_ind,hue = 'target',vars=['texture error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error','fractal dimension error'])
# no highly correlation among these ones too; the clusters drawn by the target is less obvious
sns.pairplot(df_cancer_ind,hue = 'target',vars=['worst smoothness', 'worst compactness','worst concavity', 'worst symmetry', 'worst fractal dimension'])
# no highly correlation among these ones; the clusters drawn by the target is less obvious too

# to have some details on the features scatterplot
sns.scatterplot(x='mean texture', y = 'mean concave points', hue = 'target', data = df_cancer_ind)

# FEATURES SELECTION
#---------------------
X = df_cancer_ind.iloc[:, :-1].values
y = df_cancer_ind.iloc[:, -1].values

# Backward elimination
import statsmodels.formula.api as sm
# need a global col_rest var
col_rest = []
def backwardElimination(x, sl,col_name):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    print('for p_value '+str(round(maxVar,3))+' delete column '+col_name[j])
                    # delete name from global var list header
                    col_rest.pop(j)
    print(regressor_OLS.summary())
    return x

col_name = list(df_cancer_ind.columns)
col_rest = col_name.copy()
SL = 0.05
X_opt = X
X_Modeled = backwardElimination(X_opt, SL, col_name)

# MODEL TRAINING
#------------------
from sklearn.model_selection import train_test_split
# X_all if we want to test our model without the features selection above
X_all = df_cancer_ind.iloc[:, :-1].values
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size = 0.2, random_state = 5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVC model
from sklearn.svm import SVC
# first trial with the SVC default values
classifier = SVC(C = 1,kernel='rbf', random_state = 5, gamma = 0.5)
classifier.fit(X_train, y_train)

# accuracy of the model
from sklearn.metrics import confusion_matrix, classification_report
y_pred = classifier.predict(X_test)
confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# this first model obtain an accuracy of 0.78 which is... not good !
# let's try to improve our model.
# for have a real idea of our model quality, we'll use the cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X = X_train,y = y_train, cv=10)
print('accuracy '+str(round(accuracies.mean(),3))+' +/- '+str(round(accuracies.std(),3)))
# the cv accuracy is better that the "one shot" one : 0.84 +/- 0.1
# can we do better ?

# IMPROVING THE MODEL
#--------------------
# parameters grid
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,100],'kernel':['rbf'],'gamma':[0.8,0.7,0.6,0.5,0.1,0.01,0.05]}]
gridsearch = GridSearchCV(estimator=classifier, param_grid = parameters, scoring='accuracy',cv=10,n_jobs=-1)
gridsearch = gridsearch.fit(X_train,y_train)
best_accuracy = gridsearch.best_score_
best_param = gridsearch.best_params_
print(best_accuracy)
print(best_param)

# so our best model is the next one with 98% of accuracy !
classifier = SVC(C = 10,kernel='rbf', random_state = 5, gamma = 0.01)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# we have one type two error


