#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:14:29 2022

@author: alishaparveen
"""

# =============================================================================
# ####################### Import libraries ####################################
# =============================================================================
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# =============================================================================
# ################### Load dataset (year 1,2,3) ###############################
# =============================================================================
df1 = pd.DataFrame(pd.read_csv('1year.csv'))
df2 = pd.DataFrame(pd.read_csv('2year.csv'))
df3 = pd.DataFrame(pd.read_csv('3year.csv'))
frames = [df1, df2, df3]
DF = pd.concat(frames)
####### Data Exploration ###########
DF.replace("?", np.nan, inplace = True)
sns.heatmap(DF.isnull(), cbar=False)
df=DF.dropna()
sns.heatmap(df.isnull(), cbar=False)
#visualize the target variable
g = sns.countplot(df['class'])
g.set_xticklabels(['0','1'])
plt.show()

# =============================================================================
# ######################### Oversampling class count #########################
# =============================================================================
# import library
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X = df.loc[:, df.columns!='class']
Y = df.loc[:, 'class']
x, y = smote.fit_resample(X, Y)

#df.to_csv ("df.csv", index = False, header=True)
x.to_excel('x.xlsx', index = False)
read_file = pd.read_excel ("x.xlsx")
read_file.to_csv ("x.csv", 
                  index = None,
                  header=True)
x = pd.read_csv("x.csv")
#Drop columns with more than 0.95 correlation value#
corr_matrix = x.corr('pearson')
corr_matrix.to_excel('corr_matrix.xlsx')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
x.drop(to_drop, axis=1, inplace=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
f, ax = plt.subplots(figsize=(50, 40), dpi= 300)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap="PiYG", vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .2}, annot = True)

# Feature scaling #
normalized_X = preprocessing.normalize(x)

# =============================================================================
# ##################  Machine learning models #################################
# =============================================================================
# Split the dataset into training and test dataset
normalized_X_train, normalized_X_test, Y_train, Y_test = train_test_split(normalized_X, Y, random_state=1)

# 1. Logistics regression#
log_reg = LogisticRegression()
log_reg.fit(normalized_X_train, Y_train)
y_pred = log_reg.predict(normalized_X_test)
confusion_matrix(Y_test, y_pred)
acc_log = round(log_reg.score(normalized_X_train, Y_train) * 100, 2)
acc_log

# 2. Support Vector classifier#
svc = SVC()
svc.fit(normalized_X_train, Y_train)
Y_pred = svc.predict(normalized_X_test)
acc_svc = round(svc.score(normalized_X_train, Y_train) * 100, 2)
acc_svc

# 3. K nearest neighbor#
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(normalized_X_train, Y_train)
Y_pred = knn.predict(normalized_X_test)
acc_knn = round(knn.score(normalized_X_train, Y_train) * 100, 2)
acc_knn

# 4. Gaussian Naive Bayes#
gaussian = GaussianNB()
gaussian.fit(normalized_X_train, Y_train)
Y_pred = gaussian.predict(normalized_X_test)
acc_gaussian = round(gaussian.score(normalized_X_train, Y_train) * 100, 2)
acc_gaussian

# 5. Perceptron #
perceptron = Perceptron()
perceptron.fit(normalized_X_train, Y_train)
Y_pred = perceptron.predict(normalized_X_test)
acc_perceptron = round(perceptron.score(normalized_X_train, Y_train) * 100, 2)
acc_perceptron

# 6. Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(normalized_X_train, Y_train)
Y_pred = random_forest.predict(normalized_X_test)
random_forest.score(normalized_X_train, Y_train)
acc_random_forest = round(random_forest.score(normalized_X_train, Y_train) * 100, 2)
acc_random_forest

# =============================================================================
# ###################   Model Evaluation ######################################
# =============================================================================
models = pd.DataFrame({
    'Model': ['Logistics regression', 'Support Vector classifier', 'K nearest neighbor', 
              'Gaussian Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Stochastic Gradient Decent', 'Decision Tree', 
              'Random Forest'],
    'Score': [acc_log, acc_svc, acc_knn, 
              acc_gaussian, acc_perceptron, acc_linear_svc, 
              acc_sgd, acc_decision_tree, acc_random_forest]})
models.sort_values(by='Score', ascending=False)

