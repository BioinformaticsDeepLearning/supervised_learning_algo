#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:11:30 2022

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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from statistics import mean, stdev 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold 

# =============================================================================
# ################### Load dataset (year 1,2,3) ###############################
# =============================================================================
df1 = pd.read_csv('1year.csv')
df2 = pd.read_csv('2year.csv')
df3 = pd.read_csv('3year.csv')
frames = [df1, df2, df3]
DF = pd.concat(frames)
#count1 = (DF['class'] == 0).sum()
#count2 = (DF['class'] == 1).sum()

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
# ######################### Undersampling class count #########################
# =============================================================================
class_count_0, class_count_1 = df['class'].value_counts()
class_0 =df[df['class'] == 0]
class_1 = df[df['class'] == 1]
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)
class_0_under = class_0.sample(class_count_1)
test_under = pd.concat([class_0_under, class_1], axis=0)
print("total class of 1 and0:",test_under['class'].value_counts())
test_under['class'].value_counts().plot(kind='bar', cmap='Set1', title='count (target)')
test_under.to_excel('test_under.xlsx', index = False)
read_file = pd.read_excel ("test_under.xlsx")
read_file.to_csv ("test_under.csv", 
                  index = None,
                  header=True)
dfm = pd.DataFrame(pd.read_csv("test_under.csv"))

#Generating X and Y#
X = dfm.loc[:, dfm.columns!='class']
Y = dfm.loc[:, 'class']

#Drop columns with more than 0.95 correlation value#
corr_matrix = X.corr('pearson')
corr_matrix.to_excel('corr_matrix.xlsx')
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X.drop(to_drop, axis=1, inplace=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
f, ax = plt.subplots(figsize=(50, 40), dpi= 300)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap="PiYG", vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .2}, annot = True)

# Feature scaling #
from sklearn.preprocessing import Normalizer
x = Normalizer().fit_transform(X)

# =============================================================================
# ##################  Machine learning models #################################
# =============================================================================
# Split the dataset into training and test dataset
x_train, x_test, Y_train, Y_test = train_test_split(normalized_X, Y, random_state=4, stratify= Y)

# =============================================================================
# # 1. Logistics regression#
# =============================================================================
LogR_classifier = LogisticRegression(C=1)
LogR_classifier.fit(x_train, Y_train)
LogR_prediction = LogR_classifier.predict(x_test)
LogR_report = classification_report(Y_test, LogR_prediction)
print(LogR_report)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(x, Y): 
    x_train_fold, x_test_fold = x_train, x_test 
    y_train_fold, y_test_fold = Y_train, Y_test
    LogR_classifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(LogR_classifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%') 

# =============================================================================
# # 2. Support Vector classifier#
# =============================================================================
svclassifier = SVC(kernel='rbf')
svclassifier.fit(x_train, Y_train)
svc_predictions = svclassifier.predict(x_test)
svc_report = classification_report(Y_test, svc_predictions)
print(svc_report)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(x, Y): 
    x_train_fold, x_test_fold = x_train, x_test 
    y_train_fold, y_test_fold = Y_train, Y_test
    svclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(svclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

# =============================================================================
# # 3. K nearest neighbor#
# =============================================================================
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(normalized_X_train, Y_train)
Y_pred_knn = knn.predict(normalized_X_test)
knn_report = classification_report(Y_test, Y_pred_knn)
print(knn_report)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(x, Y): 
    x_train_fold, x_test_fold = x_train, x_test 
    y_train_fold, y_test_fold = Y_train, Y_test
    knn.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(knn.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

# =============================================================================
# # 4. Gaussian Naive Bayes#
# =============================================================================
NBclassifier = GaussianNB()
NBclassifier.fit(x_train,Y_train)
NB_predictions = NBclassifier.predict(x_test)
NB_report = classification_report(Y_test, NB_predictions)
print(NB_report)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(x, Y): 
    x_train_fold, x_test_fold = x_train, x_test 
    y_train_fold, y_test_fold = Y_train, Y_test
    NBclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(NBclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

# =============================================================================
# # 5. Neural network #
# =============================================================================
NNclassifier = MLPClassifier().fit(x_train, Y_train)
NN_predictions = NNclassifier.predict(x_test)
NN_report = classification_report(Y_test, NN_predictions)
print(NN_report)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(x, Y): 
    x_train_fold, x_test_fold = x_train, x_test 
    y_train_fold, y_test_fold = Y_train, Y_test
    NNclassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(NNclassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')
print('Std:', stdev(lst_accu_stratified), '%')

# =============================================================================
# # 6. Random Forest
# =============================================================================
RFClassifier= RandomForestClassifier(n_estimators= 20, max_depth=3, random_state=0)
RFClassifier.fit(x_train,Y_train)
RFC_predictions = RFClassifier.predict(x_test)
RFC_report = classification_report(Y_test, RFC_predictions)
print(RFC_report)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1) 
lst_accu_stratified = [] 
for train_index, test_index in skf.split(x, Y): 
    x_train_fold, x_test_fold = x_train, x_test 
    y_train_fold, y_test_fold = Y_train, Y_test
    RFClassifier.fit(x_train_fold, y_train_fold) 
    lst_accu_stratified.append(RFClassifier.score(x_test_fold, y_test_fold)) 
print('List of possible accuracy:', lst_accu_stratified) 
print('\nMaximum Accuracy That can be obtained from this model is:', 
      max(lst_accu_stratified)*100, '%') 
print('\nMinimum Accuracy:', 
      min(lst_accu_stratified)*100, '%') 
print('\nOverall Accuracy:', 
      mean(lst_accu_stratified)*100, '%')

# =============================================================================
# ####################### CV Model Evaluation #################################
# =============================================================================
#models = pd.DataFrame({
#    'Model': ['Logistics regression', 'Support Vector classifier', 'K nearest neighbor', 
#              'Gaussian Naive Bayes', 'Perceptron', 'Random Forest'],
#    'Score': [LogR_report, svc_report, knn_report, 
#              NB_report, NN_report, RFC_report]})
#models.sort_values(by='Score', ascending=False)

model_CV = pd.DataFrame({
    'Model': ['Logistics regression', 'Support Vector classifier', 'K nearest neighbor', 
              'Gaussian Naive Bayes', 'Neural Network', 'Random Forest'],
    'CV10Folf_Score': [log_score, SVC_score, knn_score, 
              gaussian_score, perceptron_score, RF_scores]})
model_CV.sort_values(by='CV10Folf_Score', ascending=False)