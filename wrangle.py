import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import ols
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_recall_curve, auc
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import env
import wrangle
import env
import os
#-----------------------------------------------------------------------------------------------------------
#acquire 


def acquire_bank():
    df = pd.read_csv('Bank Customer Churn Prediction.csv')
    return df

#------------------------------------------------------------------------------------------------------------
#preparation

def prepare_bank(df):
    '''
    This function takes in dataframe and returns clean data after dropping unnecessary column and encodingthe.
    '''
    #dropping unnecesary columns
    df = df.drop(['customer_id'], axis =1)
    #encoding categorical column for country
    dummies = pd.get_dummies(df.country,drop_first = False)
    df=pd.concat([df,dummies],axis=1)
    #encoding categorical column for gender
    dummies = pd.get_dummies(df.gender,drop_first = False)
    df=pd.concat([df,dummies],axis=1)
    return df

#-----------------------------------------------------------------------------------------------------------
#split data

# this function is splitting data to train, validate, and test to avoid data leakage
def my_train_test_split(df,churn):
    '''
    This function performs split on The bank churn data, stratify churn.
    Returns train, validate, and test dfs.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df["churn"])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train["churn"])
    print(f'Train=',train.shape) 
    print(f'Validate=',validate.shape) 
    print(f'Test=',test.shape) 
    return train, validate, test

#-----------------------------------------------------------------------------------------------------------
#graphs and visualizations

def graph_0(train):
    '''
    This function takes in train and returns the visualization on a pie chart the percentage of churn vs not churn
    ''' 
    sizes = [train.churn[train['churn']==1].count(), train.churn[train['churn']==0].count()]
    labels = ['Churned', 'Not Churned']
    figure, axes = plt.subplots(figsize=(8, 6))
    axes.pie(sizes, labels=labels,shadow=True,autopct = '%1.0f%%')
    plt.legend()
    plt.title("Churned VS Not Churned", size = 15)
    plt.show()

#---------------------------------------------------------------------------------------------------------------

#UNIVARIATE ANALYSIS


#ploting quantitative variables
def plot_quant_vars(train):
    '''
    This function takes in train and returns the visualization for quantitative variables.
    '''
    plt.figure(figsize = (20,25))

    sns.set(color_codes = True)

    plt.subplot(3,2,1)
    sns.distplot(train['credit_score'])

    plt.subplot(3,2,2)
    sns.distplot(train['age'])

    plt.subplot(3,2,3)
    sns.distplot(train['balance'], kde = True)

    plt.subplot(3,2,4)
    sns.distplot(train['tenure'], kde = True)
    plt.show()

    plt.subplot(3,2,5)
    sns.distplot(train['estimated_salary'], kde = True)
    plt.show() 

#plot heatmap
def graph_1(train):
    '''
    This function takes in train and returns the visualization on a heatmap checking correlation between
     variables to churn.
    '''
    correlation = train.corr()[["churn"]].sort_values(by='churn', ascending=False)
    plt.figure(figsize = (14,7))
    sns.heatmap(correlation, annot = True,cmap='Purples')
    plt.show()

#---------------------------------------------------------------------------------------------------------------------

#BIVARIATE ANALYSIS
#ploting categorical variables
def cat_vis(train, col):
    '''
    This function takes in train and returns the visualization for analysing categorical variables 
    and their relationship to churn
    '''
    plt.title('Checking if there is a relationship between churn rate and '+col)
    sns.barplot(x=col, y='churn', data=train)
    churn_rate = train.churn.mean()
    plt.axhline(churn_rate, label='churn rate')
    plt.legend()
    plt.show()

def cat_test(train, col):
    alpha = 0.05
    null_hyp = col+' and churn rate are independent'
    alt_hyp = 'There is a relationship between churn rate and '+col
    observed = pd.crosstab(train.churn, train[col])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print('We reject the null hypothesis that', null_hyp)
        print(alt_hyp)
    else:
        print('We fail to reject the null hypothesis that', null_hyp)
        print('There appears to be no relationship between churn rate and '+col)

def cat_analysis(train, col):
    cat_vis(train, col)
    cat_test(train, col)


#----------------------------------------------------------------------------------------------------------------
#modeling
def getting_(train,validate,test):
    '''
    This function takes in train and defines x features to y target into train, validate and test
    '''
    #X will be features
    #y will be our target variable
    features = ["France","Germany", "Spain", "Male","Female", "balance","age","estimated_salary"]

    X_train = train[features]
    y_train = train.churn
    X_validate = validate[features]
    y_validate = validate.churn
    X_test = test[features]
    y_test = test.churn
    #return X_train.head(2)


 
