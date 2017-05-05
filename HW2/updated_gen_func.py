import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
pd.set_option('display.width', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from tabulate import tabulate
import csv
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
sns.set(font_scale=1.4)#for label size
import numpy as np 
import matplotlib.pyplot as plt
import pylab as pl
import sys
import random
from sklearn import svm, ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import *
from sklearn.feature_selection import RFE
from sklearn.grid_search import ParameterGrid
from multiprocessing import Pool
from functools import partial
import time
plt.rcParams["figure.figsize"] = [18.0, 8.0]

def read_data(file_name, peek=False): #to be added to general functions
	'''
	Read in data and return a pandas df
	'''
	if peek:
		return pd.read_csv(file_name, header=0).head()
	return pd.read_csv(file_name, header=0)

def summarize(data):
    '''
    Given a pandas dataframe, print dataframe statistics, correlation, and missing data.
    '''
    print("Column Names")
    print(tabulate(pd.DataFrame(data.columns.values), tablefmt='fancy_grid'))
    print()
    print('dataframe shape: ', "\n", "\n",data.shape)
    print()
    print('**** statistics: ', "\n","\n", data.describe(include='all'))
    print()
    print('Mode per columns: ', "\n", "\n",data.mode())
    print()
    print()
    print("Null values per columns:")
    print()
    print(tabulate(pd.DataFrame(data.isnull().sum()), tablefmt='fancy_grid'))
    print()
    data.hist(figsize=(20,20))

def split_data(data):
    """
    Assumes that label is in the second column of a CSV file.
    First column is a label of some kind (i.e PersonID)
    """
    splitting_set = data.copy(deep=True) #protect integrity of initial dataset.
    col_names = splitting_set.columns
    del splitting_set[splitting_set.iloc[:, 0].name] #personID
    y = splitting_set.iloc[:, 0].values#our label variable
    X = splitting_set.drop(splitting_set.iloc[:, 0].name, axis = 1).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                       stratify=y)
    X_train = pd.DataFrame(X_train, columns=col_names[2:])
    X_test = pd.DataFrame(X_test, columns=col_names[2:])
    y_train = pd.DataFrame(y_train, columns=np.array([col_names[1]]))
    y_test = pd.DataFrame(y_test, columns=np.array([col_names[1]]))
    
    print("X_train Dimensions: {}".format(X_train.shape))
    print("X_test Dimensions: {}".format(X_test.shape))
    print("y_train Dimensions: {}".format(y_train.shape))
    print("y_test Dimensions: {}".format(y_test.shape))
    
    return X_train, X_test, y_train, y_test

def analyze(training_x, test_x):
    """
    This function is dataset specific yet a few alterations will allow it to become fully dynamic.

    This functions' main use is to present key statistics for different categories in a 'pretty print' manner.

    Visual inspection of the output of this function will assist in discerning what the next steps should be.
    
    Inputs:
        training_x: training feature set - Pandas Core DataFrame type
        test_x: testing features set - Pandas Core DataFrame type
        col_name: The column name in the dataframe to be analyzed
    
    Outputs:
        Summary Statistics.

    """
    training_x.hist(figsize=(20,20))
    pl.suptitle("Training Set", fontsize=20)
    plt.show()
    print()
    test_x.hist(figsize=(20,20))
    pl.suptitle("Testing Set", fontsize=20)
    plt.show()
    for i in training_x.columns:
        print("-------------Analysis for {} feature-------------".format(i))
        print()
        if i == "RevolvingUtilizationOfUnsecuredLines":
            print("Training Set: ")
            print()
            print("Max: ", training_x.describe()[i]['max'])
            print("Mean", training_x.describe()[i]['mean'])
            print()
            print("Missing Values: ", sum(pd.isnull(training_x[i])))
            print()
            print("Testing Set: ")
            print()
            print("Max: ", test_x.describe()[i]['max'])
            print("Mean: ", test_x.describe()[i]['mean'])
            print()
            print("Missing Values: ", sum(pd.isnull(test_x[i])))
            print()
        elif i == 'age':
            print("Training Set: ")
            print()
            print(training_x.describe()[i]['max'])
            print(training_x.describe()[i]['min'])
            print(training_x.describe()[i]['mean'])
            print()
            print("Missing Values: ", sum(pd.isnull(training_x[i])))
            print()
            print("Testing Set: ")
            print()
            print(test_x.describe()[i]['max'])
            print(test_x.describe()[i]['min'])
            print(test_x.describe()[i]['mean'])
            print()
            print("Missing Values: ", sum(pd.isnull(test_x[i])))
            print()
        elif i == 'zipcode' or i == "Zipcode" or i == "Zip code" or i == "Zip Code"\
        or i == "Zip" or i == 'zip':
            print("Training Set: ")
            print()
            print(training_x[i].unique()) #looks like our zip codes are good.
            print()
            print("Missing Values: ", sum(pd.isnull(training_x[i])))
            print()
            print('Test Set: ')
            print()
            print(test_x[i].unique())
            print()
            print("Missing Values: ", sum(pd.isnull(test_x[i])))
            print()
        elif i == "MonthlyIncome":
            print("Training Set: ")
            print()
            print(training_x.describe()[i])
            print()
            print("Mean: ", training_x[i].mean())
            print()
            print("Median: ", training_x[i].median())
            print()
            print("Missing Values: ", sum(pd.isnull(training_x[i])))
            print()
            print('Test Set: ')
            print()
            print(test_x.describe()[i])
            print()
            print("Mean: ", test_x[i].mean())
            print()
            print("Median: ", test_x[i].median())
            print()
            print("Missing Values: ", sum(pd.isnull(test_x[i])))
            print()
        else:
            print("Training Set: ")
            print()
            print(training_x.describe()[i])
            print()
            print("Missing Values: ", sum(pd.isnull(training_x[i])))
            print()
            print('Test Set: ')
            print()
            print(test_x.describe()[i])
            print()
            print("Missing Values: ", sum(pd.isnull(test_x[i])))
            print()

def combine(X_train, X_test, y_train, y_test): #put into gen func
    combined_y = y_train.append(y_test)
    combined_x = X_train.append(X_test)
    combined_dataset = pd.concat([combined_y, combined_x], axis=1)
    
    return combined_dataset

def visualize_distribution(df):
    income_bins = range(0, 200000, 10000)
    income_bucket = generate_bins(df, 'MonthlyIncome', income_bins)
    average_visualisation_groups(df, [income_bucket, "SeriousDlqin2yrs"], income_bucket)
    age_bins = range(20,120, 5)
    age_bucket = generate_bins(df, 'age', age_bins)
    average_visualisation_groups(df, [age_bucket, "SeriousDlqin2yrs"], age_bucket)
    average_visualisation_groups(df, ['NumberOfDependents', 'SeriousDlqin2yrs'], 'NumberOfDependents')
    average_visualisation_groups(df, [age_bucket, "RevolvingUtilizationOfUnsecuredLines"], age_bucket)
    average_visualisation_groups(df, [age_bucket, "DebtRatio"], age_bucket)
    income_bins = range(0, 10000, 500)
    income_bucket = generate_bins(df, 'MonthlyIncome', income_bins)
    average_visualisation_groups(df, [income_bucket, "DebtRatio"], income_bucket)

def correlation_matrix(df):
    ax = plt.axes()
    sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
    ax.set_title("Correlation Matrix")

def create_binary(df, column, remove_original_col = False): #just put in gen func - no use.
    '''
    Takes a Dataframe, a column in the dataframe and returns dummy variables
    for that column. Optional argument to remove origianl column from dataframe
    after dummy variable creation.
    '''
    
    if remove_original_col:
        df_col = pd.get_dummies(df[column])
        binary_df = pd.concat([df, df_col], axis=1)
        del binary_df[column]
        return binary_df
    else:
        df_col = pd.get_dummies(df[column])
        binary_df = pd.concat([df, df_col], axis=1)
        return binary_df
def generate_bins(data, column, bins):
    '''
    Takes a continuous variable and generates it's bins which are then stored 
    as a new column in a pandas dataframe.
    
    Input:
        data: Dataframe
        column: Column from dataframe
        bins: bin size
    Output:
        Add's this feature to our data dataframe with the bin sizes specified.
    
    '''
    temp_feature = 'bins_' + str(column)

    data[temp_feature] = pd.cut(data[column], bins=bins, include_lowest=True, labels=False)

    return temp_feature

def average_visualisation_groups(data, cols, group_by_col):

    '''
    Takes a dataframe, columns in that dataframe and a column to group by
    and generates a plot with the mean of the group.
    
    Inputs:
        data: pandas dataframe with our data
        cols: in list form, state the columns of interest to visualise
        group_by_col: list of columns to be contrasted. First element in list should match 
                      group_by_col
    Outputs:
        Plot graphic contrasting two columns from the pandas dataframe
    '''

    data[cols].groupby(group_by_col).mean().plot()