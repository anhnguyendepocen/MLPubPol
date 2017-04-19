#Ibrahim Gabr
import numpy as np
np.set_printoptions(precision=2)
import itertools
import pandas as pd
import pylab as pl
pd.options.mode.chained_assignment = None
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
pl.rcParams['figure.figsize'] = (20, 6)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def summarize(data):
    '''
    Provides Summary Statistics for a Pandas Dataframe.

    Not used in the assignment, but may be useful in the future.

    This function is mean to describe key statsitics of a DataFrame in a 'pretty' manner.
    '''
    pd.set_option('display.width', 20)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print('Feature Names:')
    print()
    for i in data.columns.values:
        print(i)
    print()
    print('Dimensions of DataFrame:')
    print()
    print(data.shape)
    print()
    print('Statistical Summary of Dataset:')
    print()
    print(data.describe(include='all'))
    print()
    print('Missing Values:')
    print()
    print(data.isnull().sum())


def visualize_confusion_matrix(cnfm, labels,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Inputs:
        cnfm: Confusion matrix
        labels = list of 2 labels
    Outputs:
        Confusion Matrix Graphic.
    """
    plt.imshow(cnfm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    if normalize:
        cnfm = cnfm.astype('float') / cnfm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Regular Confusion matrix')

    print(cnfm)
    print()

    split_threshold = cnfm.max() / 2.
    for i, j in itertools.product(range(cnfm.shape[0]), range(cnfm.shape[1])):
        plt.text(j, i, cnfm[i, j],
                 horizontalalignment="center",
                 color="black" if cnfm[i, j] > split_threshold else "blue")

    plt.tight_layout()
    plt.ylabel('Expected Label')
    plt.xlabel('Predicted label')
    plt.show()

def generate_bins(data, column, bins):
    '''
    Takes a continuous variable and generates it's bins which are then stored 
    as a new column.
    '''
    temp_feature = 'bins_' + str(column)

    data[temp_feature] = pd.cut(data[column], bins=bins)

    return temp_feature

def average_visualisation_groups(data, cols, group_by_col):

    '''
    Takes a dataframe, columns in that dataframe and a column to group by
    and generates a plot with the mean of the group.
    '''

    data[cols].groupby(group_by_col).mean().plot()

def analyze(training_x, test_x, col_name):
    """
    This function is dataset specific yet a few alterations will allow it to become fully dynamic.

    This functions main use is to present key statistics for different categories in a 'pretty' manner.

    Visual inspection of the output of this function will assist in discerning what the next steps should be.

    """
    if col_name == "RevolvingUtilizationOfUnsecuredLines":
        print("Training Set: ")
        print()
        print("Max: ", training_x.describe()[col_name]['max'])
        print("Mean", training_x.describe()[col_name]['mean'])
        print()
        print("Missing Values: ", sum(pd.isnull(training_x[col_name])))
        print()
        print("Testing Set: ")
        print()
        print("Max: ", test_x.describe()[col_name]['max'])
        print("Mean: ", test_x.describe()[col_name]['mean'])
        print()
        print("Missing Values: ", sum(pd.isnull(test_x[col_name])))
    elif col_name == 'age':
        print("Training Set: ")
        print()
        print(training_x.describe()[col_name]['max'])
        print(training_x.describe()[col_name]['min'])
        print(training_x.describe()[col_name]['mean'])
        print()
        print("Missing Values: ", sum(pd.isnull(training_x[col_name])))
        print()
        print("Testing Set: ")
        print()
        print(test_x.describe()[col_name]['max'])
        print(test_x.describe()[col_name]['min'])
        print(test_x.describe()[col_name]['mean'])
        print()
        print("Missing Values: ", sum(pd.isnull(test_x[col_name])))
    elif col_name == 'zipcode' or col_name == "Zipcode" or col_name == "Zip code" or col_name == "Zip Code"\
    or col_name == "Zip" or col_name == 'zip':
        print("Training Set: ")
        print()
        print(training_x[col_name].unique()) #looks like our zip codes are good.
        print()
        print("Missing Values: ", sum(pd.isnull(training_x[col_name])))
        print()
        print('Test Set: ')
        print()
        print(test_x[col_name].unique())
        print()
        print("Missing Values: ", sum(pd.isnull(test_x[col_name])))
    elif col_name == "MonthlyIncome":
        print("Training Set: ")
        print()
        print(training_x.describe()[col_name])
        print()
        print("Mean: ", training_x[col_name].mean())
        print()
        print("Median: ", training_x[col_name].median())
        print()
        print("Missing Values: ", sum(pd.isnull(training_x[col_name])))
        print()
        print('Test Set: ')
        print()
        print(test_x.describe()[col_name])
        print()
        print("Mean: ", test_x[col_name].mean())
        print()
        print("Median: ", test_x[col_name].median())
        print()
        print("Missing Values: ", sum(pd.isnull(test_x[col_name])))
    else:
        print("Training Set: ")
        print()
        print(training_x.describe()[col_name])
        print()
        print("Missing Values: ", sum(pd.isnull(training_x[col_name])))
        print()
        print('Test Set: ')
        print()
        print(test_x.describe()[col_name])
        print()
        print("Missing Values: ", sum(pd.isnull(test_x[col_name])))
        print()