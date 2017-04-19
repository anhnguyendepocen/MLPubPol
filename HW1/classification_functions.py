#Ibrahim Gabr
import pylab as pl
pl.rcParams['figure.figsize'] = (20, 6)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from general_functions import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def classification_models(training_x, test_x, training_y, test_y, dictionary):
    '''
    Takes a training and test set in addition to a dictionary mapping column indices
    to feature indicies from the original data frame.

    Outputs:
        Evaluates several models and returns the highest scoring model
    '''
    print("Running Logistic Regression Model")
    model = LogisticRegression()
    rfe = RFE(model) #defaults to half of all features.
    fit_log = rfe.fit(training_x, training_y)
    predicted_log = rfe.predict(test_x)
    expected_log = test_y
    accuracy_log = accuracy_score(expected_log, predicted_log)
    tup_log = ("Logisitic Regression", accuracy_log)
    print()
    
    print("Running Extra Tree's Classifier")
    etc = ExtraTreesClassifier()
    etc.fit(training_x, training_y)
    expected_etc = test_y
    predicted_etc = etc.predict(test_x)
    accuracy_etc = accuracy_score(expected_etc, predicted_etc)
    tup_etc = ("Extra Tree's Classifier", accuracy_etc)
    print()
    
    print("Running Random Forest Classifier")
    rfc = RandomForestClassifier()
    rfc.fit(training_x, training_y)
    expected_rfc = test_y
    predicted_rfc = rfc.predict(test_x)
    accuracy_rfc = accuracy_score(expected_rfc, predicted_rfc)
    tup_rfc = ("Random Forest Classifier", accuracy_rfc)
    print()
    
    print("Running Gradient Boosting Classifier")
    gbc = GradientBoostingClassifier()
    gbc.fit(training_x, training_y)
    expected_gbc = test_y
    predicted_gbc = gbc.predict(test_x)
    accuracy_gbc = accuracy_score(expected_gbc, predicted_gbc)
    tup_gbc = ("Gradient Boosting Classifier", accuracy_gbc)
    print()

    print("--**Evaluating Best Model**--")
    print()
    
    best = sorted([tup_log, tup_etc, tup_rfc, tup_gbc],key=lambda x: x[1], reverse=True)

    print("The Best model is: ", best[0][0], "with an accuracy score of: ", best[0][1])

    if best[0][0] == "Logisitic Regression":
        return rfe, predicted_log, expected_log, test_x, test_y, dictionary
    if best[0][0] == "Extra Tree's Classifier":
        return etc, predicted_etc, expected_etc, test_x, test_y, dictionary
    if best [0][0] == "Random Forest Classifier":
        return rfc, predicted_rfc, expected_rfc, test_x, test_y, dictionary
    if best [0][0] == "Gradient Boosting Classifier":
        return gbc, predicted_gbc, expected_gbc, test_x, test_y, dictionary
    
    print()
    
def extract_best_model(training_x, test_x, training_y, test_y, dictionary):
    """
    Takes a training and test set in addition to a dictionary mapping column indicies to column names.

    Return:
        extracts additional information about best performing model.
            Confusion matrix, classification report and ROC curve.
    """
    obj, pred_obj, exp_obj, test_x, test_y, dictionary = classification_models(training_x, test_x, training_y, test_y, dictionary)
    best_features = obj.get_support(indices=True)
    best_feature_name = []
    for i in best_features:
        best_feature_name.append(dictionary[i])
    print("Best Feature Columns:")
    print()
    for i in best_feature_name:
        print(i)
        print()
    
    print("Classification Report: ")
    
    print()
    
    print(classification_report(exp_obj, pred_obj))
    
    print()
    
    print("Confusion Matricies: ")
    
    print()
    
    cnf_matrix = confusion_matrix(exp_obj, pred_obj)
    
    visualize_confusion_matrix(cnf_matrix, labels=["Non-Deliquent", "Deliquent"],
                  title='Regular Confusion matrix')
    print()
    visualize_confusion_matrix(cnf_matrix, labels=["Non-Deliquent", "Deliquent"], normalize=True,
                  title='Normalized confusion matrix')
    print()

    print("ROC Curve: ")
    print()
    preds = obj.predict_proba(test_x)
    fpr, tpr, thresholds = roc_curve(test_y, preds[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic example')
    pl.legend(loc="lower right")
    pl.show()
