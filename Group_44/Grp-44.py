'''
MLBA ASSIGNMENT 2
Group 44
'''

import numpy as np
import pandas as pd
import re
from collections import Counter
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from Pfeature.pfeature import *


amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
               'R', 'S', 'T', 'V', 'W', 'Y']

# Taking input for the paths of training and testing csv files
train_data_path = input("Enter train data csv file path:")
test_data_path = input("Enter test data csv file path:")

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# extract the class labels (0,1) corresponding to the amino acid residue sequences from the training data
train_data_labels = train_data.iloc[:, 0]

def feature_engineering(dataset):
    ''' The function defined as composition finds the composition of a peptide sequences by measuring 
    its frequency of the dipeptides of order one and subsequently dividing it by the length of that sequence.
    Input: A dataframe containing peptide sequences.
    Returns: A dataframe that contained the new features, or dipeptide composition that corresponds to our input data.
    '''
    new_data = pd.DataFrame(list())
    new_data.to_csv('a.csv')
    aac_wp(dataset, 'a.csv')
    data = pd.read_csv('a.csv')
    data.drop(0, inplace=True)
    return data


def machine_learning_model():
    ''' Uses Stacking Classifier to stack the Random Forest Classifier with the Logistic Regression Classifier.
    Logistic Regression Classifier is used to combine the Random Forest base estimator.
    Return: the model variable.
    '''
    model0 = list()  # base estimators
    model0.append(('lr', LogisticRegression()))
    model0.append(('rf', RandomForestClassifier(n_estimators=700,
                                                oob_score="True", n_jobs=-1, max_features="sqrt")))
    # classifier which will be used to combine the base estimators.
    model1 = LogisticRegression()
    # default 5-fold cross validation
    model = StackingClassifier(estimators=model0, final_estimator=model1, cv=5)
    return model


def evaluate_and_fit_model(train_features, train_labels):
    ''' Gets the machine learning model and preforms kfold cross validation on the model with accuracy as scoring criteria. 
    Displays the cross validation results. Then fits the model on the entire training data so that it can be used to make 
    predictions on test dataset.
    Input: the training amino acid composition features and training labels as separate dataframes.
    Return: the final refitted model
    '''
    model = machine_learning_model()  # gets the machine learning model variable
    # stratified 10-Fold cross validation repeated 5 times with different randomization in each repetition.
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
    scores = cross_val_score(model, train_features, train_labels,
                             scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print("\nAccuracy scores for the model from k-fold cross validation: \n", scores)
    print('\nMean accuracy score: %.3f' % (mean(scores)))
    model.fit(train_features, train_labels)
    return model


def main():
    global train_data_labels

    # Amino acid composition of the sequences in training dataset (feature generation)
    train_data_comp = feature_engineering(train_data_path)

    # Amino acid composition of the sequences in testing dataset (feature generation)
    test_data_comp = feature_engineering(test_data_path)

    model = evaluate_and_fit_model(train_data_comp, train_data_labels)
    predictions = model.predict(test_data_comp)

    # create the final predictions dataframe and export it as a csv.
    finalpredictions = pd.DataFrame(
        {'ID': np.array(test_data.iloc[:, 0]), 'Label': predictions[:]})
    finalpredictions.to_csv('Group_44_Predictions_Output.csv', index=False)


if __name__ == "__main__":
    main()