#!/usr/bin/env python3

"""
This script provides a general framework for training a classification model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from DataScienceShareable import ClassifierPGs

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)

if __name__ == '__main__':
    """
    Collect the Data
    """
    file_path = 'C:\\example.csv'
    all_data = pd.read_csv(file_path, encoding='ISO-8859-1')

    # SETUP KEY VARIABLES AND DATASETS
    y_Column = ''

    """
    Clean the Data
    """
    xdata = ''  # perform necessary data cleaning here
    ydata = all_data[y_Column]

    """
    Initial Data Exploration
    """
    # FOR CORRELATION MATRIX STANDARDIZE THE VARIABLES FIRST
    xydata = pd.merge(xdata, ydata, how='inner', left_index=True, right_index=True)
    scaler = preprocessing.StandardScaler()
    xydata_scaled = pd.DataFrame(scaler.fit_transform(xydata), index=xydata.index, columns=xydata.columns)
    correlation_matrix = abs(xydata_scaled.corr())
    sns.heatmap(correlation_matrix)
    sortedcorr = correlation_matrix.iloc[:, -1].sort_values(ascending=False)
    # IF DESIRED, NARROW TO ONLY FEATURES THAT CORRELATE HIGHLY WITH THE DEPENDENT VARIABLE
    narrowfeatures = sortedcorr[sortedcorr > .3].index.drop(y_Column)
    xdata = xdata[narrowfeatures]

    # PLOT FEATURES AGAINST THE DEPENDENT VARIABLE TO ASSESS FOR NON-LINEAR RELATIONSHIPS
    sns.lmplot(x=xdata, y=ydata.name, data=all_data, order=1, robust=True)
    r2 = "r2: " + str(metrics.r2_score(ydata, xdata))
    correlation = 'Cor: ' + str(np.corrcoef(xdata, ydata)[0][1])
    plt.text(min(xdata), max(ydata), r2 + '\n' + correlation)

    """    
    Setup Pipeline(s)
    """
    # SETUP CROSS-VALIDATION STRATEGY
    ss_generator = model_selection.ShuffleSplit(n_splits=5, test_size=.3, random_state=1)

    # SETUP ESTIMATORS FOR THE PIPELINE
    # fse = elastic  # Feature Selector Estimator used in pipeline.
    clf = ClassifierPGs.xgboost  # Classifier used in pipeline
    classifier_w_grid_search = model_selection.GridSearchCV(clf, param_grid=ClassifierPGs.xgboost_param_grid,
                                                            cv=ss_generator,
                                                            iid=False, scoring="roc_auc", n_jobs=-1)

    loops = 1  # How many loops to run, if varying a parameter
    rmse_coefs = pd.Series([None for _ in range(loops)])
    for loop in range(loops):
        """
        Split the data into testing set and training set
        """
        x_train, x_test, y_train, y_test = model_selection.train_test_split(xdata, ydata,
                                                                            test_size=.30, random_state=240,
                                                                            shuffle=True, stratify=ydata)

        """
        Run Pipeline
        """
        steps = [
            # ('polynomial', preprocessing.PolynomialFeatures(degree=2)),
            ('scaler', preprocessing.StandardScaler()),
            # ('featureselector', feature_selection.RFE(fse, n_features_to_select=10)),
            ('classifier', classifier_w_grid_search)]
        pipeln = Pipeline(steps)
        pipeln.fit(x_train, y_train)
        predictions = pipeln.predict(x_test)
        scor = pipeln.score(x_test, y_test)
        cm = confusion_matrix(y_test, predictions)

        print(clf)
        print("Best Cross-validated Score: ", pipeln["classifier"].best_score_)
        print("Test Score: ", scor)
        print(cm)
        print(pipeln["classifier"].best_params_)
        # print(pipeln["classifier"].best_estimator_.feature_importances_)

        # record rmse scores for each iteration, if desired
        rmse_coefs[loop] = scor

    # pprint(rmse_coefs)
    # pprint(rmse_coefs.sort_values(ascending=True))
    # plt.plot(rmse_coefs.iloc)
