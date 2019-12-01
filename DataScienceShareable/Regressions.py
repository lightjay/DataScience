#!/usr/bin/env python3

"""
This script provides a template for training regression models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection, metrics, preprocessing
from sklearn.pipeline import Pipeline

from DataScienceShareable import RegressorPGs

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)


def rmse(test, pred):
    return np.sqrt(metrics.mean_squared_error(test, pred))


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
    # SELECT CROSS-VALIDATION STRATEGY
    ss_generator = model_selection.ShuffleSplit(n_splits=5, test_size=.3, random_state=1)

    # SELECT THE ESTIMATORS FOR EACH RELEVANT PIPELINE STEP
    fse = RegressorPGs.dtr  # Feature Selector Estimator used in pipeline. Ridge
    est = RegressorPGs.dtr  # Final Estimator used in pipeline. Ridge
    regressor_w_grid_search = model_selection.GridSearchCV(est, param_grid=RegressorPGs.dtr_pg, cv=ss_generator,
                                                           iid=False, scoring="neg_mean_squared_error", n_jobs=-1)

    states = 1  # if varying a splitting parameter, set num iterations in outer loop
    coefs = 1  # if varying a pipeline parameter, set num iterations in inner loop
    rmse_coefs = pd.DataFrame([[None for _ in range(states)] for _ in range(coefs)]).iloc[1:, :]
    for state in range(states):
        """
        Split the data into testing set and training set
        """
        x_train, x_test, y_train, y_test = model_selection.train_test_split(xdata, ydata,
                                                                            test_size=.30, random_state=433,
                                                                            shuffle=True, stratify=None)

        for coefmax in range(coefs):
            steps = [
                ('polynomial', preprocessing.PolynomialFeatures(degree=1)),
                ('scaler', preprocessing.StandardScaler()),
                # ('featureselector', feature_selection.RFE(fse, n_features_to_select=coefmax + 3)),
                ('estimator', regressor_w_grid_search)]
            pipeln = Pipeline(steps)
            pipeln.fit(x_train, y_train)
            predictions = pipeln.predict(x_test)
            predictions = pd.DataFrame(predictions, index=x_test.index, columns=['Predictions'])

            """
            Print Summary of final regression
            """
            # FIND THE SCALING COEFFICIENTS FROM 2ND & 3RD STEPS OF PIPELINE
            selected_features_bool = np.array(pipeln[-2].support_)
            poly_features = np.array(pipeln[0].get_feature_names(x_train.columns))
            selected_features = poly_features[selected_features_bool]
            xdata_avg_var = pd.DataFrame([pipeln[1].mean_, pipeln[1].scale_],
                                         index=['mean', 'st. dev.'], columns=poly_features)  # scaling coeficients

            # FIND THE PREDICTION COEFFICIENTS, RSQUARED AND OTHER VALUES FOR SUMMARY PRINTOUT
            coef_df = pd.DataFrame(pipeln[-1].best_estimator_.coef_, index=selected_features, columns=['Coefficients'])
            rsquared = pipeln.score(x_test, y_test)
            comp = pd.merge(y_test, predictions, how='inner', left_index=True, right_index=True)
            comp['Residuals'] = comp.Predictions - comp[y_test.name]

            # PRINT SUMMARY AND GRAPHS
            sns.lmplot('Predictions', y_Column, data=comp)  # plot predictions
            sns.lmplot('Residuals', y_Column, data=comp)  # plot residuals

            # RECORD RMSE SCORES FOR EACH ITERATION, IF DESIRED
            rmse_coefs.loc[coefmax + 1, state] = rmse(y_test, predictions)

    # pprint(rmse_coefs)
    # pprint(rmse_coefs.T.sort_values(by=1))
    # plt.plot(rmse_coefs.iloc)
