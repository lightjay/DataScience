#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a fairly comprehensive library of regression models. The intent is to call them along with
their associate parameter grids during a training using grid search for tuning hyperparamenters. This file is likely to
be a continual work in progress.
"""

__author__ = "Josh Lloyd"
__author_email__ = "joshslloyd@outlook.com"

# std python library

# 3rd party
import numpy as np
import xgboost as xgb
from sklearn import linear_model, svm, ensemble, \
    gaussian_process, neighbors, neural_network, tree, cross_decomposition, isotonic, kernel_ridge

# local

"""
classifiers (27 in total) and Parameter Grids
"""
# PLS Regression
plsr = cross_decomposition.PLSRegression(scale=False)
plsr_pg = {
    "n_components": [1, 2]
}

# Ensemble models
rfr = ensemble.RandomForestRegressor(n_jobs=-1, random_state=0, criterion='mse')
rfr_pg = {
    "max_depth": [i for i in range(1, 101, 20)],
    "min_samples_split": [i for i in range(2, 11, 2)],
    "bootstrap": [True, False],
    "n_estimators": [i for i in range(1, 101, 20)],
}
abr = ensemble.AdaBoostRegressor()
br = ensemble.BaggingRegressor()
etsc = ensemble.ExtraTreesRegressor()
gbc = ensemble.GradientBoostingRegressor()
# vr = ensemble.VotingRegressor()
# hgbc = ensemble.HistGradientBoostingClassifier()

# Gaussian Process
gpr = gaussian_process.GaussianProcessRegressor(random_state=0)
gpr_pg = {"alpha": np.logspace(-2, 0, 5), }

# Isotonic Regression
ir = isotonic.IsotonicRegression()
ir_pg = {}

# Kernal Ridge Regressions
krr = kernel_ridge.KernelRidge()
krr_pg = {
    "alpha": [.00001, .0001, .001, .01, .1, .2, .5, .8],
    "gamma": np.append(0, np.logspace(-1, 5, 4)),
    "degree": [1, 2, 3],
    "coef0": [1, 2]
}

# linear models
ardr = linear_model.ARDRegression()
bayesr = linear_model.BayesianRidge()

en = linear_model.ElasticNet(normalize=False, random_state=0, tol=.6, fit_intercept=True)
en_pg = {
    "alpha": [.00001, .0001, .001, .01, .1, .2, .5, .8],
    "l1_ratio": [.1, .25, .5, .7, .9, .95, .99, 1],
    "selection": ['cyclic', 'random'],
}

hr = linear_model.HuberRegressor()
lars = linear_model.Lars()
lasso = linear_model.Lasso()
ll = linear_model.LassoLars()
llic = linear_model.LassoLarsIC()
olsr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
olsr_pg = {}
ompr = linear_model.OrthogonalMatchingPursuit()
par = linear_model.PassiveAggressiveRegressor()
rr = linear_model.RANSACRegressor()

ridge = linear_model.Ridge(normalize=False, fit_intercept=True, random_state=0, solver='auto')
ridge_pg = {"alpha": [.00001, .0001, .001, .01, .1, .2, .5, .8]}

sgdr = linear_model.SGDRegressor()
tsr = linear_model.TheilSenRegressor()
# rc = linear_model.ridge_regression()

# neighbors
knnr = neighbors.KNeighborsRegressor(n_jobs=-1)
knnr_pg = {
    "n_neighbors": [5, 10, 15],
    "weights": ["uniform", "distance"],
    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
    "leaf_size": [15, 30, 60, 90],
    "p": [1, 2, 3],
}
rnr = neighbors.RadiusNeighborsRegressor()

# neural networks
mlpr = neural_network.MLPRegressor(random_state=0, tol=.6)
mlpr_pg = {
    "hidden_layer_sizes": [(6, 1), (6, 3, 1), (6, 6, 1), (6, 5, 1), (6, 4, 1), (6, 2, 1)],
    "activation": ['identity', 'logistic', 'tanh', 'relu'],
    "solver": ['lbfgs', 'sgd', 'adam'],
    "alpha": [.00001, .0001, .001, .01, .1, .2, .5, .8, .9, .99],
    "learning_rate": ['constant', 'invscaling', 'adaptive'],
    "learning_rate_init": [.001, .0001],
    # "momentum": [.2, .5, .8, .9, .99],
    # "beta_1": [.2, .5, .8, .9, .99],
    # "beta_2": [.5, .8, .9, .999, .9999],
    # "epsilon ": [.000000001, .00000001, .00001, .0001, .001, .01, .1, .2, .5, .8, .9],
}

# Support Vector Machines
svr = svm.SVR()
svr_pg = {
    "kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    "degree": [1, 2, 3, 4],
    "gamma": np.append(0, np.logspace(-1, 5, 4)),
    "C": [i for i in range(1, 10, 2)],
    "epsilon": [.000000001, .00000001, .00001, .0001, .001, .01, .1, .2, .5],
}
lin_svr = svm.LinearSVR()
nu_svr = svm.NuSVR()

# Decision Trees
dtr = tree.DecisionTreeRegressor(random_state=0)
dtr_pg = {"criterion": ['mse', 'friedman_mse', 'mae'], }
etr = tree.ExtraTreeRegressor()

# XGBoost. for parameter description: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
xgbr = xgb.XGBRegressor(seed=0, objective="reg:squarederror", eval_metric="rmse")
xgbr_pg = {
    "max_depth": [i for i in range(1, 101, 20)],
    "learning_rate": np.logspace(-1, 0, 5),
    # "n_estimators": [i for i in range(1, 101, 20)],
    "booster": ["gbtree", "gblinear", "dart"],
    # gbtree and dart use tree based models while gblinear uses linear functions.
    # "gamma": np.append(0, np.logspace(-1, 5, 4)),  # float
    # "min_child_weight": [i for i in range(5)],  # int
    # "max_delta_step": [i for i in range(0, 10, 2)],  # int
    # "subsample": [.5, .75, .85],  # float
    # "colsample_bytree": [.5, 1],  # float
    # "colsample_bylevel": [.5, 1],  # float
    # "colsample_bynode": [.5, 1],  # float
    # "reg_alpha": np.logspace(-2, 0, 10),  # float
    # "reg_lambda": 1 - np.logspace(-2, 0, 9),  # float
    # "num_parallel_tree": [1, 2, 3],  # int. Used for boosting random forest
    # "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"]
}
xgbrfr = xgb.XGBRFRegressor()
