#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides a fairly comprehensive library of classification models. The intent is to call them along with
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
    gaussian_process, naive_bayes, neighbors, neural_network, tree

# local

"""
classifiers (27 in total) and Parameter Grids
"""
# linear models
lr = linear_model.LogisticRegression(fit_intercept=True, random_state=0, multi_class='auto',
                                     n_jobs=-1, max_iter=10000)
lr_param_grid1 = {"penalty": ['l2'],
                  "C": np.logspace(-1, 5, 10),
                  "class_weight": ['balanced', None],
                  "solver": ['newton-cg', 'lbfgs', 'sag', 'saga']}
lr_param_grid2 = {"penalty": ['none'],
                  "class_weight": ['balanced', None],
                  "solver": ['newton-cg', 'lbfgs', 'sag', 'saga']}
lr_param_grid3 = {"penalty": ['l2'],
                  "dual": [True, False],
                  "C": np.logspace(-1, 5, 10),
                  "class_weight": ['balanced', None],
                  "solver": ['liblinear']}
lr_param_grid4 = {"penalty": ['l1'],
                  "C": np.logspace(-1, 5, 10),
                  "class_weight": ['balanced', None],
                  "solver": ['liblinear', 'saga'], }
lr_param_grid5 = {"penalty": ['elasticnet'],
                  "C": np.logspace(-1, 5, 10),
                  "class_weight": ['balanced', None],
                  "solver": ['saga'],
                  "l1_ratio": np.logspace(-2, 0, 10)}

pac = linear_model.PassiveAggressiveClassifier()
rc = linear_model.RidgeClassifier()
sgdc = linear_model.SGDClassifier()

# neighbors
knn = neighbors.KNeighborsClassifier()
rnc = neighbors.RadiusNeighborsClassifier()
ncent = neighbors.NearestCentroid()

# Support Vector Machines
svc = svm.SVC()
lin_svc = svm.LinearSVC()
nu_svc = svm.NuSVC()

# Decision Trees
dtc = tree.DecisionTreeClassifier()
etc = tree.ExtraTreeClassifier()

# Naive Bayes
gnb = naive_bayes.GaussianNB()
mnb = naive_bayes.MultinomialNB()
cnb = naive_bayes.ComplementNB()
bnb = naive_bayes.BernoulliNB()

# neural networks
mlp = neural_network.MLPClassifier()
brbm = neural_network.BernoulliRBM

# XGBoost. for parameter description: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst
xgboost = xgb.XGBClassifier(n_jobs=1, random_state=0, objective="binary:logistic", eval_metric="auc")
xgboost_param_grid = {"max_depth": [i for i in range(1, 101, 20)],
                      "learning_rate": [i for i in range(0, 10, 2)],
                      "n_estimators": [i for i in range(1, 101, 20)],
                      "booster": ["gbtree", "gblinear", "dart"],
                      # gbtree and dart use tree based models while gblinear uses linear functions.
                      "gamma": np.append(0, np.logspace(-1, 5, 4)),  # float
                      # "min_child_weight": [i for i in range(5)],  # int
                      # "max_delta_step": [i for i in range(0, 10, 2)],  # int
                      # "subsample": [.5, .75, .85],  # float
                      # "colsample_bytree": [.5, 1],  # float
                      # "colsample_bylevel": [.5, 1],  # float
                      # "colsample_bynode": [.5, 1],  # float
                      # "reg_alpha": np.logspace(-2, 0, 10),  # float
                      # "reg_lambda": 1 - np.logspace(-2, 0, 9),  # float
                      # "num_parallel_tree": [1, 2, 3],  # int. Used for boosting random forest
                      "importance_type": ["gain", "weight", "cover", "total_gain", "total_cover"]
                      }

# Ensemble models
rfc = ensemble.RandomForestClassifier()
rfc_param_dist = {"max_features": [i for i in range(1, 11, 2)],
                  "min_samples_split": [i for i in range(2, 11, 2)],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": [i for i in range(1, 101, 20)]}
adaboost = ensemble.AdaBoostClassifier()
bc = ensemble.BaggingClassifier()
etsc = ensemble.ExtraTreesClassifier()
gbc = ensemble.GradientBoostingClassifier()
# vc = ensemble.VotingClassifier()
# hgbc = ensemble.HistGradientBoostingClassifier()

# Gaussian Process
gpc = gaussian_process.GaussianProcessClassifier()
