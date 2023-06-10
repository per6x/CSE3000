# %% [markdown]
# # Models comparison without features selection and wuth hyperparameters optimization

# %% [markdown]
# ## Imports

import pickle
from datetime import datetime

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score)
# %%
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, cross_validate,
                                     train_test_split)
from sklearn.svm import SVC
from tqdm.notebook import tqdm
from xgboost import XGBClassifier

# %% [markdown]
# ## Load data

# %%
# Load genus relative abundance data as features
# X = pd.read_csv("../genus_relative_abundance.csv", sep=";")
# Load species relative abundance data as features
X = pd.read_csv("../species_relative_abundance.csv", sep=";")
# Load labels 
y = pd.read_csv("../../labels.csv", sep=";")
y = y.loc[y["Sample"].isin(X["Sample"])].set_index("Sample", drop=True)["Class"]
X = X.set_index("Sample", drop=True)
print(X.shape)
assert X.shape[0] == y.shape[0]

# %% [markdown]
# ## Split the data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Define K-Fold, models and param distributions

# %%
# K-Fold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# models
models = {
  'CatBoost': cb.CatBoostClassifier(thread_count=-1, verbose=False, random_state=42),
  'XGBoost': XGBClassifier(n_jobs=-1, random_state=42),
  'RF': RandomForestClassifier(warm_start=True, n_jobs=-1, random_state=42),
  'AdaBoost': AdaBoostClassifier(random_state=42),
  'LR': LogisticRegression(warm_start=True, n_jobs=-1, random_state=42),
  'SVM': SVC(random_state=42),
}
# params distributions
param_distributions = {
  'RF': {
    'n_estimators': randint(1, 250),
    'criterion': ['gini', 'entropy'],
    'max_depth': [None] + list(range(1, 20)),
    'min_samples_split': randint(2, 20),
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False],
  },
  'AdaBoost': {
    'n_estimators': randint(1, 250),
    'learning_rate': uniform(0.01, 1.0),
    'estimator': [RandomForestClassifier()],
    'algorithm': ['SAMME', 'SAMME.R'],
    'random_state': [None, 42],
  },
  'XGBoost': {
    'n_estimators': randint(1, 250),
    'learning_rate': uniform(0.01, 1.0),
    'max_depth': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
  },
  'LR': { # Logistic regression
    'C': uniform(0.1, 1.0),
    'penalty': ['elasticnet', 'l1', 'l2'],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000],
    'solver': ['sag', 'saga', 'liblinear']
  },
  'SVM': {
    'C': uniform(0.1, 1.0),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': randint(1, 3),
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced'],
    'probability': [True, False],
  },
  'CatBoost': {
    'learning_rate' : uniform(0.001, 1),
    'depth' : randint(2, 10),
    'l2_leaf_reg' : randint(2, 10),
    'random_strength' : uniform(0, 10),
  }
}

# %% [markdown]
# ## Randomized search with cross validation

# %%
# define metrics
# metrics = ['accuracy', 'f1', 'roc_auc']
# dict for cv results
import time

random_search_results = {}

for m_name, model in (p_bar := tqdm(models.items())):
    p_bar.set_description(f'Running RandomizedSearchCV for {m_name}')
    random_search = RandomizedSearchCV(
      model,
      param_distributions=param_distributions[m_name],
      n_iter=60,
      cv=skf,
      random_state=42,
      n_jobs=-1,
      scoring="roc_auc",
    )
    random_search.fit(X_train, y_train)
    random_search_results[m_name] = random_search

# %%
# roc_auc_score(y_test, random_search.best_estimator_.predict(X_test))
with open(f'randomized_search_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.p', 'wb') as fp:
    pickle.dump(random_search_results, fp, protocol=pickle.HIGHEST_PROTOCOL)
