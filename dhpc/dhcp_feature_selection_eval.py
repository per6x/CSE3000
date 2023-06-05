# %% [markdown]
# ## Imports

import pickle
from datetime import datetime

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from mrmr import mrmr_classif
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
                                       SelectKBest, SelectPercentile,
                                       SequentialFeatureSelector, chi2,
                                       f_classif, mutual_info_classif)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

# %% [markdown]
# ## Load data

# %%
# X = pd.read_csv("species_relative_abundance.csv", sep=";")
X = pd.read_csv("/home/ppersianov/CSE3000/kraken_taxonomy/genus_features_all.csv", sep=";")
y = pd.read_csv("/home/ppersianov/CSE3000/labels.csv", sep=";")
y = y.loc[y["Sample"].isin(X["Sample"])].set_index("Sample", drop=True)["Label"]
X = X.set_index("Sample", drop=True)

assert X.shape[0] == y.shape[0]

# %% [markdown]
# ## Feature selection algorithms classes

# %%


class MRMRFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, k=100, **kwargs):
        self.k = k

    def fit(self, X, y):
        self.selected_features = mrmr_classif(X=X, y=y, K=self.k, show_progress=False)
        return self

    def transform(self, X):
        return X[self.selected_features]

    def get_support(self, indices=False):
        return self.selected_features if indices else None


# %% [markdown]
# ## Main class


# %%
class FeatureSelectionRandomizedSearch:
    def __init__(
        self,
        classifiers,
        param_distributions,
        feature_selection_algorithms=None,
        metrics=None,
        test_size=0.2,
        use_smote=False,
        num_features=None,
        n_cv=5,
        n_iter=10,
        scoring=None,
    ):
        self.classifiers = classifiers
        self.param_distributions = param_distributions
        self.feature_selection_algorithms = feature_selection_algorithms
        self.metrics = metrics
        self.results = {}
        self.test_size = test_size
        self.use_smote = use_smote
        self.num_features = num_features
        self.n_cv = n_cv
        self.n_iter = n_iter
        self.scoring = scoring

    def build_pipeline(self, classifier, feature_selection_algorithm=None):
        steps = []
        if feature_selection_algorithm is not None:
            steps.append(("feature_selection", feature_selection_algorithm))
        steps.append(("classification", classifier))
        return Pipeline(steps)

    def evaluate_metrics(self, pipeline, X, y):
        metric_scores = {}
        for metric in self.metrics:
            scores = cross_val_score(pipeline, X, y, cv=self.n_cv, scoring=metric)
            metric_scores[metric] = scores
        return metric_scores

    def fit(self, X, y):
        self.results = {}

        for fs_name, fs_algorithm in (
            fs_pbar := tqdm(self.feature_selection_algorithms.items(), position=0)
        ):
            fs_pbar.set_description(f"Running features selection using {fs_name}")
            fs_results = {}

            if fs_algorithm is not None:
                # fs_algorithm.set_params(**{'k': self.num_features})
                fs_algorithm.fit(X, y)
                X_train, X_test, y_train, y_test = train_test_split(
                    fs_algorithm.transform(X),
                    y,
                    test_size=self.test_size,
                    random_state=42,
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=42
                )

            for clf_name, clf in (
                clf_pbar := tqdm(self.classifiers.items(), position=1, leave=False)
            ):
                clf_pbar.set_description(f"Running classifier {clf_name}")

                # Build the pipeline
                # pipeline = self.build_pipeline(clf, fs_algorithm)
                pipeline = self.build_pipeline(clf)
                random_search = RandomizedSearchCV(
                    pipeline,
                    param_distributions=self.param_distributions[clf_name],
                    n_iter=self.n_iter,
                    cv=self.n_cv,
                    random_state=42,
                    n_jobs=-1,
                    scoring=self.scoring,
                )

                if self.use_smote:
                    # Apply SMOTE on the training data
                    smote = SMOTE()
                    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                    random_search.fit(X_train_res, y_train_res)
                else:
                    random_search.fit(X_train, y_train)

                # Get the best estimator
                best_estimator = random_search.best_estimator_

                # Evaluate the metrics using cross-validation
                metric_scores = self.evaluate_metrics(best_estimator, X, y)

                # Store the results
                fs_results[clf_name] = {
                    "best_score": random_search.best_score_,
                    "best_estimator": best_estimator,
                    "metric_scores": metric_scores,
                    "selected_features": fs_algorithm.get_support(indices=True)
                    if fs_algorithm
                    else X.columns,
                }

            self.results[fs_name] = fs_results

    def get_best_params(self, classifier_name, feature_selection_name):
        return (
            self.results.get(classifier_name, {})
            .get(feature_selection_name, {})
            .get("best_params")
        )

    def get_best_score(self, classifier_name, feature_selection_name):
        return (
            self.results.get(classifier_name, {})
            .get(feature_selection_name, {})
            .get("best_score")
        )

    def get_metric_scores(self, classifier_name, feature_selection_name):
        return (
            self.results.get(classifier_name, {})
            .get(feature_selection_name, {})
            .get("metric_scores")
        )

    def get_selected_features(self, classifier_name, feature_selection_name):
        return (
            self.results.get(classifier_name, {})
            .get(feature_selection_name, {})
            .get("selected_features")
        )

    def plot_results(self):
        df = pd.DataFrame(columns=["Model", "Metric", "Score", "Features"])
        # print(self.results)

        for fs_name, fs_results in self.results.items():
            for clf_name, clf_result in fs_results.items():
                best_estimator = clf_result["best_estimator"]
                metric_scores = clf_result["metric_scores"]
                selected_features = clf_result["selected_features"]

                for metric, scores in metric_scores.items():
                    # scores = np.concatenate(scores)  # Flatten the scores
                    for score in scores:
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    [
                                        {
                                            "Model": clf_name,
                                            "Metric": metric,
                                            "Score": score,
                                            "Features": f"{fs_name} (Top {self.num_features} features)"
                                            if self.feature_selection_algorithms[
                                                fs_name
                                            ]
                                            else fs_name,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )

        # print(df)

        g = sns.catplot(
            data=df,
            x="Model",
            y="Score",
            hue="Metric",
            col="Features",
            estimator=np.mean,
            kind="bar",
            errorbar="se",
            capsize=0.1,
            errwidth=1,
            col_wrap=3,
            legend=False,
            sharex=False,
            sharey=False,
            legend_out=True,
        )
        g.set(ylim=(0, 1))
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(
            f"Feature selection methods evaluation using {self.n_cv}-fold cross-validation"
        )
        g.fig.tight_layout()
        g.tick_params(labelbottom=True)
        for ax in g.axes.ravel():
            ax.legend()

        plt.savefig(f"/home/ppersianov/CSE3000/feature_selection_comp_dhpc.png")

        # g.fig.legend(bbox_to_anchor=(0.5, -0.1), loc='lower center', ncol=3)
        # plt.legend(loc='center right')
        # plt.show()


# %% [markdown]
# ## Running
# define number of features to select
n_features_to_select = 100

# Define the classifiers
classifiers = {
    'RF': RandomForestClassifier(warm_start=True, n_jobs=-1, random_state=42),
    # 'AdaBoost': AdaBoostClassifier(random_state=42),
    # 'XGBoost': XGBClassifier(n_jobs=-1, random_state=42),
    # 'LR': LogisticRegression(warm_start=True, n_jobs=-1, random_state=42),
    # 'SVM': SVC(random_state=42),
}

# Define the feature selection algorithms
feature_selection_algorithms = {
    'Chi-square': SelectKBest(score_func=chi2, k=n_features_to_select),
    # 'ANOVA': SelectKBest(score_func=f_classif, k=n_features_to_select),
    # 'Information Gain': SelectKBest(score_func=mutual_info_classif, k=n_features_to_select),
    # 'XGBoost': SelectFromModel(estimator=XGBClassifier(n_jobs=-1), max_features=n_features_to_select, threshold=-np.inf),
    # 'MRMR': MRMRFeatureSelection(n_jobs=-1, random_state=42),
    # 'LASSO': SelectFromModel(estimator=Lasso(alpha=1), max_features=n_features_to_select, threshold=-np.inf),
    # 'All Features': None,
}

# Define the parameter distributions for each classifier
param_distributions = {
    "RF": {
        "classification__n_estimators": randint(1, 250),
        "classification__criterion": ["gini", "entropy"],
        "classification__max_depth": [None] + list(range(1, 20)),
        "classification__min_samples_split": randint(2, 20),
        "classification__max_features": ["sqrt", "log2"],
        "classification__min_samples_leaf": randint(1, 10),
        "classification__bootstrap": [True, False],
    },
    "AdaBoost": {
        "classification__n_estimators": randint(1, 250),
        "classification__learning_rate": uniform(0.01, 1.0),
        "classification__estimator": [RandomForestClassifier()],
        "classification__algorithm": ["SAMME", "SAMME.R"],
        "classification__random_state": [None, 42],
    },
    "XGBoost": {
        "classification__n_estimators": randint(1, 250),
        "classification__learning_rate": uniform(0.01, 1.0),
        "classification__max_depth": randint(1, 10),
        "classification__subsample": uniform(0.6, 0.4),
        "classification__colsample_bytree": uniform(0.6, 0.4),
        "classification__reg_alpha": uniform(0, 1),
        "classification__reg_lambda": uniform(0, 1),
    },
    "LR": { # Logistic regression
        "classification__C": uniform(0.1, 1.0),
        "classification__penalty": ["elasticnet", "l1", "l2"],
        "classification__class_weight": [None, "balanced"],
        "classification__max_iter": [100000],
    },
    "SVM": {
        "classification__C": uniform(0.1, 1.0),
        "classification__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "classification__degree": randint(1, 3),
        "classification__gamma": ["scale", "auto"],
        "classification__class_weight": [None, "balanced"],
        "classification__probability": [True, False],
    },
}

# Instantiate the FeatureSelectionRandomizedSearch class
feature_selection_search = FeatureSelectionRandomizedSearch(
    classifiers,
    param_distributions,
    feature_selection_algorithms,
    metrics=["accuracy", "f1", "roc_auc"],
    test_size=0.2,
    use_smote=False,
    use_log_scale=False,
    num_features=n_features_to_select,
    n_cv=2,
    n_iter=2,
    scoring='f1',
)

# Fit the search using your data
feature_selection_search.fit(X, y)

# Plot the results
feature_selection_search.plot_results(plot_output_path='fs_100_species.png')

# Define the parameter distributions for each classifier
param_distributions = {
    "RF": {
        "classification__n_estimators": randint(1, 250),
        "classification__criterion": ["gini", "entropy"],
        "classification__max_depth": [None] + list(range(1, 20)),
        "classification__min_samples_split": randint(2, 20),
        "classification__max_features": ["sqrt", "log2"],
        "classification__min_samples_leaf": randint(1, 10),
        "classification__bootstrap": [True, False],
    },
    "AdaBoost": {
        "classification__n_estimators": randint(1, 250),
        "classification__learning_rate": uniform(0.01, 1.0),
        "classification__estimator": [RandomForestClassifier()],
        "classification__algorithm": ["SAMME", "SAMME.R"],
        "classification__random_state": [None, 42],
    },
    "XGBoost": {
        "classification__n_estimators": randint(1, 250),
        "classification__learning_rate": uniform(0.01, 1.0),
        "classification__max_depth": randint(1, 10),
        "classification__subsample": uniform(0.6, 0.4),
        "classification__colsample_bytree": uniform(0.6, 0.4),
        "classification__reg_alpha": uniform(0, 1),
        "classification__reg_lambda": uniform(0, 1),
    },
    "LR": { # Logistic regression
        "classification__C": uniform(0.1, 1.0),
        "classification__penalty": ["elasticnet", "l1", "l2"],
        "classification__class_weight": [None, "balanced"],
        "classification__max_iter": [100000],
    },
    "SVM": {
        "classification__C": uniform(0.1, 1.0),
        "classification__kernel": ["linear", "poly", "rbf", "sigmoid"],
        "classification__degree": randint(1, 3),
        "classification__gamma": ["scale", "auto"],
        "classification__class_weight": [None, "balanced"],
        "classification__probability": [True, False],
    },
}

# Instantiate the FeatureSelectionRandomizedSearch class
feature_selection_search = FeatureSelectionRandomizedSearch(
    classifiers,
    param_distributions,
    feature_selection_algorithms,
    metrics=["accuracy", "f1", "roc_auc"],
    test_size=0.2,
    use_smote=True,
    num_features=n_features_to_select,  # 100
    n_cv=10,
    n_iter=60,
    scoring="f1",
)

# Fit the search using your data
feature_selection_search.fit(X, y)

# Plot the results
feature_selection_search.plot_results()


with open(f'feature_selection_search_results_genus_{datetime.now().strftime("%d.%m.%Y_%H.%M.%S")}.p', 'wb') as fp:
    pickle.dump(feature_selection_search.results, fp, protocol=pickle.HIGHEST_PROTOCOL)
