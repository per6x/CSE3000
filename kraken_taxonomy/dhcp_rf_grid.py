import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from os.path import join, dirname


### Load metadata
metadata = pd.read_csv(join(dirname(__file__), '..', 'metadata.csv'), sep=";")

### Load features and labels
X = pd.read_csv(join(dirname(__file__), "features.csv"), sep=";")
y = pd.read_csv(join(dirname(__file__), '..', 'labels.csv'), sep=";")
y = y.loc[y["Sample"].isin(X["Sample"])].set_index("Sample", drop=True)["Label"]
X = X.set_index("Sample", drop=True)

### Split the data into train and test
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Grid Search for PCA and Random Forest Classifier
# Define the Pipeline
pipeline = Pipeline([
    ('variance_filter', VarianceThreshold()), # Step 0: filter out variables with variance < threshold
    ('log_transform', FunctionTransformer(np.log1p)),  # Step 1: log transformation
    ('min_max_scaler', MinMaxScaler()),  # Step 2: min-max scaling
    ('pca', PCA()),  # Step 3: PCA
    ('rf', RandomForestClassifier(random_state=42, warm_start=True))  # Step 4: Random Forest
])

#Define the parameter grid for grid search
param_grid = {
    'variance_filter__threshold': [0.0, 0.1, 0.2, 0.3, 0.4], # grid search value for variance filter threshold
    'pca__n_components': [1, 5, 10, 15, 20, 25, 30],  # Step 3:grid search number of components
    'rf__n_estimators': [100, 150, 200, 250, 300, 350],  # Step 4: grid search for n_estimators
    'rf__max_depth': [25, 50, 100, 200, 250, 300, None],  # Step 4: grid search for max_depth
    'rf__criterion' :['gini', 'entropy', 'log_loss'], # Step 4: grid search for criterion
    'rf__max_features': ['sqrt', 'log2'], # Step 4: grid search for number of max_features
}
# Define the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the GridSearchCV object on the training data
grid_search.fit(X_train, y_train)

# Get the best estimator and make predictions on the testing data
best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)

# Compute accuracy for the best found RF
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(grid_search.best_params_)
