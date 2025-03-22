import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def preprocess_data(crime_data):
    """Encode categorical features and scale numerical features."""
    categorical_features = ['shift', 'offense', 'method', 'report_dayofweek', 'report_time_category']
    numerical_features = ['latitude', 'longitude', 'report_hour', 'report_day', 'report_month']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(crime_data[categorical_features])

    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(crime_data[numerical_features])

    X = np.hstack([scaled_numerical_features, encoded_features])
    y = crime_data['ward']

    # Convert target to numerical encoding if categorical
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    return X, y

def train_and_evaluate_models(X, y):
    """Train multiple classifiers and evaluate them."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # RandomForest Classifier
    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3)
    rf_grid.fit(X_train, y_train)
    rf_predictions = rf_grid.predict(X_test)

    print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
    print("Random Forest Best Parameters:", rf_grid.best_params_)

    # Gradient Boosting Classifier
    gb_param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
    gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=3)
    gb_grid.fit(X_train, y_train)
    gb_predictions = gb_grid.predict(X_test)

    print("Gradient Boosting Classification Report:\n", classification_report(y_test, gb_predictions))
    print("Gradient Boosting Best Parameters:", gb_grid.best_params_)

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_predictions))

    return rf_grid, gb_grid, lr_model

# Usage example:
if __name__ == "__main__":
    crime_data = pd.read_csv('cleaned_crime_data.csv')
    X, y = preprocess_data(crime_data)
    train_and_evaluate_models(X, y)
