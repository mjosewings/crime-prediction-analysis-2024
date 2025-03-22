import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

try:
    df = pd.read_csv("cleaned_crime_data.csv")

    # Convert date/time columns to datetime objects.
    date_columns = ['report_dat', 'start_date', 'end_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Extract relevant features from date/time columns.
    for col in date_columns:
        if col in df.columns:
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
            df[col + '_hour'] = df[col].dt.hour
            df[col + '_dayofweek'] = df[col].dt.dayofweek

    # Drop the original date/time columns.
    df = df.drop(date_columns, axis=1, errors='ignore')

    # One-hot encode the categorical columns.
    categorical_cols = ['shift', 'method', 'anc', 'neighborhood_cluster', 'voting_precinct','report_dayofweek','report_time_category'] #added columns.
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)

    #drop block and block_group.
    df = df.drop(['block','block_group'], axis = 1, errors = 'ignore')

    # Debugging: Print data types and unique values.
    print(df.dtypes)
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"Unique values in {col}: {df[col].unique()}")

    target_column_name = 'offense'
    y = df[target_column_name]
    X = df.drop(target_column_name, axis=1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model.
    joblib.dump(model, "random_forest.pkl")
    print("Model saved as random_forest.pkl")

    #save test data.
    np.savez("test_data.npz", X_test = X_test, y_test = y_test)
    print("test data saved as test_data.npz")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except KeyError as e:
    print(f"Error: Key not found in DataFrame - {e}")
    print(f"DataFrame columns: {df.columns}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")