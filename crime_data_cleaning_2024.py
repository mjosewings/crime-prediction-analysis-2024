import pandas as pd
import os

def load_data(file_path):
    """Load crime data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(crime_data):
    """Clean and preprocess the crime dataset."""
    # Remove unnecessary columns
    columns_to_drop = ['OBJECTID', 'OCTO_RECORD_ID', 'BID']
    crime_data = crime_data.drop(columns=columns_to_drop)

    # Handle missing values
    critical_columns = ['WARD', 'DISTRICT', 'PSA', 'LATITUDE', 'LONGITUDE', 'START_DATE', 'END_DATE']
    crime_data = crime_data.dropna(subset=critical_columns)

    # Convert date columns to datetime
    date_columns = ['REPORT_DAT', 'START_DATE', 'END_DATE']
    for col in date_columns:
        crime_data[col] = pd.to_datetime(crime_data[col], errors='coerce')

    # Standardize column names
    crime_data.columns = crime_data.columns.str.lower().str.replace(' ', '_')

    # Validate geographic data
    crime_data = crime_data[(crime_data['latitude'].between(24.396308, 49.384358)) &
                            (crime_data['longitude'].between(-125.0, -66.93457))]

    # Remove duplicates
    crime_data = crime_data.drop_duplicates()

    return crime_data

def extract_features(crime_data):
    """Extract additional features from date/time columns."""
    crime_data['report_hour'] = crime_data['report_dat'].dt.hour
    crime_data['report_day'] = crime_data['report_dat'].dt.day
    crime_data['report_month'] = crime_data['report_dat'].dt.month
    crime_data['report_dayofweek'] = crime_data['report_dat'].dt.dayofweek

    # Categorize time of day
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 24:
            return 'evening'
        else:
            return 'night'

    crime_data['report_time_category'] = crime_data['report_hour'].apply(categorize_time)

    return crime_data

def save_cleaned_data(crime_data, output_path='cleaned_crime_data.csv'):
    """Save cleaned dataset to a CSV file."""
    crime_data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# Usage example:
if __name__ == "__main__":
    file_path = 'Crime_Incidents_in_2024.csv'
    crime_data = load_data(file_path)
    crime_data = clean_data(crime_data)
    crime_data = extract_features(crime_data)
    save_cleaned_data(crime_data)
