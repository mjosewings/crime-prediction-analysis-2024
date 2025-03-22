import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_visualizations(crime_data, plot_dir='crime_prediction_plots'):
    """Generate and save various crime data visualizations."""
    os.makedirs(plot_dir, exist_ok=True)

    # Crime Distribution by Day of the Week
    plt.figure(figsize=(12, 6))
    sns.countplot(x=crime_data['report_dayofweek'])
    plt.title('Crime Distribution by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Crimes')
    plt.savefig(os.path.join(plot_dir, 'crime_distribution_dayofweek.svg'))
    plt.show()

    # Crime Type vs. Time of Day (Heatmap)
    crime_time_heatmap = crime_data.groupby(['offense', 'report_hour']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(crime_time_heatmap, cmap='viridis')
    plt.title('Crime Type vs. Time of Day')
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('Offense')
    plt.savefig(os.path.join(plot_dir, 'crime_type_vs_timeofday.svg'))
    plt.show()

    # Crime Incidents by Ward
    plt.figure(figsize=(12, 6))
    sns.countplot(x=crime_data['ward'])
    plt.title('Crime Incidents by Ward')
    plt.xlabel('Ward')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(plot_dir, 'crime_incidents_by_ward.svg'))
    plt.show()

# Usage example:
if __name__ == "__main__":
    crime_data = pd.read_csv('cleaned_crime_data.csv')
    create_visualizations(crime_data)
