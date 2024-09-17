import pandas as pd
import matplotlib.pyplot as plt

def load_and_explore_data(data_path):
    data_path = '/Users/syedmohathashimali/Pratice/Steel_industry_data.csv'
    # Load the dataset
    data = pd.read_csv(data_path)

    # Display the first few rows
    print(data.head())
    print(data.info())
    # Basic statistics
    print(data.describe())
    # Check for missing values
    print(data.isnull().sum())

    data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')

    print(data['date'].head)
    print(data.duplicated().sum())
    data = data.drop_duplicates()

    # Plot outliers in 'Usage_kWh'
    plt.boxplot(data['Usage_kWh'])
    plt.title('Usage_kWh Outliers')
    plt.show()

    # Extract hour, month, and day from 'date'
    data['hour'] = data['date'].dt.hour
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day