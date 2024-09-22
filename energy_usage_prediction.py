import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor
from feature_engine.preprocessing import MatchVariables
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

def load_and_explore_data(data_path: str) -> pd.DataFrame:
    """
    Load the dataset and perform initial exploration.
    """
    data = pd.read_csv(data_path)
    print(data.head())
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())
    return data

def calculate_efficiency_score(facility_data: pd.Series, industry_benchmark: pd.Series) -> float:
    return facility_data.mean() / industry_benchmark.mean()


def preprocess_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset, converting the 'date' column to datetime
    and extracting year, month, day, and hour features.
    """
    if 'date' in X.columns:
        X['date'] = pd.to_datetime(X['date'], errors='coerce')
        
        if X['date'].isnull().any():
            print("Warning: Some date values could not be parsed and are set to NaT.")
        
        X['year'] = X['date'].dt.year
        X['month'] = X['date'].dt.month
        X['day'] = X['date'].dt.day
        X['hour'] = X['date'].dt.hour
        X.drop(columns=['date'], inplace=True)
    
    X = X.select_dtypes(include=[np.number])
    return X




def plot_outliers_and_distribution(data: pd.DataFrame) -> None:
    """
    Plot outliers and distribution of the 'Usage_kWh' column.
    """
    plt.boxplot(data['Usage_kWh'])
    plt.title('Usage_kWh Outliers')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(data['Usage_kWh'], kde=True, palette="viridis")
    plt.title('Distribution of Usage_kWh')
    plt.show()


def power_usage_by_hour_and_weekstatus(data: pd.DataFrame) -> None:
    """
    Plot the distribution of power usage by hour and week status.

    Args:
    data (pd.DataFrame): The dataset with 'WeekStatus' column.
    """
    # Check columns
    print("DataFrame columns:", data.columns)

    # Ensure 'WeekStatus' is in the DataFrame
    if 'WeekStatus' not in data.columns:
        print("Error: 'WeekStatus' column not found in data.")
        return

    # Proceed with plotting
    sns.boxplot(x='WeekStatus', y='Usage_kWh', palette="viridis", data=data)
    plt.title('Power Usage by Hour and Week Status')
    plt.show()



def calculate_corr_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and visualize the correlation matrix of numerical columns.
    """
    numeric_cols = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

    return corr_matrix


def apply_pca(data: pd.DataFrame, features_to_reduce: list[str]) -> pd.DataFrame:
    """
    Apply PCA to reduce the dimensionality of specified features.
    """
    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(data[features_to_reduce])
    data['PCA_Current_Power_Factor'] = pca_features[:, 0]
    print(f'Explained Variance Ratio: {pca.explained_variance_ratio_[0]}')
    data.drop(columns=features_to_reduce, inplace=True)
    return data


def scale_and_encode_data(data: pd.DataFrame, features_to_scale: list[str]) -> pd.DataFrame:
    """
    Apply scaling and one-hot encoding to the dataset.
    """
    print("DataFrame columns:", data.columns)

    columns_to_encode = ['WeekStatus', 'Day_of_week', 'Load_Type']
    missing_columns = [col for col in columns_to_encode if col not in data.columns]

    if missing_columns:
        print(f"Warning: The following columns are missing and will be skipped: {missing_columns}")
        columns_to_encode = [col for col in columns_to_encode if col in data.columns]

    # Perform one-hot encoding
    data = pd.get_dummies(data, columns=columns_to_encode, drop_first=True)
    scaler = StandardScaler()
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    return data


def split_data(data: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the data into training and testing sets.
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def compute_vif(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.
    """
    numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
    vif_data = pd.DataFrame()
    X = numeric_data.drop(columns=["Usage_kWh"], errors="ignore")  # Handle case where Usage_kWh is not present
    X = X.replace([np.inf, -np.inf], np.nan).dropna() 
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    return vif_data



def train_model_with_preprocessing(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model_type: str = 'hist_gb'
) -> Pipeline:
    """
    Train a model with a preprocessing pipeline.
    """
    categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_features)],
        remainder='passthrough'
    )

    # Choose model
    if model_type == 'hist_gb':
        classifier = HistGradientBoostingClassifier()
    elif model_type == 'rf':
        classifier = RandomForestClassifier()
    else:
        raise ValueError("Unsupported model type!")

    # Create full pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])

    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")
    print(f"Classification Report:\n{report}")

    return model


def calculate_cost_scenarios(
    model: Pipeline, X_test: pd.DataFrame, cost_per_kwh: float = 0.12
) -> None:
    """
    Predict power usage and cost for different scenarios.
    """
    normal_scenario = X_test.iloc[0:1].copy()
    high_demand_scenario = normal_scenario.copy()
    optimized_scenario = normal_scenario.copy()

    # Modify scenarios
    high_demand_scenario.loc[:, "hour"] = 12
    high_demand_scenario.loc[:, "Load_Type_Light Load"] = 0
    high_demand_scenario.loc[:, "Load_Type_Maximum Load"] = 1

    optimized_scenario.loc[:, "hour"] = 3
    optimized_scenario.loc[:, "Load_Type_Light Load"] = 1

    def predict_cost(input_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        predicted_usage = model.predict(input_data)
        predicted_cost = predicted_usage * cost_per_kwh
        return predicted_usage, predicted_cost

    # Calculate costs
    normal_usage, normal_cost = predict_cost(normal_scenario)
    high_usage, high_cost = predict_cost(high_demand_scenario)
    opt_usage, opt_cost = predict_cost(optimized_scenario)

    print(
        f"Normal Usage: {normal_usage[0]:.2f} kWh, Cost: ${normal_cost[0]:.2f}"
    )
    print(
        f"High Demand Usage: {high_usage[0]:.2f} kWh, Cost: ${high_cost[0]:.2f}"
    )
    print(f"Optimized Usage: {opt_usage[0]:.2f} kWh, Cost: ${opt_cost[0]:.2f}")
    print(f"Potential Savings: ${high_cost[0] - opt_cost[0]:.2f}")


def time_series_forecast(data: pd.Series, periods: int = 30) -> pd.Series:
    """
    Perform time-series forecasting using ARIMA.
    """
    model = ARIMA(data, order=(1, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=periods)[0]
    return pd.Series(forecast, index=data.index[-periods:])


def short_term_forecast(time_series: pd.Series) -> None:
    """
    Generate and plot a 30-day short-term forecast.
    """
    forecast = time_series_forecast(time_series, periods=30)

    plt.figure(figsize=(12, 6))
    time_series.plot(label="Historical Data")
    forecast.plot(label="30-day Forecast")
    plt.title("Short-term Power Usage Forecast")
    plt.xlabel("Date")
    plt.ylabel("Power Usage")
    plt.legend()
    plt.show()


def long_term_trend_analysis(time_series: pd.Series) -> None:
    """
    Analyze long-term trends and decompose the time-series data.
    """
    time_series = time_series.dropna() 
    decomposition = seasonal_decompose(time_series, model="additive", period=30)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    ax1.plot(time_series, label="Observed")
    ax1.legend(loc="best")
    ax2.plot(decomposition.trend, label="Trend")
    ax2.legend(loc="best")
    ax3.plot(decomposition.seasonal, label="Seasonal")
    ax3.legend(loc="best")
    ax4.plot(decomposition.resid, label="Residual")
    ax4.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print("\nLong-term Trend Analysis:")
    print(f"Overall trend: {'Increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'Decreasing'}")
    print(f"Seasonal pattern detected with a period of {decomposition.seasonal.shape[0]} days")


def compare_facility_performance(facility_performance: np.ndarray, industry_data: pd.DataFrame) -> None:
    """
    Compare facility performance against industry benchmarks and plot performance data.
    """
    industry_benchmark = industry_data.mean(axis=1).values  # Ensure this is an array
    efficiency_score = calculate_efficiency_score(
        pd.Series(facility_performance), pd.Series(industry_benchmark)
    )

    print(f"\nEfficiency Score: {efficiency_score:.2f}")
    print(f"{'Above' if efficiency_score > 1 else 'Below'} industry average")

    # Plot performance vs benchmarks
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=industry_data)
    plt.plot(
        range(len(industry_data.columns)),
        facility_performance[:len(industry_data.columns)],
        'ro',
        label='Analyzed Facility'
    )
    plt.title('Facility Performance vs Industry Benchmarks')
    plt.xlabel('Facilities')
    plt.ylabel('Monthly Average Power Usage')
    plt.legend()
    plt.show()

    industry_benchmark = industry_data.mean(axis=1).values

    print(f"Facility Performance Length: {len(facility_performance)}")
    print(f"Industry Benchmark Length: {len(industry_benchmark)}")

    # Repeat or truncate the benchmark to match the facility performance
    industry_average_repeated = np.repeat(industry_benchmark, len(facility_performance) // len(industry_benchmark))
    industry_average_repeated = industry_average_repeated[:len(facility_performance)]  # Ensure matching length

    monthly_comparison = pd.DataFrame({
        'Facility': facility_performance[:len(industry_average_repeated)],
        'Industry Average': industry_average_repeated
    })


    plt.figure(figsize=(12, 6))
    monthly_comparison.plot()
    plt.title('Monthly Comparison: Facility vs Industry Average')
    plt.xlabel('Date')
    plt.ylabel('Power Usage')
    plt.legend()
    plt.show()

    # Performance analysis
    outperform_months = (monthly_comparison['Facility'] < monthly_comparison['Industry Average']).sum()
    underperform_months = (monthly_comparison['Facility'] > monthly_comparison['Industry Average']).sum()

    print("\nPerformance Analysis:")
    print(f"Months outperforming industry average: {outperform_months}")
    print(f"Months underperforming industry average: {underperform_months}")

    if outperform_months > underperform_months:
        print("The facility is generally outperforming the industry average.")
    else:
        print("The facility is generally underperforming compared to the industry average.")
    print("Recommend focusing on months with the highest underperformance for targeted improvements.")

def simulate_scenario(X, changes):
    """
    Simulate the effect of changes in operational parameters or energy-saving measures on the input data.
    
    Parameters:
    X: np.ndarray or pd.DataFrame
        The input features.
    changes: dict
        A dictionary containing the parameters to change and their respective adjustment values.
        
    Returns:
    np.ndarray
        The modified predictions based on the changes applied.
    """
    # Create a copy of the data to avoid modifying the original data
    X_modified = X.copy()
    
    # Apply changes based on the keys in the changes dictionary
    for param, change in changes.items():
        if param in X_modified.columns:  # Ensure the parameter exists in the DataFrame
            X_modified[param] *= (1 + change)  # Apply the percentage change
            
    # Return the modified predictions (assuming you want to predict after changes)
    return model.predict(X_modified)  # Make sure 'model' is accessible here

 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import pandas as pd

def detect_anomalies(X, contamination=0.05):
    """
    Detect anomalies in the dataset using Isolation Forest.
    
    Parameters:
    X: np.ndarray or pd.DataFrame
        The input features.
    contamination: float
        The proportion of anomalies in the data set.

    Returns:
    np.ndarray
        Boolean array where True indicates anomalies.
    """
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_scaled = imputer.fit_transform(X_scaled)

    # Use Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)

    # -1 for anomalies, 1 for normal points
    return anomaly_labels == -1 

def detect_and_simulate_anomalies(X, model, contamination=0.05):
    """
    Detect anomalies in the data and simulate different scenarios.
    
    Parameters:
    X: np.ndarray or pd.DataFrame
        The input features.
    model: trained model
        The model to make predictions.
    contamination: float
        The proportion of anomalies in the data set.
    """
    anomalies = detect_anomalies(X, contamination)
    
    if any(anomalies):
        anomaly_examples = X[anomalies].head()
        print("\nExamples of detected anomalies:")
        print(anomaly_examples)

    # Simulate different scenarios
    operational_changes = {
        'Temperature': 0.1,  # 10% increase in temperature
        'Production_Rate': 0.05  # 5% increase in production rate
    }

    energy_saving_changes = {
        'Equipment_Efficiency': 0.15,  # 15% increase in equipment efficiency
        'Cooling_System_Power': -0.2  # 20% decrease in cooling system power
    }

    original_predictions = model.predict(X)
    modified_predictions = simulate_scenario(X, operational_changes)
    energy_saving_predictions = simulate_scenario(X, energy_saving_changes)

    print("\nSimulated Scenario - Adjusting Operational Parameters:")
    print(f"Average original power usage: {original_predictions.mean():.2f}")
    print(f"Average modified power usage: {modified_predictions.mean():.2f}")
    print(f"Percentage change: {((modified_predictions.mean() - original_predictions.mean()) / original_predictions.mean()) * 100:.2f}%")

    print("\nSimulated Scenario - Implementing Energy-Saving Measures:")
    print(f"Average original power usage: {original_predictions.mean():.2f}")
    print(f"Average power usage after energy-saving measures: {energy_saving_predictions.mean():.2f}")
    print(f"Percentage change: {((energy_saving_predictions.mean() - original_predictions.mean()) / original_predictions.mean()) * 100:.2f}%")

    # Visualize the impact of scenarios
    plt.figure(figsize=(12, 6))
    plt.boxplot([original_predictions, modified_predictions, energy_saving_predictions], 
                labels=['Original', 'Operational Changes', 'Energy-Saving Measures'])
    plt.title('Impact of Different Scenarios on Power Usage')
    plt.ylabel('Predicted Power Usage')
    plt.show()

from sklearn.model_selection import GridSearchCV
 
def optimize_model_parameters(X, y):
    """
    Find the optimal parameters for a RandomForestRegressor model.
    
    Parameters:
    X: np.ndarray or pd.DataFrame
        The input features.
    y: np.ndarray or pd.Series
        The target variable.
        
    Returns:
    tuple
        A tuple containing the best parameters and the best estimator.
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Create a new RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    grid_search.fit(X, y)
    
    return grid_search.best_params_, grid_search.best_estimator_
def show_feature_importance(best_model: RandomForestRegressor, X: pd.DataFrame) -> None:
    """Generate feature importance
    """
    # Check for null pointer references
    if best_model is None:
        raise ValueError("best_model is None")
    if X is None:
        raise ValueError("X is None")

    feature_importance = best_model.feature_importances_
    feature_names = X.columns

    # Check for unhandled exceptions
    if feature_importance is None:
        raise ValueError("feature_importance is None")
    if feature_names is None:
        raise ValueError("feature_names is None")

    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(10, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, feature_names[sorted_idx])
    plt.title('Feature Importance for Power Usage')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    print("\nTop 5 most important features:")
    for i in sorted_idx[-5:]:
        print(f"{feature_names[i]}: {feature_importance[i]:.4f}")

    # b) Identify areas for efficiency improvements
    print("\nRecommendations for efficiency improvements:")
    print("1. Focus on optimizing the top 3 most important features.")
    print("2. Investigate features with low importance for potential removal or consolidation.")
    print("3. Consider collecting more data on mid-range importance features to improve model accuracy.")





# Example usage flow:

# Load the dataset
data_path = '/Users/syedmohathashimali/Pratice/Steel_industry_data.csv'
data = load_and_explore_data(data_path)

# Preprocess the data
data = preprocess_data(data)

# Visualize outliers and distribution
plot_outliers_and_distribution(data)

# Visualize power usage by hour and week status
power_usage_by_hour_and_weekstatus(data)

# Calculate and visualize correlation matrix
corr_matrix = calculate_corr_matrix(data)

# Apply PCA
# Updated features for PCA
features_to_reduce = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh']
data = apply_pca(data, features_to_reduce)

# Scale and encode the data
features_to_scale = ['Usage_kWh', 'hour', 'day', 'month']
data = scale_and_encode_data(data, features_to_scale)

# Split the data into training and testing sets
target_column = 'Usage_kWh'
X_train, X_test, y_train, y_test = split_data(data, target_column)

# Compute Variance Inflation Factor (VIF)
vif_data = compute_vif(data)

# Train and evaluate the model
model = train_model_with_preprocessing(X_train, X_test, y_train.astype(int), y_test.astype(int), model_type='rf')

# Predict usage and cost under different scenarios
calculate_cost_scenarios(model, X_test)
calculate_cost_scenarios(model, X_train)

# Compare facility performance against industry benchmarks
compare_facility_performance(model.predict(X_test), data)

# Detect anomalies
detect_and_simulate_anomalies(X_train, model)

# Separate features (X) and target (y)
X = data.drop(columns=['Usage_kWh'])  # Replace 'target_column' with your actual target column name
y = data['Lagging_Current_Power_Factor']

# Now call the functions
best_model, _ = optimize_model_parameters(X, y)
show_feature_importance(best_model, X)





