import pandas as pd
import numpy as np
import datetime
import time

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Plotting (Optional)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
WEATHER_DATA_PATH = "synthetic_weather_data.csv"
OPERATIONS_DATA_PATH = "synthetic_operations_log.csv"
# !!! IMPORTANT: Set this to the actual start date used for weather generation !!!
SIMULATION_START_DATE = '2023-01-01'

# --- Feature Engineering Parameters ---
FORECAST_HORIZON_HOURS = 48 # How many hours of weather forecast to use as features
TIME_WINDOWS = [6, 12, 24, 48] # Summary windows for weather features (in hours)
TARGET_COLUMN = 'total_weather_delay_hrs'

# --- Load Data ---
print("Loading data...")
try:
    weather_df = pd.read_csv(WEATHER_DATA_PATH, index_col=0, parse_dates=True)
    ops_df = pd.read_csv(OPERATIONS_DATA_PATH)
    print(f"Loaded {len(weather_df)} weather records and {len(ops_df)} operations records.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the CSV files exist at the specified paths.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- Feature Engineering ---
print("Performing feature engineering...")
base_datetime = pd.to_datetime(SIMULATION_START_DATE)

# 1. Convert arrival_time (hours) to actual timestamp
ops_df['arrival_timestamp'] = ops_df['arrival_time'].apply(lambda h: base_datetime + pd.Timedelta(hours=h))

# 2. Add basic time features from arrival_timestamp
ops_df['arrival_hour'] = ops_df['arrival_timestamp'].dt.hour
ops_df['arrival_dayofweek'] = ops_df['arrival_timestamp'].dt.dayofweek
ops_df['arrival_month'] = ops_df['arrival_timestamp'].dt.month

# 3. Function to extract future weather features for a given arrival time
def get_weather_features(arrival_ts, weather_data, horizon_h, windows):
    """
    Extracts summary statistics of weather variables over future time windows.
    """
    forecast_start_time = arrival_ts
    forecast_end_time = forecast_start_time + pd.Timedelta(hours=horizon_h)

    # Select the relevant forecast period from weather data
    # Ensure index is sorted for efficient lookup
    if not weather_data.index.is_monotonic_increasing:
         weather_data = weather_data.sort_index()

    # Use slicing which is efficient on DatetimeIndex
    forecast_data = weather_data.loc[forecast_start_time : forecast_end_time]

    if forecast_data.empty:
        # Handle cases where arrival is too late for the forecast horizon
        # Return NaNs or default values - NaNs are better to see issues
        cols = [f'{var}_{agg}_{w}h'
                for w in windows
                for var in ['wind_speed_knots', 'visibility_nm', 'wave_height_m'] # Add other relevant vars
                for agg in ['mean', 'max', 'min', 'std']]
        return pd.Series(index=cols, dtype=np.float64)

    features = {}
    weather_vars_to_summarize = ['wind_speed_knots', 'visibility_nm', 'wave_height_m'] # Add others if needed

    for window_h in windows:
        window_end_time = forecast_start_time + pd.Timedelta(hours=window_h)
        window_data = forecast_data.loc[forecast_start_time : window_end_time]

        if window_data.empty: continue # Skip if window is empty

        for var in weather_vars_to_summarize:
            if var not in window_data.columns: continue # Skip if weather var is missing

            series = window_data[var]
            features[f'{var}_mean_{window_h}h'] = series.mean()
            features[f'{var}_max_{window_h}h'] = series.max()
            # Min visibility is important, min wind/wave might be less so, but include for now
            features[f'{var}_min_{window_h}h'] = series.min()
            features[f'{var}_std_{window_h}h'] = series.std() # Captures variability

    # Example: Add count of 'Fog' or 'HighWind' states if 'state' column exists
    if 'state' in forecast_data.columns:
         for window_h in windows:
             window_end_time = forecast_start_time + pd.Timedelta(hours=window_h)
             window_data = forecast_data.loc[forecast_start_time : window_end_time]
             if window_data.empty: continue
             for state in ['Fog', 'HighWind']: # Add other critical states
                 features[f'{state}_hours_{window_h}h'] = (window_data['state'] == state).sum()


    return pd.Series(features)

# Apply the function - This can be slow for large datasets!
print(f"Extracting weather features for {len(ops_df)} vessels (horizon={FORECAST_HORIZON_HOURS}h)...")
start_feature_time = time.time()
weather_feature_list = ops_df['arrival_timestamp'].apply(
    get_weather_features,
    args=(weather_df, FORECAST_HORIZON_HOURS, TIME_WINDOWS)
)
end_feature_time = time.time()
print(f"Weather feature extraction took {end_feature_time - start_feature_time:.2f} seconds.")

# Join features back to ops_df
model_data = pd.concat([ops_df, weather_feature_list], axis=1)

# --- Data Cleaning ---
print("Cleaning data...")
# Drop rows where weather features couldn't be calculated (e.g., arrivals too close to end of weather data)
initial_rows = len(model_data)
model_data.dropna(subset=weather_feature_list.columns, inplace=True) # Drop rows missing weather features
model_data.dropna(subset=[TARGET_COLUMN], inplace=True) # Ensure target is not NaN
# Impute NaNs in STD columns (if window had only 1 data point, std is NaN) - use 0
std_cols = [col for col in weather_feature_list.columns if '_std_' in col]
model_data[std_cols] = model_data[std_cols].fillna(0)

# Optional: Check for other NaNs and decide on imputation strategy if needed
# print(model_data.isnull().sum())

final_rows = len(model_data)
print(f"Dropped {initial_rows - final_rows} rows due to missing features/target.")
if final_rows == 0:
    print("Error: No data remaining after cleaning. Check weather data coverage and feature extraction.")
    exit()

# --- Define Features (X) and Target (y) ---
categorical_features = ['type', 'arrival_hour', 'arrival_dayofweek', 'arrival_month']
# Ensure all numerical features are included
numerical_features = ['teu'] + list(weather_feature_list.columns)

# Remove constant columns if any (e.g., if a state never occurred)
for col in numerical_features[:]: # Iterate over a copy
    if col in model_data.columns and model_data[col].nunique() <= 1:
        print(f"Warning: Removing constant or near-constant feature: {col}")
        numerical_features.remove(col)


X = model_data[categorical_features + numerical_features]
y = model_data[TARGET_COLUMN]

print(f"Features selected: {list(X.columns)}")
print(f"Target selected: {TARGET_COLUMN}")

# --- Train-Test Split ---
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- Preprocessing Pipeline ---
print("Setting up preprocessing pipeline...")
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # handle_unknown='ignore' is important

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# --- Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), # n_jobs=-1 uses all cores
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, objective='reg:squarederror') # Use squared error objective
}

# --- Train, Predict, and Evaluate Models ---
results = {}
trained_pipelines = {}

print("Training and evaluating models...")
for name, model in models.items():
    start_train_time = time.time()
    print(f"\n--- Training {name} ---")

    # Create full pipeline with preprocessing and model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Train
    pipeline.fit(X_train, y_train)
    trained_pipelines[name] = pipeline # Store the trained pipeline

    # Predict
    y_pred = pipeline.predict(X_test)

    # Ensure predictions are non-negative (delay cannot be negative)
    y_pred = np.maximum(0, y_pred)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    end_train_time = time.time()
    duration = end_train_time - start_train_time

    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Training Time (s)': duration}
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Training/Evaluation Time: {duration:.2f} seconds")

# --- Compare Models ---
print("\n--- Model Comparison ---")
results_df = pd.DataFrame(results).T # Transpose for better readability
results_df = results_df.sort_values(by='RMSE', ascending=True) # Sort by RMSE (lower is better)
print(results_df)

# --- Optional: Feature Importance (for tree-based models) ---
for name in ["Random Forest", "XGBoost"]:
    if name in trained_pipelines:
        print(f"\n--- Feature Importances for {name} ---")
        try:
            # Need to get feature names after one-hot encoding
            feature_names_raw = numerical_features + \
                list(trained_pipelines[name].named_steps['preprocessor']
                     .named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features))

            importances = trained_pipelines[name].named_steps['regressor'].feature_importances_

            # Handle cases where number of features doesn't match importances length
            if len(feature_names_raw) != len(importances):
                 print(f"Warning: Mismatch between feature names ({len(feature_names_raw)}) and importances ({len(importances)}) for {name}. Skipping plot.")
                 continue


            feature_importance_df = pd.DataFrame({'Feature': feature_names_raw, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(20) # Show top 20

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
            plt.title(f'Top 20 Feature Importances - {name}')
            plt.tight_layout()
            plt.savefig(f"{name}_feature_importance.png") # Save the plot
            print(f"Saved feature importance plot to {name}_feature_importance.png")
            # plt.show() # Uncomment to display plots directly

        except AttributeError:
            print(f"Could not retrieve feature importances for {name}.")
        except Exception as e:
            print(f"An error occurred during feature importance plotting for {name}: {e}")


print("\n--- Script Finished ---")