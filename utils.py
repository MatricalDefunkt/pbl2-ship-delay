# utils.py
import pandas as pd
import numpy as np

# --- Make these match your training configuration ---
FORECAST_HORIZON_HOURS = 48
TIME_WINDOWS = [6, 12, 24, 48]
WEATHER_VARS_TO_SUMMARIZE = [
    "wind_speed_knots",
    "visibility_nm",
    "wave_height_m",
]  # Add/remove based on training
CRITICAL_STATES = ["Fog", "HighWind"]  # Add/remove based on training
# --- End of configuration ---


def calculate_weather_features_from_forecast(arrival_ts, hourly_forecast_list):
    """
    Calculates weather summary features from a list of hourly forecast dictionaries.

    Args:
        arrival_ts (pd.Timestamp): The vessel's arrival timestamp.
        hourly_forecast_list (list): A list of dictionaries, each representing one hour
                                      of forecast data (e.g.,
                                      {'timestamp': '2023-01-01T10:00:00Z', 'wind_speed_knots': 10, ...})

    Returns:
        pd.Series: A series containing the calculated weather summary features.
                   Returns an empty Series on error or insufficient data.
    """
    if not hourly_forecast_list:
        print("Warning: Received empty hourly_forecast_list.")
        return pd.Series(dtype=np.float64)

    try:
        # Convert list of dicts to DataFrame
        forecast_df = pd.DataFrame(hourly_forecast_list)
        if "timestamp" not in forecast_df.columns:
            raise ValueError("Forecast data must contain a 'timestamp' column.")

        # Convert timestamp string/object to datetime and set as index
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
        forecast_df = forecast_df.set_index("timestamp").sort_index()

        # Select the relevant forecast period (from arrival onwards)
        forecast_start_time = arrival_ts
        # Use the data frame index max if it's less than theoretical horizon end
        horizon_end_time = forecast_start_time + pd.Timedelta(
            hours=FORECAST_HORIZON_HOURS
        )
        forecast_end_time = min(horizon_end_time, forecast_df.index.max())

        # Slice the data for the required forecast horizon starting from arrival_ts
        # Allow for slightly flexible start time if exact arrival_ts isn't in index
        forecast_data_period = forecast_df.loc[forecast_df.index >= forecast_start_time]
        if forecast_data_period.empty:
            print(
                f"Warning: No forecast data found at or after arrival_ts {forecast_start_time}"
            )
            return pd.Series(dtype=np.float64)

        # Further slice up to the calculated end time
        forecast_data_period = forecast_data_period.loc[
            forecast_data_period.index <= forecast_end_time
        ]

        if forecast_data_period.empty:
            print(
                f"Warning: No forecast data found within the effective horizon {forecast_start_time} to {forecast_end_time}"
            )
            return pd.Series(dtype=np.float64)

        # --- Feature Calculation (similar to training script) ---
        features = {}
        required_vars = set(WEATHER_VARS_TO_SUMMARIZE)
        if "state" in forecast_df.columns:
            required_vars.add("state")

        if not required_vars.issubset(forecast_df.columns):
            missing = required_vars - set(forecast_df.columns)
            raise ValueError(f"Forecast data missing required columns: {missing}")

        for window_h in TIME_WINDOWS:
            window_end_time = forecast_start_time + pd.Timedelta(hours=window_h)
            # Slice the already selected period for the current window
            window_data = forecast_data_period.loc[
                forecast_data_period.index <= window_end_time
            ]

            if window_data.empty:
                continue

            for var in WEATHER_VARS_TO_SUMMARIZE:
                series = window_data[var]
                features[f"{var}_mean_{window_h}h"] = series.mean()
                features[f"{var}_max_{window_h}h"] = series.max()
                features[f"{var}_min_{window_h}h"] = series.min()
                # Calculate std, handle potential NaN if only one data point
                std_val = series.std()
                features[f"{var}_std_{window_h}h"] = (
                    std_val if pd.notna(std_val) else 0.0
                )

            # Calculate state counts if 'state' column exists
            if "state" in window_data.columns:
                for state in CRITICAL_STATES:
                    features[f"{state}_hours_{window_h}h"] = (
                        window_data["state"] == state
                    ).sum()

        return pd.Series(features)

    except Exception as e:
        print(f"Error calculating weather features: {e}")
        import traceback

        traceback.print_exc()  # Print detailed traceback for debugging
        return pd.Series(dtype=np.float64)  # Return empty series on error
