import pandas as pd
import numpy as np
import datetime

def generate_weather_data(start_date, end_date, freq='h', lat=None):
    """
    Generates synthetic weather data for a given time period and frequency.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        freq (str): Pandas frequency string (e.g., 'h' for hourly, '3h', 'd').
        lat (float, optional): Latitude for basic seasonal adjustments.
                               Defaults to None (generic seasonality).

    Returns:
        pandas.DataFrame: DataFrame with synthetic weather data.
    """
    print(f"Generating data from {start_date} to {end_date} with frequency '{freq}'...")

    # Create DatetimeIndex
    try:
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq, name='timestamp')
    except ValueError as e:
        print(f"Error creating date range: {e}")
        print("Ensure start_date and end_date are valid and start_date <= end_date.")
        return None

    n_periods = len(timestamps)
    if n_periods == 0:
        print("Error: Date range resulted in zero periods.")
        return None

    print(f"Number of periods to generate: {n_periods}")
    df = pd.DataFrame(index=timestamps)

    # --- Generate Temperature (°C) ---
    base_temp = 15
    seasonal_amplitude = 10
    daily_amplitude = 5

    day_of_year = df.index.dayofyear
    phase_shift = -80 if lat is None or lat >= 0 else 100
    seasonal_variation = seasonal_amplitude * np.sin(2 * np.pi * (day_of_year + phase_shift) / 365.25)

    hour_of_day = df.index.hour
    _daily_variation_calc = daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 6) / 24) # Peak around 2 PM
    daily_variation = np.array(_daily_variation_calc) # Ensure numpy array

    random_noise_temp = np.random.normal(0, 1.5, n_periods)

    df['temperature_celsius'] = base_temp + seasonal_variation + daily_variation + random_noise_temp
    df['temperature_celsius'] = df['temperature_celsius'].round(1)

    # --- Generate Wind Speed (knots) ---
    base_wind = 8
    wind_variation = np.random.rand(n_periods) * 15
    gust_chance = 0.05
    gust_intensity = np.random.rand(n_periods) * 20
    is_gusting = np.random.rand(n_periods) < gust_chance
    df['wind_speed_knots'] = base_wind + wind_variation + (is_gusting * gust_intensity)
    df['wind_speed_knots'] += (daily_variation / 5).clip(0) # Small effect
    df['wind_speed_knots'] = df['wind_speed_knots'].clip(0).round(1) # Ensure non-negative final result

    # --- Generate Wind Direction (degrees) ---
    df['wind_direction_deg'] = np.random.uniform(0, 360, n_periods).round(0)

    # --- Generate Precipitation (mm/hour) ---
    precip_chance = 0.08
    precip_intensity_scale = 2.0
    is_raining = np.random.rand(n_periods) < precip_chance
    precip_values = np.random.exponential(scale=precip_intensity_scale, size=n_periods)
    df['precipitation_mmhr'] = np.where(is_raining, precip_values, 0)
    df['precipitation_mmhr'] = df['precipitation_mmhr'].round(2)

    # --- Generate Visibility (nautical miles) ---
    base_visibility_nm = 10.0
    visibility_noise = np.random.normal(0, 0.5, n_periods)
    df['visibility_nm'] = base_visibility_nm + visibility_noise

    # Reduce visibility based on precipitation (applying mask to right side too)
    heavy_precip_threshold = 3.0
    visibility_reduction_precip = np.random.uniform(0.2, 1.5, n_periods)
    precip_mask = df['precipitation_mmhr'] > heavy_precip_threshold
    # Calculate the reduction amount only for relevant points
    reduction_amount = (visibility_reduction_precip * df['precipitation_mmhr'])[precip_mask]
    # Apply the reduction using loc with the mask
    df.loc[precip_mask, 'visibility_nm'] -= reduction_amount

    # Add occasional fog events (independent of rain for simplicity here)
    fog_chance = 0.015
    is_foggy = np.random.rand(n_periods) < fog_chance
    fog_visibility = np.random.uniform(0.1, 0.5, n_periods) # Potential fog values for all points

    # ****** Applying boolean mask to the right side (value being assigned) ******
    df.loc[is_foggy, 'visibility_nm'] = fog_visibility[is_foggy]
    # *******************************************************************************

    # Clip visibility to realistic bounds AFTER all modifications
    df['visibility_nm'] = df['visibility_nm'].clip(0.1, 10.0).round(1)


    # --- Generate Wave Height (meters) ---
    base_wave_height = 0.2
    wind_wave_factor = 0.08
    wave_noise = np.random.normal(0, 0.15, n_periods)

    df['wave_height_m'] = base_wave_height + (df['wind_speed_knots'] * wind_wave_factor) + wave_noise
    # Use np.where for cleaner conditional assignment
    increase_factor = np.random.uniform(1.2, 1.8, n_periods)
    df['wave_height_m'] = np.where(is_gusting, df['wave_height_m'] * increase_factor, df['wave_height_m'])

    df['wave_height_m'] = df['wave_height_m'].clip(0).round(2)


    print("Data generation complete.")
    return df

# --- Configuration ---
START_DATE = '2022-01-01'
END_DATE = '2023-12-31'
FREQUENCY = 'h'       # 'h' for hourly, '3h', 'd' for daily etc.
OUTPUT_FILENAME = 'synthetic_weather_data.csv'
LATITUDE = 33.7       # Optional: Approximate latitude for seasonality (e.g., Long Beach, CA)
                      # Set to None for generic Northern Hemisphere seasonality

# --- Generate and Save ---
weather_df = generate_weather_data(START_DATE, END_DATE, FREQUENCY, LATITUDE)

if weather_df is not None:
    print(f"\nGenerated DataFrame head:\n{weather_df.head()}")
    print(f"\nGenerated DataFrame info:\n")
    weather_df.info()
    print(f"\nGenerated DataFrame basic stats:\n{weather_df.describe()}")

    # Save to CSV
    try:
        weather_df.to_csv(OUTPUT_FILENAME)
        print(f"\nSynthetic weather data saved to '{OUTPUT_FILENAME}'")
    except Exception as e:
        print(f"\nError saving file: {e}")