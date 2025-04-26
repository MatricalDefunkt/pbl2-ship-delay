import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import simpy

def generate_weather_data(start_date_str, end_date_str, freq='H'):
    """
    Generates synthetic hourly weather data for a port using a Markov chain
    for weather states and conditional distributions.

    Args:
        start_date_str (str): Start date 'YYYY-MM-DD'
        end_date_str (str): End date 'YYYY-MM-DD'
        freq (str): Pandas frequency string (default 'H' for hourly)

    Returns:
        pandas.DataFrame: DataFrame with timestamped weather data.
    """
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_periods = len(timestamps)

    # --- 1. Markov Chain for Weather States ---
    states = ['Clear', 'Fog', 'Rain', 'ModerateWind', 'HighWind']
    # Simplified Transition Matrix (P(to_state | from_state)) - NEEDS CALIBRATION!
    # Rows/Columns correspond to states list order
    # High persistence on diagonal. Low chance of jumping from Clear to HighWind etc.
    # Should ideally vary by season.
    transition_matrix = np.array([
        [0.90, 0.03, 0.03, 0.03, 0.01], # From Clear
        [0.20, 0.75, 0.02, 0.03, 0.00], # From Fog
        [0.10, 0.05, 0.75, 0.10, 0.00], # From Rain
        [0.15, 0.05, 0.10, 0.65, 0.05], # From ModerateWind
        [0.10, 0.00, 0.05, 0.20, 0.65]  # From HighWind
    ])

    weather_states = ['Clear'] # Start state
    current_state_idx = states.index(weather_states[0])

    for _ in range(1, n_periods):
        next_state_idx = np.random.choice(
            len(states),
            p=transition_matrix[current_state_idx, :]
        )
        weather_states.append(states[next_state_idx])
        current_state_idx = next_state_idx

    # --- 2. Generate Variables Conditionally on State ---
    weather_data = pd.DataFrame(index=timestamps)
    weather_data['state'] = weather_states

    wind_speeds = []
    visibilities = []
    precipitations = []
    wave_heights = []

    # Define parameters for distributions based on state (EXAMPLE VALUES!)
    # Format: {state: {variable: {param_name: value}}}
    state_params = {
        'Clear': {'wind_shape': 1.8, 'wind_scale': 5, 'vis_a': 15, 'vis_b': 1, 'wave_scale': 0.5},
        'Fog': {'wind_shape': 1.5, 'wind_scale': 3, 'vis_a': 1, 'vis_b': 10, 'wave_scale': 0.3},
        'Rain': {'wind_shape': 1.7, 'wind_scale': 8, 'vis_a': 5, 'vis_b': 2, 'precip_a': 0.8, 'precip_scale': 3, 'wave_scale': 1.0},
        'ModerateWind': {'wind_shape': 2.0, 'wind_scale': 15, 'vis_a': 10, 'vis_b': 1, 'wave_scale': 1.5},
        'HighWind': {'wind_shape': 2.2, 'wind_scale': 25, 'vis_a': 8, 'vis_b': 1.5, 'wave_scale': 3.0}
    }
    # Visibility Range (e.g., 0 to 10 Nautical Miles)
    VIS_MAX = 10.0

    for i in range(n_periods):
        state = weather_data['state'].iloc[i]
        params = state_params[state]
        month = weather_data.index[i].month # For seasonality (example)

        # Wind Speed (Weibull) - Add some noise/variability
        # Scale could slightly vary by month here if needed
        ws = stats.weibull_min.rvs(params['wind_shape'], loc=0, scale=params['wind_scale'] * (1 + np.random.randn()*0.1))
        wind_speeds.append(max(0, ws)) # Ensure non-negative

        # Visibility (Beta scaled)
        # Make fog more likely in certain months (e.g., May/June/Winter for US West Coast)
        vis_a, vis_b = params['vis_a'], params['vis_b']
        if state == 'Fog' and month in [5, 6, 12, 1]:
             vis_b *= 1.5 # Increase skew towards low visibility
        elif state == 'Clear' and month not in [5, 6, 12, 1]:
             vis_a *= 1.2 # Increase skew towards high visibility

        visibility = stats.beta.rvs(vis_a, vis_b) * VIS_MAX
        visibilities.append(visibility)

        # Precipitation (Gamma if Rain state, else 0)
        if state == 'Rain':
            # Add seasonality: Higher intensity/chance in winter months
            precip_intensity = stats.gamma.rvs(params['precip_a'], loc=0, scale=params['precip_scale'])
            if month in [11, 12, 1, 2, 3]:
                 precip_intensity *= 1.5
            else:
                 precip_intensity *= 0.5 # Dryer season
            precipitations.append(max(0, precip_intensity))
        else:
            precipitations.append(0)

        # Wave Height (Correlated with Wind Speed, base on state) - Simplified approach
        # Example: Log-normal based on wind speed and base state wave scale
        base_wave = params['wave_scale']
        # Simple correlation: higher wind -> higher waves (needs proper physical modelling/data)
        wind_effect = np.log1p(ws) * 0.1 # Log transform to moderate effect
        # Use lognormal: sigma parameter controls variability, scale related to mean
        wave_sigma = 0.4
        wave_mu = np.log(base_wave + wind_effect)
        wh = stats.lognorm.rvs(s=wave_sigma, scale=np.exp(wave_mu))
        wave_heights.append(max(0.1, wh)) # Ensure minimum wave height

    weather_data['wind_speed_knots'] = wind_speeds
    weather_data['visibility_nm'] = visibilities
    weather_data['precipitation_mmhr'] = precipitations
    weather_data['wave_height_m'] = wave_heights

    # Wind Direction (Simplified - random choice, could depend on state)
    directions = [0, 45, 90, 135, 180, 225, 270, 315] # N, NE, E, SE, S, SW, W, NW
    # Example: HighWind state often has specific direction (e.g., offshore for Santa Ana)
    wind_directions = []
    for state in weather_data['state']:
        if state == 'HighWind': # Example: Bias towards NE/E for Santa Ana
            direction = np.random.choice(directions, p=[0.1, 0.3, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05])
        else: # Default: More common onshore flow (e.g., W/NW for US West Coast)
            direction = np.random.choice(directions, p=[0.1, 0.05, 0.05, 0.05, 0.1, 0.2, 0.3, 0.15])
        wind_directions.append(direction)
    weather_data['wind_direction_deg'] = wind_directions


    return weather_data

# Example Usage:
# weather_df = generate_weather_data('2023-01-01', '2023-12-31')
# print(weather_df.head())
# print(weather_df.describe())
# weather_df['state'].value_counts().plot(kind='bar', title='Simulated Weather State Frequency')

# --- Helper Function to Get Weather Data ---
def get_weather(sim_time_hours, weather_df):
    """Looks up weather data for the closest hour."""
    # Convert simulation time (hours) to timestamp
    current_timestamp = weather_df.index[0] + pd.Timedelta(hours=sim_time_hours)
    # Find the closest index in the weather dataframe
    closest_timestamp = weather_df.index.asof(current_timestamp)
    if pd.isna(closest_timestamp) or closest_timestamp > weather_df.index[-1]:
         # Handle edge case: Simulation time beyond weather data
         print(f"Warning: Simulation time {sim_time_hours}hr outside weather data range.")
         return weather_df.iloc[-1] # Return last known weather
    return weather_df.loc[closest_timestamp]

# --- Simulation Parameters (Example Values) ---
SIM_DURATION_HOURS = 24 * 30 * 3 # Hours * Days * Months
AVG_ARRIVAL_RATE_PER_HOUR = 0.75 # 0.5: Average 1 vessel every 2 hours
NUM_BERTHS = 5
CHECK_INTERVAL_HOURS = 1 # How often to re-check weather/resources if waiting

# Weather Thresholds (Port-wide - could be vessel-specific)
VIS_MIN_ENTRY_NM = 0.5
WIND_MAX_BERTHING_KNOTS = 30
WIND_MAX_CARGO_KNOTS = 40
WAVE_MAX_ENTRY_M = 3.0

# Vessel Characteristics Distributions (Example)
VESSEL_TYPES = ['Container_ULCS', 'Container_PostPanamax', 'Container_Panamax']
VESSEL_PROBS = [0.4, 0.4, 0.2]
TEU_MEAN = {'Container_ULCS': 18000, 'Container_PostPanamax': 10000, 'Container_Panamax': 5000}
TEU_STD_DEV_FACTOR = 0.1 # Std dev as fraction of mean
HANDLING_RATE_TEU_PER_HOUR = 100 # TEUs handled per hour per vessel at berth

# Data Collector
event_log = []

# --- SimPy Process Functions ---
def vessel_process(env, vessel_id, characteristics, resources, weather_df):
    """Simulates the lifecycle of a single vessel."""
    arrival_time = env.now
    vessel_type = characteristics['type']
    vessel_teu = characteristics['teu']
    log_entry = {'vessel_id': vessel_id, 'type': vessel_type, 'teu': vessel_teu,
                   'arrival_time': arrival_time}

    print(f"{env.now:.2f} hr: Vessel {vessel_id} ({vessel_type}) arrives.")

    # --- 1. Entry / Channel Transit ---
    entry_permit_time = None
    weather_delay_entry = 0
    while True:
        weather = get_weather(env.now, weather_df)
        # Check Entry Conditions
        if weather['visibility_nm'] < VIS_MIN_ENTRY_NM or weather['wave_height_m'] > WAVE_MAX_ENTRY_M:
            print(f"{env.now:.2f} hr: Vessel {vessel_id} waiting for entry weather (Vis: {weather['visibility_nm']:.1f}, Wave: {weather['wave_height_m']:.1f})")
            delay_start = env.now
            yield env.timeout(CHECK_INTERVAL_HOURS)
            weather_delay_entry += (env.now - delay_start)
        else:
            # Assume channel is available (simplify - could add channel resource)
            entry_permit_time = env.now
            print(f"{env.now:.2f} hr: Vessel {vessel_id} entry permitted.")
            # Simulate transit time to berth area
            transit_duration = 1.0 # Simplified fixed duration
            yield env.timeout(transit_duration)
            break # Exit weather check loop

    log_entry['entry_permit_time'] = entry_permit_time
    log_entry['weather_delay_entry_hrs'] = weather_delay_entry

    # --- 2. Request Berth & Berthing ---
    print(f"{env.now:.2f} hr: Vessel {vessel_id} requests berth.")
    berth_request_time = env.now
    berth_start_time = None
    weather_delay_berthing = 0

    with resources['berths'].request() as req:
        # Wait for a berth to become available
        yield req
        print(f"{env.now:.2f} hr: Vessel {vessel_id} assigned berth. Checking berthing conditions.")
        berth_assigned_time = env.now

        # Check Berthing Conditions
        while True:
            weather = get_weather(env.now, weather_df)
            if weather['wind_speed_knots'] > WIND_MAX_BERTHING_KNOTS:
                 print(f"{env.now:.2f} hr: Vessel {vessel_id} waiting for berthing weather (Wind: {weather['wind_speed_knots']:.1f})")
                 delay_start = env.now
                 yield env.timeout(CHECK_INTERVAL_HOURS)
                 weather_delay_berthing += (env.now - delay_start)
            else:
                 print(f"{env.now:.2f} hr: Vessel {vessel_id} starting berthing maneuvers.")
                 berthing_duration = 0.5 # Simplified fixed duration (could depend on weather/size)
                 yield env.timeout(berthing_duration)
                 berth_start_time = env.now
                 print(f"{env.now:.2f} hr: Vessel {vessel_id} berthed.")
                 break # Exit weather check loop

    log_entry['berth_wait_time_hrs'] = berth_assigned_time - berth_request_time # Time spent waiting *after* requesting
    log_entry['weather_delay_berthing_hrs'] = weather_delay_berthing
    log_entry['berth_start_time'] = berth_start_time

    # --- 3. Cargo Operations ---
    cargo_ops_duration = vessel_teu / HANDLING_RATE_TEU_PER_HOUR
    print(f"{env.now:.2f} hr: Vessel {vessel_id} starting cargo ops (Est: {cargo_ops_duration:.2f} hrs)")
    remaining_ops_time = cargo_ops_duration
    cargo_ops_weather_delay = 0

    while remaining_ops_time > 0:
        weather = get_weather(env.now, weather_df)
        if weather['wind_speed_knots'] > WIND_MAX_CARGO_KNOTS:
            print(f"{env.now:.2f} hr: Vessel {vessel_id} cargo ops paused (Wind: {weather['wind_speed_knots']:.1f})")
            delay_start = env.now
            yield env.timeout(CHECK_INTERVAL_HOURS) # Wait until next check
            cargo_ops_weather_delay += (env.now - delay_start)
        else:
            # Work for one interval or until finished
            work_time = min(remaining_ops_time, CHECK_INTERVAL_HOURS)
            yield env.timeout(work_time)
            remaining_ops_time -= work_time

    berth_end_time = env.now
    print(f"{env.now:.2f} hr: Vessel {vessel_id} finished cargo ops.")
    log_entry['berth_end_time'] = berth_end_time
    log_entry['cargo_ops_weather_delay_hrs'] = cargo_ops_weather_delay

    # --- 4. Departure ---
    # Simplified: Assume departure checks are similar to entry, immediate unberthing/transit
    # Add departure weather checks if needed
    unberth_transit_duration = 1.5 # Simplified
    yield env.timeout(unberth_transit_duration)
    departure_time = env.now
    print(f"{env.now:.2f} hr: Vessel {vessel_id} departs.")
    log_entry['departure_time'] = departure_time

    # --- Record Results ---
    event_log.append(log_entry)


def arrival_generator(env, resources, weather_df):
    """Generates vessel arrivals according to a Poisson process."""
    vessel_count = 0
    while True:
        # Calculate inter-arrival time (Exponential distribution)
        # Could make arrival_rate vary by time of day/season
        inter_arrival_time = np.random.exponential(1.0 / AVG_ARRIVAL_RATE_PER_HOUR)
        yield env.timeout(inter_arrival_time)

        vessel_count += 1
        # Assign characteristics
        vessel_type = np.random.choice(VESSEL_TYPES, p=VESSEL_PROBS)
        mean_teu = TEU_MEAN[vessel_type]
        vessel_teu = max(100, np.random.normal(mean_teu, mean_teu * TEU_STD_DEV_FACTOR)) # Ensure positive TEU

        characteristics = {'type': vessel_type, 'teu': vessel_teu}

        # Start the vessel process
        env.process(vessel_process(env, f"V_{vessel_count}", characteristics, resources, weather_df))

# --- Main Simulation Execution ---
print("Starting Port Simulation...")

# 1. Generate Weather Data First
weather_df = generate_weather_data('2023-01-01', '2023-03-31') # Generate for longer than sim duration if needed
print("Weather data generated.")
print(weather_df.head())
print(weather_df.describe())
weather_df['state'].value_counts().plot(kind='bar', title='Simulated Weather State Frequency')

# 2. Setup SimPy Environment and Resources
env = simpy.Environment()
port_resources = {
    'berths': simpy.Resource(env, capacity=NUM_BERTHS)
    # Add other resources like 'channel', 'tugs' here if needed
}

# 3. Start Processes
env.process(arrival_generator(env, port_resources, weather_df))

# 4. Run Simulation
env.run(until=SIM_DURATION_HOURS)

print("\nSimulation Finished.")

# 5. Process Results
if event_log:
    operations_df = pd.DataFrame(event_log)
    # Calculate additional metrics
    operations_df['total_time_hrs'] = operations_df['departure_time'] - operations_df['arrival_time']
    operations_df['berth_time_hrs'] = operations_df['berth_end_time'] - operations_df['berth_start_time']
    operations_df['total_weather_delay_hrs'] = (operations_df['weather_delay_entry_hrs'].fillna(0) +
                                                operations_df['weather_delay_berthing_hrs'].fillna(0) +
                                                operations_df['cargo_ops_weather_delay_hrs'].fillna(0))

    print("\nGenerated Operations Log Sample:")
    print(operations_df.head())
    print("\nOperations Log Description:")
    print(operations_df.describe())


    operations_df.to_csv("synthetic_operations_log.csv", index=False)
    # For larger data, consider Parquet:
    # operations_df.to_parquet("synthetic_operations_log.parquet", index=False)
else:
    print("\nNo vessel events were logged.")

# --- Saving Data (Task 2.5 related) ---
weather_df.to_csv("synthetic_weather_data.csv")
    