import pandas as pd
import numpy as np
import datetime
import random
import uuid

# --- Configuration ---
NUM_RECORDS = 1000  # Number of vessel port visits to generate
START_DATE = datetime.datetime(2022, 1, 1)
END_DATE = datetime.datetime(2023, 12, 31)
PORT_NAME = "Port Synthetic"

# --- Helper Functions ---

def random_date(start, end):
  """Generate a random datetime between start and end."""
  delta = end - start
  int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
  random_second = random.randrange(int_delta)
  return start + datetime.timedelta(seconds=random_second)

def generate_vessel_details():
  """Generates random vessel type and associated characteristics."""
  vessel_type = random.choice(['Container Ship', 'Tanker', 'Bulk Carrier', 'RoRo', 'General Cargo'])
  loa_mean, loa_std = 150, 50 # Base Length Overall (meters)
  ops_time_mean, ops_time_std = 24, 8 # Base Ops time (hours)

  if vessel_type == 'Container Ship':
    loa_mean, loa_std = 250, 80
    ops_time_mean, ops_time_std = 30, 10
  elif vessel_type == 'Tanker':
    loa_mean, loa_std = 200, 60
    ops_time_mean, ops_time_std = 36, 12
  elif vessel_type == 'Bulk Carrier':
      loa_mean, loa_std = 180, 50
      ops_time_mean, ops_time_std = 40, 15
  elif vessel_type == 'RoRo':
      loa_mean, loa_std = 160, 30
      ops_time_mean, ops_time_std = 12, 4

  loa = max(50, np.random.normal(loa_mean, loa_std)) # Ensure min length
  base_ops_hours = max(4, np.random.normal(ops_time_mean, ops_time_std)) # Ensure min ops time

  # Simple unique ID and name
  vessel_id = str(uuid.uuid4())[:8]
  imo = f"IMO{random.randint(9000000, 9999999)}"
  vessel_name = f"{random.choice(['Sea', 'Ocean', 'Star', 'Lady', 'Pacific', 'Atlantic'])}_{vessel_id}"

  return vessel_id, imo, vessel_name, vessel_type, loa, base_ops_hours

# --- Main Generation Logic ---

data = []
print(f"Generating {NUM_RECORDS} synthetic vessel visit records for {PORT_NAME}...")

for _ in range(NUM_RECORDS):
  vessel_id, imo, name, v_type, loa, base_ops_hrs = generate_vessel_details()

  # 1. Schedule & Arrival
  scheduled_arrival = random_date(START_DATE, END_DATE)
  # Simulate arrival delay (most minor, some significant) - exponential distribution good for this
  arrival_delay_hours = np.random.exponential(scale=2.0) # Avg delay 2 hours
  actual_arrival_anchorage = scheduled_arrival + datetime.timedelta(hours=arrival_delay_hours)

  # 2. Waiting Time at Anchorage
  # Simulate waiting time (again, often short, sometimes long)
  waiting_time_hours = np.random.exponential(scale=8.0) # Avg wait 8 hours
  # Make large ships slightly more likely to wait longer (simple heuristic)
  if loa > 250:
       waiting_time_hours *= np.random.uniform(1.0, 1.5)
  time_berth = actual_arrival_anchorage + datetime.timedelta(hours=waiting_time_hours)

  # 3. Berthing & Operations Start
  # Small delay getting started after berthing
  berthing_delay_hours = np.random.uniform(0.5, 2.0) # 30 mins to 2 hours
  time_start_ops = time_berth + datetime.timedelta(hours=berthing_delay_hours)

  # 4. Operations Duration
  # Use base ops hours generated earlier, maybe add some random variation
  ops_duration_hours = base_ops_hrs * np.random.uniform(0.9, 1.2)
  time_end_ops = time_start_ops + datetime.timedelta(hours=ops_duration_hours)

  # 5. Unberthing & Departure
  # Small delay before unberthing
  unberthing_delay_hours = np.random.uniform(1.0, 3.0) # 1 to 3 hours
  time_unberth = time_end_ops + datetime.timedelta(hours=unberthing_delay_hours)

  # Departure delay (e.g., waiting for tide, pilot)
  departure_delay_hours = np.random.exponential(scale=1.5) # Avg 1.5 hours
  actual_departure = time_unberth + datetime.timedelta(hours=departure_delay_hours)

  # 6. Calculate Metrics
  wait_td = time_berth - actual_arrival_anchorage
  service_td = time_end_ops - time_start_ops
  turnaround_td = time_unberth - time_berth # Time spent at berth

  # Store record
  data.append({
      'Record_ID': str(uuid.uuid4()),
      'Vessel_ID': vessel_id,
      'IMO': imo,
      'Vessel_Name': name,
      'Vessel_Type': v_type,
      'LOA_m': round(loa, 1),
      'Scheduled_Arrival': scheduled_arrival,
      'Actual_Arrival_Anchorage': actual_arrival_anchorage,
      'Time_Berth': time_berth,
      'Time_Start_Ops': time_start_ops,
      'Time_End_Ops': time_end_ops,
      'Time_Unberth': time_unberth,
      'Actual_Departure': actual_departure,
      'Waiting_Time_Hours': round(wait_td.total_seconds() / 3600, 2),
      'Service_Time_Hours': round(service_td.total_seconds() / 3600, 2),
      'Turnaround_Time_Hours': round(turnaround_td.total_seconds() / 3600, 2),
      'Port_Name': PORT_NAME
  })

# --- Create DataFrame ---
df_shipping = pd.DataFrame(data)

# Convert timestamp columns to datetime objects if they aren't already
time_cols = ['Scheduled_Arrival', 'Actual_Arrival_Anchorage', 'Time_Berth',
             'Time_Start_Ops', 'Time_End_Ops', 'Time_Unberth', 'Actual_Departure']
for col in time_cols:
    df_shipping[col] = pd.to_datetime(df_shipping[col])

# --- Display Sample ---
print("\n--- Generated Synthetic Shipping Data Sample ---")
print(df_shipping.head())
print("\n--- Data Info ---")
print(df_shipping.info())
print("\n--- Descriptive Statistics ---")
# Selecting only numeric columns for describe
numeric_cols = df_shipping.select_dtypes(include=np.number).columns
print(df_shipping[numeric_cols].describe())

# --- Optional: Save to CSV ---
output_filename = "synthetic_shipping_data.csv"
df_shipping.to_csv(output_filename, index=False)
print(f"\nData saved to {output_filename}")