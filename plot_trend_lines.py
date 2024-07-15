#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.stats import linregress

# Argument parsing
parser = argparse.ArgumentParser(description='Plot archived data with trend lines.')
parser.add_argument('file_path', type=str, nargs='?', help='Path to the CSV file containing the archived data', default="Data-ingested-over-time.csv")
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

# Load the CSV file
data = pd.read_csv(args.file_path)

# Rename columns for easier access
data.rename(columns={'@timestamp per 30 days': 'Date', 'Archive Size': 'ArchivedData'}, inplace=True)

# Ensure the date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

# Remove '.00' and convert ArchivedData to numeric, handling commas and possible issues
data['ArchivedData'] = data['ArchivedData'].str.replace('.00', '', regex=False).str.replace(',', '')
data['ArchivedData'] = pd.to_numeric(data['ArchivedData'], errors='coerce')

# Drop rows with NaN values in 'ArchivedData' or where ArchivedData is zero
data = data.dropna(subset=['ArchivedData'])
data = data[data['ArchivedData'] != 0]

# Sum ArchivedData per month
data['YearMonth'] = data['Date'].dt.to_period('M')
monthly_data = data.groupby('YearMonth')['ArchivedData'].sum().reset_index()
monthly_data['ArchivedDataTB'] = (monthly_data['ArchivedData'] / (1024 ** 4)).round().astype(int)

# Ensure the data is sorted by YearMonth
monthly_data = monthly_data.sort_values(by='YearMonth')

# Function to calculate linear regression for specified periods
def get_trendline_data(data, start_period, end_period):
    """
    Calculate the linear regression parameters for a subset of data 
    between start_period and end_period, inclusive.
    """
    subset = data[(data['YearMonth'] >= start_period) & (data['YearMonth'] <= end_period)]
    x_vals = np.arange(len(subset))
    y_vals = subset['ArchivedDataTB']
    slope, intercept, _, _, _ = linregress(x_vals, y_vals)
    return x_vals, (x_vals * slope + intercept)

# Calculate trend lines
start_2008 = pd.Period('2008-10', 'M')
end_2017 = pd.Period('2017-10', 'M')
start_2018 = pd.Period('2018-06', 'M')
end_2024 = monthly_data['YearMonth'].iloc[-1]  # Use the last period in the data

x_vals_2008_2017, trend_2008_2017 = get_trendline_data(monthly_data, start_2008, end_2017)
x_vals_2018_2024, trend_2018_2024 = get_trendline_data(monthly_data, start_2018, end_2024)

# Adjust original x-values to align with the entire timeline for plotting
x_vals_2008_2017 += monthly_data.index[monthly_data['YearMonth'] == start_2008][0]
x_vals_2018_2024 += monthly_data.index[monthly_data['YearMonth'] == start_2018][0]

# If debug mode is enabled, print the total archived data
if args.debug:
    total_archived_data_tb = monthly_data['ArchivedDataTB'].sum()
    total_archived_data_pb = total_archived_data_tb / 1024
    print(f"Total Archived Data: {total_archived_data_pb} PB")

# Plot the data with trend lines
plt.figure(figsize=(12, 8))
plt.plot(monthly_data['YearMonth'].astype(str), monthly_data['ArchivedDataTB'], linestyle='-', label='Monthly Archived Data')

# Plotting the trend lines
plt.plot(monthly_data['YearMonth'].astype(str).iloc[x_vals_2008_2017], trend_2008_2017, 'r--', label='Trend 2008-2017')
plt.plot(monthly_data['YearMonth'].astype(str).iloc[x_vals_2018_2024], trend_2018_2024, 'g--', label='Trend 2018-2024')

# Set x-axis minor ticks to show even years only from 2008 to 2024
even_years = [str(year) for year in range(2008, 2025, 2)]
plt.xticks(even_years, rotation=45)

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('Archived Data (TB/month)')
plt.title('Monthly Archived Data with Trend Lines')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
