import pandas as pd
import os
import sys

def load_and_process_data(outage_path, density_path, forecast_path, start_date, end_date):
    """
    Loads outage, spatial, and forecast data, merging and filtering by date.

    Args:
        outage_path (str): Path to outage CSV.
        density_path (str): Path to spatial features CSV.
        forecast_path (str): Path to forecast CSV.
        start_date (str): Filter start date (inclusive).
        end_date (str): Filter end date (exclusive).

    Returns:
        tuple: (outages_df, forecast_df, town_names)
    """
    print(f"Loading data from {start_date} to {end_date}...")

    # 1. Load and Filter Outages
    if not os.path.exists(outage_path):
        raise FileNotFoundError(f"Outage file not found at: {outage_path}")

    outages_df = pd.read_csv(outage_path, parse_dates=["Date and Time Out"])
    outages_df = outages_df[
        (outages_df["Date and Time Out"] >= start_date) &
        (outages_df["Date and Time Out"] < end_date)
    ].copy()

    # 2. Merge Population Density
    if os.path.exists(density_path):
        density_df = pd.read_csv(density_path)
        outages_df = outages_df.merge(
            density_df[["Town", "pop20_density"]],
            how="left",
            left_on="City/Town",
            right_on="Town"
        )
        # Fill missing density with a default value
        outages_df["pop_density"] = outages_df["pop20_density"].fillna(100)
    else:
        print("Warning: Spatial density file not found. Using default density.")
        outages_df["pop_density"] = 100

    # 3. Load Weather Forecasts
    if not os.path.exists(forecast_path):
        print(f"Warning: Forecast file not found at {forecast_path}.")
        print("Simulation will run without proactive weather data (zeros).")
        forecast_df = pd.DataFrame(columns=["datetime", "town", "forecast_hour", "predicted_prob"])
        town_names = outages_df["City/Town"].unique().tolist()
        return outages_df, forecast_df, town_names

    raw_forecast_df = pd.read_csv(forecast_path)
    # Handle mixed date formats if necessary
    raw_forecast_df["datetime"] = pd.to_datetime(raw_forecast_df["datetime"], format="mixed")

    # Filter by towns present in the outage data
    town_names = outages_df["City/Town"].unique().tolist()
    forecast_df = raw_forecast_df[raw_forecast_df["town"].isin(town_names)]

    # Filter by date range
    forecast_df = forecast_df[
        (forecast_df["datetime"] >= start_date) &
        (forecast_df["datetime"] < end_date)
    ].copy()

    print(f"Data loaded successfully. {len(outages_df)} outages, {len(forecast_df)} forecast records.")
    return outages_df, forecast_df, town_names